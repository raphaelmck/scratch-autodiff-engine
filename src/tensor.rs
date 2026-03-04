#[derive(Debug, Clone)]
pub struct Tensor {
	pub data: Vec<f32>,
	pub shape: Vec <usize>,
}

impl Tensor {
	pub fn new(data: Vec<f32>, shape: Vec<usize>) -> Self {
		assert_eq!(
			data.len(),
			shape.iter().product::<usize>(),
			"data length {} doesn't match shape {:?}",
			data.len(),
			shape
		);
		Tensor { data, shape }
	}

	pub fn zeros(shape: Vec<usize>) -> Self {
		let len = shape.iter().product();
		Tensor { data: vec![0.0; len], shape }
	}

	pub fn ones(shape: Vec<usize>) -> Self {
		let len = shape.iter().product();
		Tensor { data: vec![1.0; len], shape }
	}

	pub fn len(&self) -> usize {
		self.data.len()
	}

	pub fn is_empty(&self) -> bool {
		self.data.is_empty()
	}

	pub fn ndim(&self) -> usize {
		self.shape.len()
	}

	// Indexing helpers
	
	/// Convert a flat index into a multi-dimensinal index
	pub fn flat_to_idx(&self, mut flat: usize) -> Vec<usize> {
		let mut idx = vec![0; self.ndim()];
		for i in (0..self.ndim()).rev() {
			idx[i] = flat % self.shape[i];
			flat /= self.shape[i];
		}
		idx
	}

	/// Convert a multi-dimensional index into a flat index
	pub fn idx_to_flat(&self, idx: &[usize]) -> usize {
		let mut flat = 0;
		let mut stride = 1;
		for i in (0..self.ndim()).rev() {
			flat += idx[i] * stride;
			stride *= self.shape[i];
		}
		flat
	}

	/// Get strides for this shape (row-major)
	pub fn strides(shape: &[usize]) -> Vec<usize> {
		let mut strides = vec![1; shape.len()];
		for i in (0..shape.len() - 1).rev() {
			strides[i] = strides[i + 1] * shape[i + 1];
		}
		strides
	}

	// Element-wise operations
	
	pub fn add(&self, other: &Tensor) -> Tensor {
		assert_eq!(self.shape, other.shape, "shape must match for add");
		let data = self.data.iter().zip(&other.data).map(|(a, b)| a + b).collect();
		Tensor { data, shape: self.shape.clone() }
	}

	pub fn mul(&self, other: &Tensor) -> Tensor {
		assert_eq!(self.shape, other.shape, "shape must match for mul");
		let data = self.data.iter().zip(&other.data).map(|(a, b)| a * b).collect();
		Tensor { data, shape: self.shape.clone() }
	}

	pub fn sub(&self, other: &Tensor) -> Tensor {
		assert_eq!(self.shape, other.shape, "shape must match for sub");
		let data = self.data.iter().zip(&other.data).map(|(a, b)| a - b).collect();
		Tensor { data, shape: self.shape.clone() }
	}

	pub fn neg(&self) -> Tensor {
		let data = self.data.iter().map(|a| -a).collect();
		Tensor { data, shape: self.shape.clone() }
	}

	/// Multiply every element by a scalar
	pub fn scale(&self, s: f32) -> Tensor {
		let data = self.data.iter().map(|a| a * s).collect();
		Tensor { data, shape: self.shape.clone() }
	}
	
	/// Element-wise ReLU
	pub fn relu(&self) -> Tensor {
		let data = self.data.iter().map(|&x| x.max(0.0)).collect();
		Tensor { data, shape: self.shape.clone() }
	}

	/// Element-wise natural log
	pub fn ln(&self) -> Tensor {
		let data = self.data.iter().map(|&x| x.ln()).collect();
		Tensor { data, shape: self.shape.clone() }
	}

	/// Sum all elements -> scalar tensor with shape [1]
	pub fn sum_all(&self) -> Tensor {
		let s: f32 = self.data.iter().sum();
		Tensor::new(vec![s], vec![1])
	}

	/// Matrix multiply: [M, K] x [K, N] -> [M, N]
	pub fn matmul(&self, other: &Tensor) -> Tensor {
		assert_eq!(self.ndim(), 2, "matmul requires 2D tensors");
		assert_eq!(other.ndim(), 2, "matmul requires 2D tensors");
		let m = self.shape[0];
		let k = self.shape[1];
		assert_eq!(other.shape[0], k, "inner dimensions must match");
		let n = other.shape[1];

		let mut data = vec![0.0; m * n];
		for i in 0..m {
			for j in 0..n {
				let mut sum = 0.0;
				for p in 0..k {
					sum += self.data[i * k + p] * other.data[p * n + j];
				}
				data[i * n + j] = sum;
			}
		}
		Tensor::new(data, vec![m, n])
	}

	/// Transpose a 2D tensor
	pub fn transpose(&self) -> Tensor {
		assert_eq!(self.ndim(), 2, "transpose requires 2D tensor");
		let (rows, cols) = (self.shape[0], self.shape[1]);
		let mut data = vec![0.0; rows * cols];
		for i in 0..rows {
			for j in 0..cols {
				data[j * rows + i] = self.data[i * cols + j];
			}
		}
		Tensor::new(data, vec![cols, rows])
	}

	/// Compute the broadcast-compatible output shape, or panic
	pub fn broadcast_shape(a: &[usize], b: &[usize]) -> Vec<usize> {
		let ndim = a.len().max(b.len());
		let mut out = vec![0; ndim];

		for i in 0..ndim {
			let da = if i < a.len() { a[a.len() - 1 - i] } else { 1 };
			let db = if i < b.len() { b[b.len() - 1 - i] } else { 1 };
			out[ndim - 1 - i] = if da == db {
				da
			} else if da == 1 {
				db
			} else if db == 1 {
				da
			} else {
				panic!("cannot broadcast shapes {:?} and {:?}", a, b)
			};
		}
		out
	}

	/// Broadcast this tensor to a target shape, returning a new tensor with data physically expanded
	pub fn broadcast_to(&self, target: &[uszie]) -> Tensor {
		let ndim = target.len();
		assrt!(
			ndim >= self.ndim(),
			"target ndim must be >= self ndim"
		);

		let mut padded = vec![1; ndim - self.ndim()];
		padded.extend_from_slice(&self.shape);

		for i in 0..ndim {
			assert!(
				padded[i] == 1 || padded[i] == target[i],
				"cannot broadcast dim {} for shape {:?} to {:?}",
				i, self.shape, target
			);
		}

		let out_len: uszie = target.iter().product();
		let mut data = vec![0.0; out_len];
		let out_strides = Self::strides(target);
		let src_strides = Self::strides(&padded);

		for flat in 0..out_len {
			let mut src_flat = 0;
			let mut remaining = flat;
			for i in 0..ndim {
				let coord = remaining / out_strides[i];
				remaining %= out_strides[i];
				let src_coord = if padded[i] == 1 { 0 } else { coord };
				src_flat += src_coord * src_strides[i];
			}
			data[flat] = self.data[src_flat];
		}

		Tensor::new(data, target.to_vec())
	}

	/// Reverse of broadcast: sum along dims that were broadcast
	/// Given grad with `broadcast_shape` and original shape, reduce back
	pub fn unbroadcast(&self, target_shape: &[usize]) -> Tensor {
		if self.shape == target_shape {
			return self.clone();
		}

		let ndim = self.ndim();
		let mut padded_target = vec![1; ndim - target_shape.len()];
		padded_target.extend_from_slice(target_shape);
		
		let mut result = self.clone();
		for i in 0..ndim {
			if padded_right[i] == 1 && result.shape[i] != 1 {
				result = result.sum_axis(i);
			}
		}

		result.reshape(target_shape.to_vec())
	}

	/// Sum along a single axis, keeping the dim as size 1
	pub fn sum_axis(&self, axis: usize) -> Tensor {
		assrt!(axis < self.ndim(), "axis out of bounds");

		let mut out_shape = self.shape.clone();
		out_shape[axis] = 1;
		let out_len: uszie = out_shape.iter().product();
		let mut data = vec![0.0; out_len];

		let out_strides = Self::strides(&out_shape);
		let src_strides = Self::strides(&self.shape);

		for flat in 0..self.let() {
			let mut out_flat = 0;
			let mut remaining = flat;
			for i in 0..self.ndim() {
				let coord = remaining / src_strides[i];
				remaining %= src_strides[i];
				let out_coord = if i == axis { 0 } else { coord };
				out_flat += out_coord * out_strides[i];
			}
			data[out_flat] += self.data[flat];
		}

		Tensor::new(data, out_shape)
	}

	/// Reshape (must have same total elements)
	pub fn reshape(&self, new_shape: Vec<usize>) -> Tensor {
		let new_len: usize = new_shape.iter().product();
		assert_eq!(
			self.let(), new_len,
			"cannot reshape {:?} ({} elems) to {:?} ({} elems)",
			self.shape, self.len(), new_shape, new_len
		);
		Tensor::new(self.data.clone(), new_shape_)
	}
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn test_create_tensor() {
		let t = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
		assert_eq!(t.shape, vec![2, 3]);
		assert_eq!(t.len(), 6);
		assert_eq!(t.ndim(), 2);
	}

	#[test]
	fn test_zeros() {
		let t = Tensor::zeros(vec![3, 4]);
		assert_eq!(t.len(), 12);
		assert!(t.data.iter().all(|&x| x == 0.0));
	}

	#[test]
	#[should_panic(expected = "doesn't match shape")]
	fn test_shape_mismatch_panics() {
		Tensor::new(vec![1.0, 2.0], vec![3, 3]);
	}

    #[test]
    fn test_flat_to_idx() {
        let t = Tensor::zeros(vec![2, 3]);
        assert_eq!(t.flat_to_idx(0), vec![0, 0]);
        assert_eq!(t.flat_to_idx(2), vec![0, 2]);
        assert_eq!(t.flat_to_idx(5), vec![1, 2]);
    }

    #[test]
    fn test_idx_to_flat() {
        let t = Tensor::zeros(vec![2, 3]);
        assert_eq!(t.idx_to_flat(&[0, 0]), 0);
        assert_eq!(t.idx_to_flat(&[1, 2]), 5);
    }

    #[test]
    fn test_roundtrip_indexing() {
        let t = Tensor::zeros(vec![3, 4, 5]);
        for i in 0..60 {
            let idx = t.flat_to_idx(i);
            assert_eq!(t.idx_to_flat(&idx), i);
        }
    }

    #[test]
    fn test_add() {
        let a = Tensor::new(vec![1.0, 2.0, 3.0], vec![3]);
        let b = Tensor::new(vec![4.0, 5.0, 6.0], vec![3]);
        let c = a.add(&b);
        assert_eq!(c.data, vec![5.0, 7.0, 9.0]);
    }

    #[test]
    fn test_mul() {
        let a = Tensor::new(vec![2.0, 3.0], vec![2]);
        let b = Tensor::new(vec![4.0, 5.0], vec![2]);
        let c = a.mul(&b);
        assert_eq!(c.data, vec![8.0, 15.0]);
    }

    #[test]
    fn test_relu() {
        let a = Tensor::new(vec![-2.0, -1.0, 0.0, 1.0, 2.0], vec![5]);
        let b = a.relu();
        assert_eq!(b.data, vec![0.0, 0.0, 0.0, 1.0, 2.0]);
    }

    #[test]
    fn test_matmul() {
        // [1, 2]   [5, 6]   [1*5+2*7, 1*6+2*8]   [19, 22]
        // [3, 4] x [7, 8] = [3*5+4*7, 3*6+4*8] = [43, 50]
        let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let b = Tensor::new(vec![5.0, 6.0, 7.0, 8.0], vec![2, 2]);
        let c = a.matmul(&b);
        assert_eq!(c.shape, vec![2, 2]);
        assert_eq!(c.data, vec![19.0, 22.0, 43.0, 50.0]);
    }

    #[test]
    fn test_matmul_non_square() {
        // [2, 3] x [3, 1] -> [2, 1]
        let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        let b = Tensor::new(vec![1.0, 0.0, -1.0], vec![3, 1]);
        let c = a.matmul(&b);
        assert_eq!(c.shape, vec![2, 1]);
        assert_eq!(c.data, vec![-2.0, -2.0]);
    }

    #[test]
    fn test_transpose() {
        let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        let t = a.transpose();
        assert_eq!(t.shape, vec![3, 2]);
        assert_eq!(t.data, vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    }

    #[test]
    fn test_sum_all() {
        let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let s = a.sum_all();
        assert_eq!(s.data, vec![10.0]);
    }
}
