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
