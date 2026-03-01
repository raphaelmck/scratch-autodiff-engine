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

	pub fn ndim(&self) -> usize {
		self.shape.len()
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
}
