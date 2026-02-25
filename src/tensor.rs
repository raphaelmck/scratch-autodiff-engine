#[derive(Debug, Clone)]
pub struct Tensor {
	pub data: Vec<f32>,
	pub shape: Vec <usize>,
}

impl Tensor {
	pub fn new(data: Vec<f32>, shape: Vec<usize>) -> Self {
		asser_eq!(
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
		Tensor { data: vec![0.0; len] shape }
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
