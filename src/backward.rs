use crate::graph::{Graph, NodeId, Op};
use crate::tensor::Tensor;

impl Graph {
	/// Reverse-mode autodiff: compute gradients for all nodes that contributed to `loss_id`
	pub fn backward(&self, loss_id: NodeId) -> Vec<Option<Tensor>> {
		let n = self.nodes.len();

		// Gradient accumulator for each node
		let mut grads: Vec<Option<Tensor>> = vec![None; n];

		// Seed: dL/dL = 1.0
		grads[loss_id] = Some(Tensor::ones(self.nodes[loss_id].shape.clone()));

		// Walk in reverse order (guaranteed topological since we always append nodes and inputs have lower IDs than outputs)
		for id in (0..=loss_id).rev() {
			let grad = match &grads[id] {
				Some(g) => g.clone(),
				None => continue,
			};

			if !self.nodes[id].requires_grad {
				continue;
			}

			match &self.nodes[id].op {
				Op::Leaf => {
					// Nothing to propagate, this is where grad accumulates
				}

				Op::Add(a, b) => {
					// d(a+b)/da = 1, d(a+b)/db = 1
					// Unbroadcast if shapes differ
					let (a, b) = (*a, *b);
					let ga = grad.unbroadcast(&self.nodes[a].shape);
					let gb = grad.unbroadcast(&self.nodes[b].shape);
					accumulate(&mut grads, a, ga);
					accumulate(&mut grads, b, gb);
				}

				Op::Mul(a, b) => {
                    // d(a*b)/da = b, d(a*b)/db = a
                    let a_data = &self.nodes[*a].data;
                    let b_data = &self.nodes[*b].data;
                    let ga = grad.mul(b_data).unbroadcast(&a_data.shape);
                    let gb = grad.mul(a_data).unbroadcast(&b_data.shape);
                    accumulate(&mut grads, *a, ga);
                    accumulate(&mut grads, *b, gb);
                }

				Op::MatMul(a, b) => {
                    // C = A @ B
                    // dL/dA = dL/dC @ B^T
                    // dL/dB = A^T @ dL/dC
                    let a_data = &self.nodes[*a].data;
                    let b_data = &self.nodes[*b].data;
                    let ga = grad.matmul(&b_data.transpose());
                    let gb = a_data.transpose().matmul(&grad);
                    accumulate(&mut grads, *a, ga);
                    accumulate(&mut grads, *b, gb);
                }

                Op::ReLU(a) => {
                    // d(relu(x))/dx = 1 if x > 0, else 0
                    let a_data = &self.nodes[*a].data;
                    let mask_data: Vec<f32> = a_data.data.iter()
                        .map(|&x| if x > 0.0 { 1.0 } else { 0.0 })
                        .collect();
                    let mask = Tensor::new(mask_data, a_data.shape.clone());
                    let ga = grad.mul(&mask);
                    accumulate(&mut grads, *a, ga);
                }

                Op::Log(a) => {
                    // d(ln(x))/dx = 1/x
                    let a_data = &self.nodes[*a].data;
                    let recip_data: Vec<f32> = a_data.data.iter()
                        .map(|&x| 1.0 / x)
                        .collect();
                    let recip = Tensor::new(recip_data, a_data.shape.clone());
                    let ga = grad.mul(&recip);
                    accumulate(&mut grads, *a, ga);
                }

                Op::Sum(a) => {
                    // d(sum(x))/dx_i = 1 for all i
                    // grad is shape [1], broadcast to input shape
                    let ga = grad.broadcast_to(&self.nodes[*a].shape);
                    accumulate(&mut grads, *a, ga);
                }

                Op::SumAxis(a, _axis) => {
                    // grad has shape with dim[axis]=1, broadcast back
                    let ga = grad.broadcast_to(&self.nodes[*a].shape);
                    accumulate(&mut grads, *a, ga);
                }

                Op::Scale(a, s) => {
                    // d(s*x)/dx = s
                    let ga = grad.scale(*s);
                    accumulate(&mut grads, *a, ga);
                }

                Op::Neg(a) => {
                    // d(-x)/dx = -1
                    let ga = grad.neg();
                    accumulate(&mut grads, *a, ga);
                }
			}
		}

		grads
	}
}

/// Add `grad` into the accumulator for node `id`
fn accumulate(grads: &mut [Option<Tensor>], id: NodeId, grad: Tensor) {
	match &mut grads[id] {
		Some(existing) => {
			*existing = existing.add(&grad);
		}
		None => {
			grads[id] = Some(grad);
		}
	}
}

#[cfg(test)]
mod tests {
    use crate::graph::{Graph, NodeId};
    use crate::tensor::Tensor;

    /// Helper: finite-difference gradient check
    /// Computes numerical gradient and compares to analytical
	fn grad_check(
		build_graph: impl Fn(&mut Graph, NodeId) -> crate::graph::NodeId,
		input_data: Vec<f32>,
		input_shape: Vec<usize>,
	) {
		let eps = 1e-3;
		let tol = 1e-2;

		// Analytical gradient
		let mut g = Graph::new();
		let x = g.param(Tensor::new(input_data.clone(), input_shape.clone()));
		let loss = build_graph(&mut g, x);  // pass the NodeId
		let grads = g.backward(loss);
		let analytical = grads[x].as_ref().unwrap().clone();

		// Numerical gradient (central difference)
		let mut numerical = vec![0.0f32; input_data.len()];
		for i in 0..input_data.len() {
			let mut data_plus = input_data.clone();
			let mut data_minus = input_data.clone();
			data_plus[i] += eps;
			data_minus[i] -= eps;

			let mut gp = Graph::new();
			let xp = gp.param(Tensor::new(data_plus, input_shape.clone()));
			let lp = build_graph(&mut gp, xp);

			let mut gm = Graph::new();
			let xm = gm.param(Tensor::new(data_minus, input_shape.clone()));
			let lm = build_graph(&mut gm, xm);

			numerical[i] = (gp.data(lp).data[0] - gm.data(lm).data[0]) / (2.0 * eps);
		}

		// Compare
		for i in 0..input_data.len() {
			let a = analytical.data[i];
			let n = numerical[i];
			let denom = a.abs().max(n.abs()).max(1e-8);
			let rel_err = (a - n).abs() / denom;
			assert!(
				rel_err < tol,
				"grad_check failed at index {}: analytical={}, numerical={}, rel_err={}",
				i, a, n, rel_err
			);
		}
	}

    #[test]
    fn test_backward_add() {
        // L = sum(x + x) = 2*sum(x), dL/dx = 2
        let mut g = Graph::new();
        let x = g.param(Tensor::new(vec![1.0, 2.0, 3.0], vec![3]));
        let y = g.add(x, x);
        let loss = g.sum(y);
        let grads = g.backward(loss);
        assert_eq!(grads[x].as_ref().unwrap().data, vec![2.0, 2.0, 2.0]);
    }

    #[test]
    fn test_backward_mul() {
        // L = sum(x * x) = sum(x^2), dL/dx_i = 2*x_i
        let mut g = Graph::new();
        let x = g.param(Tensor::new(vec![1.0, 2.0, 3.0], vec![3]));
        let y = g.mul(x, x);
        let loss = g.sum(y);
        let grads = g.backward(loss);
        assert_eq!(grads[x].as_ref().unwrap().data, vec![2.0, 4.0, 6.0]);
    }

	#[test]
	fn test_backward_matmul() {
		grad_check(
			|g: &mut Graph, x: NodeId| {
				let w = g.param(Tensor::new(vec![0.5, -0.3, 0.8, 0.1], vec![2, 2]));
				let y = g.matmul(x, w);
				g.sum(y)
			},
			vec![1.0, 2.0, 3.0, 4.0],
			vec![2, 2],
		);
	}

	#[test]
	fn test_backward_relu() {
		grad_check(
			|g: &mut Graph, x: NodeId| {
				let y = g.relu(x);
				g.sum(y)
			},
			vec![-2.0, -0.5, 0.5, 2.0],
			vec![4],
		);
	}

	#[test]
	fn test_backward_log() {
		grad_check(
			|g: &mut Graph, x: NodeId| {
				let y = g.log(x);
				g.sum(y)
			},
			vec![0.5, 1.0, 2.0, 3.0],
			vec![4],
		);
	}

	#[test]
	fn test_backward_broadcast_add() {
		grad_check(
			|g: &mut Graph, x: NodeId| {
				let bias = g.param(Tensor::new(vec![0.1, 0.2, 0.3], vec![3]));
				let y = g.add(x, bias);
				g.sum(y)
			},
			vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
			vec![2, 3],
		);
	}

	#[test]
	fn test_backward_mlp_layer() {
		grad_check(
			|g: &mut Graph, x: NodeId| {
				let w = g.param(Tensor::new(vec![0.1, 0.2, -0.3, 0.4, 0.5, -0.1], vec![3, 2]));
				let b = g.param(Tensor::new(vec![0.01, -0.01], vec![2]));
				let xw = g.matmul(x, w);
				let xw_b = g.add(xw, b);
				let out = g.relu(xw_b);
				g.sum(out)
			},
			vec![1.0, 0.5, -1.0, 2.0, -0.5, 0.3],
			vec![2, 3],
		);
	}
}
