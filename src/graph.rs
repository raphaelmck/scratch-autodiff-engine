use crate::tensor::Tensor;

pub type NodeId = usize;

/// What operation produced this node
#[derive(Debug, Clone)]
pub enum Op {
	/// No operation, this is a leaf (input of parameter)
	Leaf,
	Add(NodeId, NodeId),
	Mul(NodeId, NodeId),
	MatMul(NodeId, NodeId),
	ReLU(NodeId),
	Log(NodeId),
	Sum(NodeId),
	SumAxis(NodeId, usize),
	Scale(NodeId, f32),
	Neg(NodeId),
}

/// A single node in the graph
#[derive(Debug, Clone)]
pub struct Node {
	pub data: Tensor,
	pub op: Op,
	pub requires_grad: bool,
	pub shape: Vec<usize>,
}

pub struct Graph {
	pub nodes: Vec<Node>,
}

impl Graph {
	pub fn new() -> Self {
		Graph { nodes: Vec::new() }
	}

	pub fn clear(&mut self) {
		self.nodes.clear();
	}

	fn push(&mut self, data: Tensor, op: Op, requires_grad: bool) -> NodeId {
		let shape = data.shape.clone();
		let id = self.nodes.len();
		self.nodes.push(Node { data, op, requires_grad, shape });
		id
	}

	pub fn leaf(&mut self, data: Tensor, requires_grad: bool) -> NodeId {
		self.push(data, Op::Leaf, requires_grad)
	}

	pub fn param(&mut self, data: Tensor) -> NodeId {
		self.leaf(data, true)
	}

	pub fn input(&mut self, data: Tensor) -> NodeId {
		self.leaf(data, false)
	}

	pub fn add(&mut self, a: NodeId, b: NodeId) -> NodeId {
        let data = self.nodes[a].data.add(&self.nodes[b].data);
        let rg = self.nodes[a].requires_grad || self.nodes[b].requires_grad;
        self.push(data, Op::Add(a, b), rg)
    }

    pub fn mul(&mut self, a: NodeId, b: NodeId) -> NodeId {
        let data = self.nodes[a].data.mul(&self.nodes[b].data);
        let rg = self.nodes[a].requires_grad || self.nodes[b].requires_grad;
        self.push(data, Op::Mul(a, b), rg)
    }

    pub fn matmul(&mut self, a: NodeId, b: NodeId) -> NodeId {
        let data = self.nodes[a].data.matmul(&self.nodes[b].data);
        let rg = self.nodes[a].requires_grad || self.nodes[b].requires_grad;
        self.push(data, Op::MatMul(a, b), rg)
    }

    pub fn relu(&mut self, a: NodeId) -> NodeId {
        let data = self.nodes[a].data.relu();
        let rg = self.nodes[a].requires_grad;
        self.push(data, Op::ReLU(a), rg)
    }

    pub fn log(&mut self, a: NodeId) -> NodeId {
        let data = self.nodes[a].data.ln();
        let rg = self.nodes[a].requires_grad;
        self.push(data, Op::Log(a), rg)
    }

    pub fn sum(&mut self, a: NodeId) -> NodeId {
        let data = self.nodes[a].data.sum_all();
        let rg = self.nodes[a].requires_grad;
        self.push(data, Op::Sum(a), rg)
    }

    pub fn sum_axis(&mut self, a: NodeId, axis: usize) -> NodeId {
        let data = self.nodes[a].data.sum_axis(axis);
        let rg = self.nodes[a].requires_grad;
        self.push(data, Op::SumAxis(a, axis), rg)
    }

    pub fn scale(&mut self, a: NodeId, s: f32) -> NodeId {
        let data = self.nodes[a].data.scale(s);
        let rg = self.nodes[a].requires_grad;
        self.push(data, Op::Scale(a, s), rg)
    }

    pub fn neg(&mut self, a: NodeId) -> NodeId {
        let data = self.nodes[a].data.neg();
        let rg = self.nodes[a].requires_grad;
        self.push(data, Op::Neg(a), rg)
    }

    /// Read a node's data
    pub fn data(&self, id: NodeId) -> &Tensor {
        &self.nodes[id].data
    }

    /// Mutably access a node's data (for optimizer weight updates)
    pub fn data_mut(&mut self, id: NodeId) -> &mut Tensor {
        &mut self.nodes[id].data
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_forward_add() {
        let mut g = Graph::new();
        let a = g.param(Tensor::new(vec![1.0, 2.0, 3.0], vec![3]));
        let b = g.param(Tensor::new(vec![4.0, 5.0, 6.0], vec![3]));
        let c = g.add(a, b);
        assert_eq!(g.data(c).data, vec![5.0, 7.0, 9.0]);
    }

    #[test]
    fn test_forward_chain() {
        let mut g = Graph::new();
        let x = g.input(Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]));
        let w = g.param(Tensor::new(vec![0.5, 0.5, 0.5, 0.5], vec![2, 2]));
        let b = g.param(Tensor::new(vec![0.1, 0.2], vec![2]));

        let xw = g.matmul(x, w);          // [2,2] x [2,2] -> [2,2]
        let xw_b = g.add(xw, b);          // [2,2] + [2] -> broadcast!
        let out = g.relu(xw_b);
        let loss = g.sum(out);

        // Just verify shapes are right and it didn't panic
        assert_eq!(g.data(xw).shape, vec![2, 2]);
        assert_eq!(g.data(xw_b).shape, vec![2, 2]);
        assert_eq!(g.data(loss).shape, vec![1]);
        assert!(g.data(loss).data[0] > 0.0);
    }

    #[test]
    fn test_requires_grad_propagation() {
        let mut g = Graph::new();
        let x = g.input(Tensor::new(vec![1.0], vec![1]));   // no grad
        let w = g.param(Tensor::new(vec![2.0], vec![1]));    // grad
        let y = g.mul(x, w);

        assert!(!g.nodes[x].requires_grad);
        assert!(g.nodes[w].requires_grad);
        assert!(g.nodes[y].requires_grad); // inherits from w
    }

    #[test]
    fn test_graph_clear() {
        let mut g = Graph::new();
        let _ = g.param(Tensor::new(vec![1.0], vec![1]));
        assert_eq!(g.nodes.len(), 1);
        g.clear();
        assert_eq!(g.nodes.len(), 0);
    }
}
