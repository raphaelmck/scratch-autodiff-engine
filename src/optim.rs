use crate::graph::{Graph, NodeId};
use crate::tensor::Tensor;

/// Dead-simple SGD: param -= lr * grad
pub fn sgd_step(
    graph: &mut Graph,
    params: &[NodeId],
    grads: &[Option<Tensor>],
    lr: f32,
) {
    for &p in params {
        if let Some(ref grad) = grads[p] {
            let data = &graph.nodes[p].data;
            let updated: Vec<f32> = data.data.iter()
                .zip(&grad.data)
                .map(|(w, g)| w - lr * g)
                .collect();
            graph.nodes[p].data.data = updated;
        }
    }
}
