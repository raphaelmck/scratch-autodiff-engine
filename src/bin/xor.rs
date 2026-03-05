use autodiff::graph::Graph;
use autodiff::tensor::Tensor;
use autodiff::optim::sgd_step;
use rand::Rng;

fn main() {
    let mut rng = rand::thread_rng();

    // XOR dataset
    // inputs: [4, 2], targets: [4, 1]
    let x_data = Tensor::new(
        vec![0.0, 0.0,
             0.0, 1.0,
             1.0, 0.0,
             1.0, 1.0],
        vec![4, 2],
    );
    let targets = vec![0.0, 1.0, 1.0, 0.0];

    // Initialize weights (small random)
    let mk_rand = |n: usize, rng: &mut rand::rngs::ThreadRng| -> Vec<f32> {
        (0..n).map(|_| rng.gen_range(-0.5..0.5)).collect()
    };

    // Layer 1: [2, 8]
    let mut w1_data = mk_rand(2 * 8, &mut rng);
    let mut b1_data = mk_rand(8, &mut rng);
    // Layer 2: [8, 1]
    let mut w2_data = mk_rand(8 * 1, &mut rng);
    let mut b2_data = mk_rand(1, &mut rng);

    let lr = 0.1;
    let epochs = 1000;

    for epoch in 0..epochs {
        // Build a fresh graph each forward pass
        let mut g = Graph::new();

        // Inputs (no grad needed)
        let x = g.input(x_data.clone());

        // Parameters
        let w1 = g.param(Tensor::new(w1_data.clone(), vec![2, 8]));
        let b1 = g.param(Tensor::new(b1_data.clone(), vec![8]));
        let w2 = g.param(Tensor::new(w2_data.clone(), vec![8, 1]));
        let b2 = g.param(Tensor::new(b2_data.clone(), vec![1]));

        // Forward: relu(x @ w1 + b1) @ w2 + b2
        let h = g.matmul(x, w1);       // [4, 8]
        let h = g.add(h, b1);          // [4, 8] + [8] broadcast
        let h = g.relu(h);             // [4, 8]
        let out = g.matmul(h, w2);     // [4, 1]
        let out = g.add(out, b2);      // [4, 1] + [1] broadcast

        // MSE loss: mean((out - target)^2)
        let t = g.input(Tensor::new(targets.clone(), vec![4, 1]));
        let neg_t = g.neg(t);
        let diff = g.add(out, neg_t);          // out - target
        let sq = g.mul(diff, diff);            // (out - target)^2
        let loss = g.sum(sq);                  // sum
        let loss = g.scale(loss, 0.25);        // mean (divide by 4)

        // Backward
        let grads = g.backward(loss);

        // Print loss every 100 epochs
        if epoch % 100 == 0 {
            println!("epoch {:4}  loss: {:.6}", epoch, g.data(loss).data[0]);
        }

        // SGD step — update our local copies
        sgd_step(&mut g, &[w1, b1, w2, b2], &grads, lr);

        // Copy updated weights back out for next iteration
        w1_data = g.nodes[w1].data.data.clone();
        b1_data = g.nodes[b1].data.data.clone();
        w2_data = g.nodes[w2].data.data.clone();
        b2_data = g.nodes[b2].data.data.clone();
    }

    // Final predictions
    let mut g = Graph::new();
    let x = g.input(x_data.clone());
    let w1 = g.param(Tensor::new(w1_data, vec![2, 8]));
    let b1 = g.param(Tensor::new(b1_data, vec![8]));
    let w2 = g.param(Tensor::new(w2_data, vec![8, 1]));
    let b2 = g.param(Tensor::new(b2_data, vec![1]));

    let h = g.matmul(x, w1);
    let h = g.add(h, b1);
    let h = g.relu(h);
    let out = g.matmul(h, w2);
    let out = g.add(out, b2);

    println!("\nPredictions:");
    println!("  0 XOR 0 = {:.4}  (target 0)", g.data(out).data[0]);
    println!("  0 XOR 1 = {:.4}  (target 1)", g.data(out).data[1]);
    println!("  1 XOR 0 = {:.4}  (target 1)", g.data(out).data[2]);
    println!("  1 XOR 1 = {:.4}  (target 0)", g.data(out).data[3]);
}
