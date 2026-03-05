/// Classic CartPole-v1 environment
/// State: [x, x_dot, theta, theta_dot]
/// Action: 0 = push left, 1 = push right
/// Done when pole falls >12° or cart leaves ±2.4

pub struct CartPole {
    pub state: [f32; 4],
    pub step_count: usize,
    pub max_steps: usize,
}

impl CartPole {
    // Physics constants (same as OpenAI gym)
    const GRAVITY: f32 = 9.8;
    const CART_MASS: f32 = 1.0;
    const POLE_MASS: f32 = 0.1;
    const TOTAL_MASS: f32 = Self::CART_MASS + Self::POLE_MASS;
    const POLE_HALF_LEN: f32 = 0.5;
    const FORCE_MAG: f32 = 10.0;
    const DT: f32 = 0.02;

    // Termination thresholds
    const X_LIMIT: f32 = 2.4;
    const THETA_LIMIT: f32 = 12.0 * std::f32::consts::PI / 180.0; // 12 degrees

    pub fn new() -> Self {
        CartPole {
            state: [0.0; 4],
            step_count: 0,
            max_steps: 200,
        }
    }

    /// Reset with small random initial state
    pub fn reset(&mut self, rng: &mut impl rand::Rng) -> [f32; 4] {
        self.state = [
            rng.gen_range(-0.05..0.05),
            rng.gen_range(-0.05..0.05),
            rng.gen_range(-0.05..0.05),
            rng.gen_range(-0.05..0.05),
        ];
        self.step_count = 0;
        self.state
    }

    /// Take one step. Returns (next_state, reward, done)
    pub fn step(&mut self, action: usize) -> ([f32; 4], f32, bool) {
        let [x, x_dot, theta, theta_dot] = self.state;

        let force = if action == 1 { Self::FORCE_MAG } else { -Self::FORCE_MAG };

        let cos_theta = theta.cos();
        let sin_theta = theta.sin();

        // Physics (semi-implicit Euler, matches gym)
        let temp = (force + Self::POLE_MASS * Self::POLE_HALF_LEN * theta_dot * theta_dot * sin_theta)
            / Self::TOTAL_MASS;
        let theta_acc = (Self::GRAVITY * sin_theta - cos_theta * temp)
            / (Self::POLE_HALF_LEN * (4.0 / 3.0 - Self::POLE_MASS * cos_theta * cos_theta / Self::TOTAL_MASS));
        let x_acc = temp - Self::POLE_MASS * Self::POLE_HALF_LEN * theta_acc * cos_theta / Self::TOTAL_MASS;

        let x = x + Self::DT * x_dot;
        let x_dot = x_dot + Self::DT * x_acc;
        let theta = theta + Self::DT * theta_dot;
        let theta_dot = theta_dot + Self::DT * theta_acc;

        self.state = [x, x_dot, theta, theta_dot];
        self.step_count += 1;

        let done = x.abs() > Self::X_LIMIT
            || theta.abs() > Self::THETA_LIMIT
            || self.step_count >= self.max_steps;

        let reward = if done && self.step_count < self.max_steps {
            0.0 // failed
        } else {
            1.0 // survived this step
        };

        (self.state, reward, done)
    }
}
