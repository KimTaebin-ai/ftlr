use serde::{Deserialize, Serialize};

pub struct TrainConfig {
    pub learning_rate: f64, // alpha
    pub iterations: usize, // epoch
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct Model {
    pub theta0: f64, // intercept
    pub theta1: f64, // slope
}

#[inline]
fn estimate_price(model: &Model, x: f64) -> f64 {
    model.theta0 + model.theta1 * x
}

fn compute_gradients(model: &Model, xs: &[f64], ys: &[f64]) -> (f64, f64) {
    let m = xs.len() as f64;
    let mut sum0 = 0.0_f64;
    let mut sum1 = 0.0_f64;

    for (&x_i, &y_i) in xs.iter().zip(ys.iter()) {
        let residual = estimate_price(model, x_i) - y_i;
        sum0 += residual;
        sum1 += residual * x_i;
    }

    ((1.0 / m) * sum0, (1.0 / m) * sum1)
}

// pub fn mean_squared_error(model: &Model, xs: &[f64], ys: &[f64]) -> f64 {
//     let m = xs.len() as f64;
//     let sum_sq: f64 = xs
//         .iter()
//         .zip(ys.iter())
//         .map(|(&x, &y)| {
//             let residual = estimate_price(model, x) - y;
//             residual * residual
//         })
//         .sum();

//     sum_sq / (2.0 * m)
// }

pub fn gradient_descent(xs: &[f64], ys: &[f64], config: &TrainConfig) -> Model {
    assert_eq!(xs.len(), ys.len(), "X and Y must have the same length");
    assert!(!xs.is_empty(), "training set must not be empty");

    let mut model = Model { theta0: 0.0, theta1: 0.0 };

    for _ in 0..config.iterations {
        let (grad0, grad1) = compute_gradients(&model, xs, ys);

        let tmp_theta0 = config.learning_rate * grad0;
        let tmp_theta1 = config.learning_rate * grad1;

        model.theta0 -= tmp_theta0;
        model.theta1 -= tmp_theta1;
    }

    model
}
