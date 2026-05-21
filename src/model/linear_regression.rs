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
pub fn estimate_price(model: &Model, x: f64) -> f64 {
    model.theta0 + model.theta1 * x
}

// Ordinary least squares 의 closed-form 해. 동일한 회귀 문제를 반복 없이 한 번에 푼다.
//   θ₁ = Σ (xᵢ − x̄)(yᵢ − ȳ) / Σ (xᵢ − x̄)²
//   θ₀ = ȳ − θ₁ · x̄
// 경사하강법 결과가 이 값에 수렴하는지 비교용으로 사용.
pub fn normal_equation(xs: &[f64], ys: &[f64]) -> Model {
    assert_eq!(xs.len(), ys.len(), "X and Y must have the same length");
    assert!(xs.len() >= 2, "normal equation needs at least 2 points");

    let m = xs.len() as f64;
    let mean_x = xs.iter().sum::<f64>() / m;
    let mean_y = ys.iter().sum::<f64>() / m;

    let mut num = 0.0_f64;
    let mut den = 0.0_f64;
    for (&x, &y) in xs.iter().zip(ys.iter()) {
        let dx = x - mean_x;
        num += dx * (y - mean_y);
        den += dx * dx;
    }

    assert!(den > 0.0, "x has zero variance, slope is undefined");
    let theta1 = num / den;
    let theta0 = mean_y - theta1 * mean_x;

    Model { theta0, theta1 }
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
