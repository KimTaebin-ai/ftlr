use std::fs;

use serde::{Deserialize, Serialize};

/// 학습된 모델이 저장되는 기본 경로.
pub const THETA_PATH: &str = "theta.json";

pub struct TrainConfig {
    pub learning_rate: f64, // alpha
    pub iterations: usize,  // epoch
}

#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
pub struct Model {
    pub theta0: f64, // intercept
    pub theta1: f64, // slope
}

impl Model {
    /// JSON 파일에서 모델을 읽어 역직렬화한다.
    pub fn load(path: &str) -> Result<Self, String> {
        let content =
            fs::read_to_string(path).map_err(|e| format!("failed to read {path}: {e}"))?;
        serde_json::from_str(&content).map_err(|e| format!("failed to parse {path}: {e}"))
    }

    /// 모델을 pretty JSON으로 직렬화해 파일에 저장한다.
    pub fn save(&self, path: &str) -> Result<(), String> {
        let json = serde_json::to_string_pretty(self)
            .map_err(|e| format!("failed to serialize model: {e}"))?;
        fs::write(path, json).map_err(|e| format!("failed to write {path}: {e}"))
    }
}

/// 가설 ŷ = θ₀ + θ₁ · x. 정규화 공간과 원 공간 모두에서 쓰는 순수 선형 결합.
#[inline]
pub fn predict(model: &Model, x: f64) -> f64 {
    model.theta0 + model.theta1 * x
}

// Ordinary least squares 의 closed-form 해. 동일한 회귀 문제를 반복 없이 한 번에 푼다.
//   θ₁ = Σ (xᵢ − x̄)(yᵢ − ȳ) / Σ (xᵢ − x̄)²
//   θ₀ = ȳ − θ₁ · x̄
// 경사하강법 결과가 이 값에 수렴하는지 비교용으로 사용.
pub fn normal_equation(xs: &[f64], ys: &[f64]) -> Result<Model, String> {
    assert_eq!(xs.len(), ys.len(), "X and Y must have the same length");
    if xs.len() < 2 {
        return Err("normal equation needs at least 2 points".to_string());
    }

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

    if den <= 0.0 {
        return Err("x has zero variance, slope is undefined".to_string());
    }
    let theta1 = num / den;
    let theta0 = mean_y - theta1 * mean_x;

    Ok(Model { theta0, theta1 })
}

fn compute_gradients(model: &Model, xs: &[f64], ys: &[f64]) -> (f64, f64) {
    let m = xs.len() as f64;
    let mut sum0 = 0.0_f64;
    let mut sum1 = 0.0_f64;

    for (&x_i, &y_i) in xs.iter().zip(ys.iter()) {
        let residual = predict(model, x_i) - y_i;
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

        let step0 = config.learning_rate * grad0;
        let step1 = config.learning_rate * grad1;

        model.theta0 -= step0;
        model.theta1 -= step1;
    }

    model
}
