use std::fs;

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy)]
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
    // JSON 파일에서 모델을 읽어 역직렬화한다.
    pub fn load(path: &str) -> Result<Self, String> {
        let content =
            fs::read_to_string(path).map_err(|e| format!("failed to read {path}: {e}"))?;
        serde_json::from_str(&content).map_err(|e| format!("failed to parse {path}: {e}"))
    }

    // 모델을 pretty JSON으로 직렬화해 파일에 저장한다.
    pub fn save(&self, path: &str) -> Result<(), String> {
        let json = serde_json::to_string_pretty(self)
            .map_err(|e| format!("failed to serialize model: {e}"))?;
        fs::write(path, json).map_err(|e| format!("failed to write {path}: {e}"))
    }
}

// 가설 ŷ = θ₀ + θ₁ · x. 정규화 공간과 원 공간 모두에서 쓰는 순수 선형 결합.
#[inline]
pub fn predict(model: &Model, x: f64) -> f64 {
    model.theta0 + model.theta1 * x
}

// Ordinary least squares 의 closed-form 해. 동일한 회귀 문제를 반복 없이 한 번에 푼다.
//   θ₁ = Σ (xᵢ − x̄)(yᵢ − ȳ) / Σ (xᵢ − x̄)²
//   θ₀ = ȳ − θ₁ · x̄
// 경사하강법 결과가 이 값에 수렴하는지 비교용으로 사용.
pub fn normal_equation(xs: &[f64], ys: &[f64]) -> Result<Model, String> {
    if xs.len() != ys.len() {
        return Err("X and Y must have the same length".to_string());
    }
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

pub fn gradient_descent(
    xs: &[f64],
    ys: &[f64],
    config: &TrainConfig,
) -> Result<Model, String> {
    if xs.len() != ys.len() {
        return Err("X and Y must have the same length".to_string());
    }
    if xs.is_empty() {
        return Err("training set must not be empty".to_string());
    }

    let mut model = Model { theta0: 0.0, theta1: 0.0 };

    for _ in 0..config.iterations {
        let (grad0, grad1) = compute_gradients(&model, xs, ys);

        let step0 = config.learning_rate * grad0;
        let step1 = config.learning_rate * grad1;

        model.theta0 -= step0;
        model.theta1 -= step1;
    }

    // 학습률이 과하거나 데이터 스케일이 어긋나면 θ가 NaN/Inf로 발산할 수 있음
    if !model.theta0.is_finite() || !model.theta1.is_finite() {
        return Err(format!(
            "gradient descent diverged: theta=({}, {}). try a smaller learning_rate",
            model.theta0, model.theta1
        ));
    }

    Ok(model)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn predict_is_linear_combination() {
        let model = Model { theta0: 2.0, theta1: 3.0 };
        assert_eq!(predict(&model, 0.0), 2.0);
        assert_eq!(predict(&model, 1.0), 5.0);
        assert_eq!(predict(&model, -1.0), -1.0);
    }

    // y = 2x + 5 노이즈 없는 정합 → OLS는 정확히 (5, 2)를 복원해야 함.
    #[test]
    fn normal_equation_recovers_exact_line() {
        let xs: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let ys: Vec<f64> = xs.iter().map(|&x| 2.0 * x + 5.0).collect();
        let m = normal_equation(&xs, &ys).unwrap();
        assert!((m.theta0 - 5.0).abs() < 1e-9);
        assert!((m.theta1 - 2.0).abs() < 1e-9);
    }

    #[test]
    fn normal_equation_rejects_length_mismatch_and_too_few_points() {
        assert!(normal_equation(&[1.0, 2.0], &[1.0]).is_err());
        assert!(normal_equation(&[1.0], &[1.0]).is_err());
    }

    // 모든 x가 동일하면 기울기가 정의되지 않으므로 Err.
    #[test]
    fn normal_equation_errors_on_zero_variance_x() {
        let xs = [3.0, 3.0, 3.0];
        let ys = [1.0, 2.0, 3.0];
        assert!(normal_equation(&xs, &ys).is_err());
    }

    #[test]
    fn gradient_descent_rejects_invalid_input() {
        let cfg = TrainConfig { learning_rate: 0.1, iterations: 10 };
        assert!(gradient_descent(&[], &[], &cfg).is_err());
        assert!(gradient_descent(&[1.0, 2.0], &[1.0], &cfg).is_err());
    }

    // 정규화 안 된 큰 스케일 + 큰 학습률이면 θ가 발산. 종료 가드가 잡아내야 함.
    #[test]
    fn gradient_descent_detects_divergence() {
        let xs: Vec<f64> = (0..10).map(|i| (i * 1000) as f64).collect();
        let ys: Vec<f64> = xs.iter().map(|&x| 2.0 * x + 5.0).collect();
        let cfg = TrainConfig { learning_rate: 1.0, iterations: 100 };
        let result = gradient_descent(&xs, &ys, &cfg);
        assert!(result.is_err(), "expected divergence error, got {result:?}");
    }
}

