use crate::model::linear_regression::{predict, Model};

// 평균제곱오차: (1/m) · Σ (ŷᵢ − yᵢ)².
pub fn mse(model: &Model, xs: &[f64], ys: &[f64]) -> Result<f64, String> {
    if xs.len() != ys.len() {
        return Err("X and Y must have the same length".to_string());
    }
    if xs.is_empty() {
        return Err("evaluation set must not be empty".to_string());
    }

    let m = xs.len() as f64;
    let sum_sq: f64 = xs
        .iter()
        .zip(ys.iter())
        .map(|(&x, &y)| {
            let residual = predict(model, x) - y;
            residual * residual
        })
        .sum();

    Ok(sum_sq / m)
}

// √MSE. y와 같은 단위라서 "평균 ±X원 오차"로 직관적 해석 가능.
pub fn rmse(model: &Model, xs: &[f64], ys: &[f64]) -> Result<f64, String> {
    Ok(mse(model, xs, ys)?.sqrt())
}

// 결정계수 R² = 1 − SS_res / SS_tot.
//   SS_res = Σ (yᵢ − ŷᵢ)²
//   SS_tot = Σ (yᵢ − ȳ)²
// 해석:  1 → 완벽,  0 → ȳ로 예측한 baseline과 동급,  <0 → baseline보다도 못함.
pub fn r_squared(model: &Model, xs: &[f64], ys: &[f64]) -> Result<f64, String> {
    if xs.len() != ys.len() {
        return Err("X and Y must have the same length".to_string());
    }
    if ys.is_empty() {
        return Err("evaluation set must not be empty".to_string());
    }

    let m = ys.len() as f64;
    let mean_y = ys.iter().sum::<f64>() / m;

    let mut ss_res = 0.0_f64;
    let mut ss_tot = 0.0_f64;
    for (&x, &y) in xs.iter().zip(ys.iter()) {
        let res = y - predict(model, x);
        let tot = y - mean_y;
        ss_res += res * res;
        ss_tot += tot * tot;
    }

    if ss_tot <= 0.0 {
        return Err("y has zero variance, R² is undefined".to_string());
    }
    Ok(1.0 - ss_res / ss_tot)
}

#[cfg(test)]
mod tests {
    use super::*;

    // 완벽 적합 모델: y = 2x + 1. residual = 0, 따라서 RMSE = 0, R² = 1.
    #[test]
    fn perfect_fit_yields_zero_rmse_and_unit_r_squared() {
        let model = Model { theta0: 1.0, theta1: 2.0 };
        let xs = [0.0, 1.0, 2.0, 3.0, 4.0];
        let ys: Vec<f64> = xs.iter().map(|&x| 1.0 + 2.0 * x).collect();

        assert!(rmse(&model, &xs, &ys).unwrap().abs() < 1e-12);
        assert!((r_squared(&model, &xs, &ys).unwrap() - 1.0).abs() < 1e-12);
    }

    // 손으로 계산한 MSE: 모델 y=0, 실제 [1,2,3] → 평균(1+4+9)/3 = 14/3.
    #[test]
    fn mse_matches_hand_calculation() {
        let model = Model::default();
        let xs = [1.0, 2.0, 3.0];
        let ys = [1.0, 2.0, 3.0];
        assert!((mse(&model, &xs, &ys).unwrap() - 14.0 / 3.0).abs() < 1e-12);
    }

    #[test]
    fn length_mismatch_returns_err() {
        let model = Model::default();
        assert!(mse(&model, &[1.0, 2.0], &[1.0]).is_err());
        assert!(r_squared(&model, &[1.0, 2.0], &[1.0]).is_err());
    }

    #[test]
    fn empty_input_returns_err() {
        let model = Model::default();
        assert!(mse(&model, &[], &[]).is_err());
        assert!(r_squared(&model, &[], &[]).is_err());
    }

    // y가 모두 동일하면 SS_tot = 0이므로 R²는 정의되지 않음.
    #[test]
    fn r_squared_errors_on_zero_variance_y() {
        let model = Model::default();
        let xs = [0.0, 1.0, 2.0];
        let ys = [5.0, 5.0, 5.0];
        assert!(r_squared(&model, &xs, &ys).is_err());
    }
}

