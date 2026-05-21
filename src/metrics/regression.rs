use crate::model::linear_regression::{estimate_price, Model};

// 평균제곱오차: (1/m) · Σ (ŷᵢ − yᵢ)².
pub fn mse(model: &Model, xs: &[f64], ys: &[f64]) -> f64 {
    assert_eq!(xs.len(), ys.len(), "X and Y must have the same length");
    assert!(!xs.is_empty(), "evaluation set must not be empty");

    let m = xs.len() as f64;
    let sum_sq: f64 = xs
        .iter()
        .zip(ys.iter())
        .map(|(&x, &y)| {
            let residual = estimate_price(model, x) - y;
            residual * residual
        })
        .sum();

    sum_sq / m
}

// √MSE. y와 같은 단위라서 "평균 ±X원 오차"로 직관적 해석 가능.
pub fn rmse(model: &Model, xs: &[f64], ys: &[f64]) -> f64 {
    mse(model, xs, ys).sqrt()
}

// 결정계수 R² = 1 − SS_res / SS_tot.
//   SS_res = Σ (yᵢ − ŷᵢ)²
//   SS_tot = Σ (yᵢ − ȳ)²
// 해석:  1 → 완벽,  0 → ȳ로 예측한 baseline과 동급,  <0 → baseline보다도 못함.
pub fn r_squared(model: &Model, xs: &[f64], ys: &[f64]) -> f64 {
    assert_eq!(xs.len(), ys.len(), "X and Y must have the same length");
    assert!(!xs.is_empty(), "evaluation set must not be empty");

    let m = ys.len() as f64;
    let mean_y = ys.iter().sum::<f64>() / m;

    let mut ss_res = 0.0_f64;
    let mut ss_tot = 0.0_f64;
    for (&x, &y) in xs.iter().zip(ys.iter()) {
        let res = y - estimate_price(model, x);
        let tot = y - mean_y;
        ss_res += res * res;
        ss_tot += tot * tot;
    }

    assert!(ss_tot > 0.0, "y has zero variance, R² is undefined");
    1.0 - ss_res / ss_tot
}
