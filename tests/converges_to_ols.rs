use ftlr::model::linear_regression::{gradient_descent, normal_equation, TrainConfig};
use ftlr::preprocess::normalize::{denormalize_thetas, MinMax};

// 합성 데이터 y = 2x + 5 + 작은 노이즈에 대해
// 정규화-경사하강법 결과가 OLS closed-form 해와 충분히 가까운지 검증.
// 옵티마이저/정규화/역정규화 파이프라인 전체를 가드한다.
#[test]
fn gd_matches_normal_equation_on_synthetic_data() {
    let xs: Vec<f64> = (0..100).map(|i| i as f64).collect();
    let ys: Vec<f64> = xs
        .iter()
        .enumerate()
        .map(|(i, &x)| 2.0 * x + 5.0 + ((i as i64 % 7) as f64 - 3.0) * 0.05)
        .collect();

    let ols = normal_equation(&xs, &ys).expect("normal equation should succeed");

    let x_scale = MinMax::from_slice(&xs).expect("x scale should be valid");
    let y_scale = MinMax::from_slice(&ys).expect("y scale should be valid");
    let xs_norm = x_scale.normalize_slice(&xs);
    let ys_norm = y_scale.normalize_slice(&ys);

    let config = TrainConfig { learning_rate: 0.1, iterations: 20_000 };
    let gd_norm = gradient_descent(&xs_norm, &ys_norm, &config);
    let (gd_t0, gd_t1) = denormalize_thetas(
        gd_norm.theta0,
        gd_norm.theta1,
        &x_scale,
        &y_scale,
    );

    // θ가 충분히 수렴하면 |GD − OLS|는 부동소수점 잡음 수준이어야 함.
    let tol_t0 = 1e-6;
    let tol_t1 = 1e-8;
    assert!(
        (gd_t0 - ols.theta0).abs() < tol_t0,
        "theta0 mismatch: gd={gd_t0}, ols={}, diff={}",
        ols.theta0,
        (gd_t0 - ols.theta0).abs(),
    );
    assert!(
        (gd_t1 - ols.theta1).abs() < tol_t1,
        "theta1 mismatch: gd={gd_t1}, ols={}, diff={}",
        ols.theta1,
        (gd_t1 - ols.theta1).abs(),
    );
}
