use std::env;

use ftlr::model::linear_regression::Model;
use ftlr::utils::dataset::parse_dataset;
use ftlr::utils::metrics::{r_squared, rmse};
use ftlr::utils::THETA_PATH;

fn main() -> Result<(), String> {
    let args: Vec<String> = env::args().collect();
    if args.len() != 2 {
        return Err(format!("usage: {} <data.csv>", args[0]));
    }

    let (xs, ys) = parse_dataset(&args[1])?;

    let model = Model::load(THETA_PATH)
        .map_err(|e| format!("{e}\nhint: run `cargo run --bin train -- {}` first", &args[1]))?;

    // 학습 데이터로 측정한 in-sample error
    // held-out 평가가 아니므로 일반화 성능(미지 데이터 예측 오차)과는 다름
    let r2 = r_squared(&model, &xs, &ys)?;
    let err = rmse(&model, &xs, &ys)?;

    println!("R^2  : {r2:.6}    (1=완벽, 0=평균예측 수준, <0=평균보다 못함)");
    println!("RMSE : {err:.4}   (price 단위. 예측이 평균 ±{err:.0} 만큼 빗나감)");

    Ok(())
}
