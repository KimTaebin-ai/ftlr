use std::env;
use std::fs;

use ftlr::metrics::regression::{r_squared, rmse};
use ftlr::model::linear_regression::Model;
use ftlr::parse::dataset::parse_dataset;

const THETA_PATH: &str = "theta.json";

fn main() -> Result<(), String> {
    let args: Vec<String> = env::args().collect();
    if args.len() != 2 {
        return Err(format!("usage: {} <data.csv>", args[0]));
    }

    let (xs, ys) = parse_dataset(&args[1])?;

    let content = fs::read_to_string(THETA_PATH).map_err(|_| {
        format!("{THETA_PATH} not found — run `cargo run --bin train -- {}` first", &args[1])
    })?;
    let model: Model = serde_json::from_str(&content)
        .map_err(|e| format!("failed to parse {THETA_PATH}: {e}"))?;

    let r2 = r_squared(&model, &xs, &ys);
    let err = rmse(&model, &xs, &ys);

    println!("R^2  : {r2:.6}    (1=완벽, 0=평균예측 수준, <0=평균보다 못함)");
    println!("RMSE : {err:.4}   (price 단위. 예측이 평균 ±{err:.0} 만큼 빗나감)");

    Ok(())
}
