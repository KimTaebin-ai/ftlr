use std::env;
use std::fs;

use ftlr::model::linear_regression::{gradient_descent, Model, TrainConfig};
use ftlr::parse::dataset::parse_dataset;
use ftlr::preprocess::normalize::{denormalize_thetas, MinMax};
use ftlr::viz::plot;

const THETA_PATH: &str = "theta.json";
const SCATTER_PATH: &str = "scatter.png";

fn main() -> Result<(), String> {
    let args: Vec<String> = env::args().collect();
    if args.len() != 2 {
        return Err(format!("usage: {} <data.csv>", args[0]));
    }

    let (xs, ys) = parse_dataset(&args[1])?;

    // 입력/출력 스케일 차이로 손실함수 곡률이 폭발하므로 min-max로 [0,1] 정규화.
    let x_scale = MinMax::from_slice(&xs);
    let y_scale = MinMax::from_slice(&ys);
    let xs_norm = x_scale.normalize_slice(&xs);
    let ys_norm = y_scale.normalize_slice(&ys);

    let config = TrainConfig {
        learning_rate: 1e-5,
        iterations: 100000000,
    };

    let model_norm = gradient_descent(&xs_norm, &ys_norm, &config);

    // θ' (정규화 공간) → θ (원 km/price 공간)
    let (theta0, theta1) = denormalize_thetas(
        model_norm.theta0,
        model_norm.theta1,
        &x_scale,
        &y_scale,
    );

    let model = Model { theta0, theta1 };
    let json = serde_json::to_string_pretty(&model)
        .map_err(|e| format!("failed to serialize model: {e}"))?;
    fs::write(THETA_PATH, json)
        .map_err(|e| format!("failed to write {THETA_PATH}: {e}"))?;

    println!("theta0: {theta0}");
    println!("theta1: {theta1}");
    println!("saved to {THETA_PATH}");

    plot::scatter(&xs, &ys, SCATTER_PATH)
        .map_err(|e| format!("failed to draw {SCATTER_PATH}: {e}"))?;
    println!("scatter saved to {SCATTER_PATH}");

    Ok(())
}
