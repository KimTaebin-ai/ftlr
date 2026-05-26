use std::env;
use std::path::Path;

use ftlr::model::linear_regression::{predict, Model};
use ftlr::utils::dataset::parse_dataset;
use ftlr::utils::THETA_PATH;
use ftlr::viz::plot;

const REGRESSION_PATH: &str = "regression.png";

fn load_model() -> Result<Model, String> {
    // requirement: 학습 전이면 (theta0, theta1)=(0, 0)으로 동작.
    if Path::new(THETA_PATH).exists() {
        Model::load(THETA_PATH)
    } else {
        eprintln!("warning: {THETA_PATH} not found, defaulting to theta0 = 0, theta1 = 0");
        Ok(Model::default())
    }
}

fn main() -> Result<(), String> {
    let args: Vec<String> = env::args().collect();
    if !(2..=3).contains(&args.len()) {
        return Err(format!("usage: {} <km> [data.csv]", args[0]));
    }

    let km: f64 = args[1]
        .parse()
        .map_err(|_| format!("invalid km: `{}` is not a number", args[1]))?;
    if !km.is_finite() || km < 0.0 {
        return Err(format!("km must be a non-negative finite number, got {km}"));
    }

    let model = load_model()?;
    let price = predict(&model, km);

    println!("{price}");

    // data.csv가 함께 주어진 경우, 산점도 + 회귀선 + 예측 지점을 그림
    if let Some(data_path) = args.get(2) {
        let (xs, ys) = parse_dataset(data_path)?;
        plot::regression(&xs, &ys, &model, Some((km, price)), REGRESSION_PATH)
            .map_err(|e| format!("failed to draw {REGRESSION_PATH}: {e}"))?;
        println!("regression chart saved to {REGRESSION_PATH}");
    }

    Ok(())
}
