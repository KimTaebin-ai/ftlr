use std::env;
use std::fs;

use ftlr::model::linear_regression::Model;

const THETA_PATH: &str = "theta.json";

fn load_model() -> Result<Model, String> {
    // requirement: 학습 전이면 (theta0, theta1)=(0, 0)으로 동작.
    match fs::read_to_string(THETA_PATH) {
        Ok(content) => serde_json::from_str::<Model>(&content)
            .map_err(|e| format!("failed to parse {THETA_PATH}: {e}")),
        Err(_) => {
            eprintln!("warning: {THETA_PATH} not found, defaulting to theta0=0, theta1=0");
            Ok(Model { theta0: 0.0, theta1: 0.0 })
        }
    }
}

fn main() -> Result<(), String> {
    let args: Vec<String> = env::args().collect();
    if args.len() != 2 {
        return Err(format!("usage: {} <km>", args[0]));
    }

    let km: f64 = args[1]
        .parse()
        .map_err(|_| format!("invalid km: `{}` is not a number", args[1]))?;
    if !km.is_finite() || km < 0.0 {
        return Err(format!("km must be a non-negative finite number, got {km}"));
    }

    let model = load_model()?;
    let price = model.theta0 + model.theta1 * km;

    println!("{price}");

    Ok(())
}
