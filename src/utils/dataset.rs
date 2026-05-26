use std::path::Path;

use csv::ReaderBuilder;
use serde::Deserialize;

const EXPECTED_HEADER: [&str; 2] = ["km", "price"];

#[derive(Debug, Deserialize)]
struct Row {
    km: f64,
    price: f64,
}

pub fn parse_dataset(path: &str) -> Result<(Vec<f64>, Vec<f64>), String> {
    let p = Path::new(path);

    if !p.exists() {
        return Err(format!("file not found: {path}"));
    }
    if !p.is_file() {
        return Err(format!("not a regular file: {path}"));
    }
    match p.extension().and_then(|e| e.to_str()) {
        Some(ext) if ext.eq_ignore_ascii_case("csv") => {}
        _ => return Err(format!("file must have .csv extension: {path}")),
    }

    let mut reader = ReaderBuilder::new()
        .has_headers(true)
        .trim(csv::Trim::All)
        .from_path(p)
        .map_err(|e| format!("failed to open {path}: {e}"))?;

    let headers = reader
        .headers()
        .map_err(|e| format!("failed to read header: {e}"))?;
    let header_cols: Vec<&str> = headers.iter().collect();
    if header_cols.len() != EXPECTED_HEADER.len()
        || header_cols[0] != EXPECTED_HEADER[0]
        || header_cols[1] != EXPECTED_HEADER[1]
    {
        return Err(format!(
            "invalid header: expected `{},{}`, got `{}`",
            EXPECTED_HEADER[0],
            EXPECTED_HEADER[1],
            header_cols.join(",")
        ));
    }

    let mut xs: Vec<f64> = Vec::new();
    let mut ys: Vec<f64> = Vec::new();

    for (idx, result) in reader.deserialize::<Row>().enumerate() {
        // +2: 1행은 헤더, 사용자에게 보이는 줄 번호는 1-base
        let line_no = idx + 2;
        let row = result.map_err(|e| format!("line {line_no}: {e}"))?;

        if !row.km.is_finite() || !row.price.is_finite() {
            return Err(format!("line {line_no}: values must be finite"));
        }

        xs.push(row.km);
        ys.push(row.price);
    }

    if xs.is_empty() {
        return Err("no data rows found".to_string());
    }

    Ok((xs, ys))
}
