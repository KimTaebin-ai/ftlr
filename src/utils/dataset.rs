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

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicU64, Ordering};

    // 테스트 종료 시 자동 정리되는 임시 CSV. 외부 crate 의존 없이 사용
    struct TempCsv(PathBuf);

    impl TempCsv {
        fn new(ext: &str, content: &str) -> Self {
            static COUNTER: AtomicU64 = AtomicU64::new(0);
            let n = COUNTER.fetch_add(1, Ordering::SeqCst);
            let path = std::env::temp_dir().join(format!(
                "ftlr_test_{}_{}.{ext}",
                std::process::id(),
                n
            ));
            std::fs::write(&path, content).unwrap();
            Self(path)
        }
        fn path(&self) -> &str {
            self.0.to_str().unwrap()
        }
    }

    impl Drop for TempCsv {
        fn drop(&mut self) {
            let _ = std::fs::remove_file(&self.0);
        }
    }

    #[test]
    fn parses_well_formed_csv() {
        let f = TempCsv::new("csv", "km,price\n10,100\n20,200\n30,300\n");
        let (xs, ys) = parse_dataset(f.path()).unwrap();
        assert_eq!(xs, vec![10.0, 20.0, 30.0]);
        assert_eq!(ys, vec![100.0, 200.0, 300.0]);
    }

    #[test]
    fn rejects_missing_file() {
        assert!(parse_dataset("/definitely/does/not/exist.csv").is_err());
    }

    #[test]
    fn rejects_non_csv_extension() {
        let f = TempCsv::new("txt", "km,price\n10,100\n");
        assert!(parse_dataset(f.path()).is_err());
    }

    #[test]
    fn rejects_wrong_header() {
        let f = TempCsv::new("csv", "mileage,price\n10,100\n");
        let err = parse_dataset(f.path()).unwrap_err();
        assert!(err.contains("header"), "unexpected error: {err}");
    }

    #[test]
    fn rejects_empty_body() {
        let f = TempCsv::new("csv", "km,price\n");
        assert!(parse_dataset(f.path()).is_err());
    }

    #[test]
    fn rejects_non_finite_values() {
        let f = TempCsv::new("csv", "km,price\n10,100\nNaN,200\n");
        assert!(parse_dataset(f.path()).is_err());
    }
}

