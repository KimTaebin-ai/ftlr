#[derive(Debug, Clone, Copy)]
pub struct MinMax {
    pub min: f64,
    pub max: f64,
}

impl MinMax {
    pub fn from_slice(values: &[f64]) -> Result<Self, String> {
        if values.is_empty() {
            return Err("cannot compute scale of empty slice".to_string());
        }
        let min = values.iter().copied().fold(f64::INFINITY, f64::min);
        let max = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        if max <= min {
            return Err("min == max: all values are identical, cannot normalize".to_string());
        }
        Ok(MinMax { min, max })
    }

    #[inline]
    pub fn range(&self) -> f64 {
        self.max - self.min
    }

    // v -> (v − min) / (max − min) ∈ [0, 1]
    #[inline]
    pub fn normalize(&self, v: f64) -> f64 {
        (v - self.min) / self.range()
    }

    pub fn normalize_slice(&self, values: &[f64]) -> Vec<f64> {
        values.iter().map(|&v| self.normalize(v)).collect()
    }
}

// 정규화 공간 (x', y') 에서 학습된 (θ₀', θ₁') 을
// 원 스케일 (x, y) 의 (θ₀, θ₁) 로 환산한다.
//
// 유도: y' = θ₀' + θ₁' · x' 에  x' = (x − x_min)/x_range,  y' = (y − y_min)/y_range
//   ⇒  y = [y_min + y_range·θ₀' − (y_range·θ₁'/x_range)·x_min]
//        + (y_range·θ₁'/x_range) · x
// 따라서
//   θ₁ = y_range · θ₁' / x_range
//   θ₀ = y_min + y_range · θ₀' − θ₁ · x_min
pub fn denormalize_thetas(
    theta0_norm: f64,
    theta1_norm: f64,
    x_scale: &MinMax,
    y_scale: &MinMax,
) -> (f64, f64) {
    let xr = x_scale.range();
    let yr = y_scale.range();
    let theta1 = yr * theta1_norm / xr;
    let theta0 = y_scale.min + yr * theta0_norm - theta1 * x_scale.min;
    (theta0, theta1)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn from_slice_rejects_empty() {
        assert!(MinMax::from_slice(&[]).is_err());
    }

    // max == min 이면 range = 0 → normalize 시 0/0 발생하므로 생성 단계에서 막아야 함.
    #[test]
    fn from_slice_rejects_identical_values() {
        assert!(MinMax::from_slice(&[3.0, 3.0, 3.0]).is_err());
    }

    #[test]
    fn normalize_maps_endpoints_to_zero_and_one() {
        let scale = MinMax::from_slice(&[10.0, 20.0, 30.0]).unwrap();
        assert!((scale.normalize(10.0) - 0.0).abs() < 1e-12);
        assert!((scale.normalize(30.0) - 1.0).abs() < 1e-12);
        assert!((scale.normalize(20.0) - 0.5).abs() < 1e-12);
    }

    // 정규화 공간에서 학습한 (θ₀', θ₁')을 역변환하면, 원 공간 데이터에 대한
    // 예측이 정규화 전 예측 ŷ = θ₀ + θ₁·x 와 같아야 한다.
    #[test]
    fn denormalize_thetas_roundtrip() {
        let xs = [0.0, 10.0, 20.0, 30.0, 40.0];
        let ys: Vec<f64> = xs.iter().map(|&x| 3.0 * x + 7.0).collect();

        let x_scale = MinMax::from_slice(&xs).unwrap();
        let y_scale = MinMax::from_slice(&ys).unwrap();
        let xs_norm = x_scale.normalize_slice(&xs);
        let ys_norm = y_scale.normalize_slice(&ys);

        // 정규화된 데이터에서 두 점으로 직선 (θ₀', θ₁') 직접 계산.
        let t1_norm = (ys_norm[4] - ys_norm[0]) / (xs_norm[4] - xs_norm[0]);
        let t0_norm = ys_norm[0] - t1_norm * xs_norm[0];

        let (t0, t1) = denormalize_thetas(t0_norm, t1_norm, &x_scale, &y_scale);

        assert!((t0 - 7.0).abs() < 1e-9, "theta0: expected 7, got {t0}");
        assert!((t1 - 3.0).abs() < 1e-9, "theta1: expected 3, got {t1}");
    }
}

