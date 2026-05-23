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

    /// v ↦ (v − min) / (max − min) ∈ [0, 1]
    #[inline]
    pub fn normalize(&self, v: f64) -> f64 {
        (v - self.min) / self.range()
    }

    pub fn normalize_slice(&self, values: &[f64]) -> Vec<f64> {
        values.iter().map(|&v| self.normalize(v)).collect()
    }
}

/// 정규화 공간 (x', y') 에서 학습된 (θ₀', θ₁') 을
/// 원 스케일 (x, y) 의 (θ₀, θ₁) 로 환산한다.
///
/// 유도: y' = θ₀' + θ₁' · x' 에  x' = (x − x_min)/x_range,  y' = (y − y_min)/y_range
///   ⇒  y = [y_min + y_range·θ₀' − (y_range·θ₁'/x_range)·x_min]
///        + (y_range·θ₁'/x_range) · x
/// 따라서
///   θ₁ = y_range · θ₁' / x_range
///   θ₀ = y_min + y_range · θ₀' − θ₁ · x_min
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
