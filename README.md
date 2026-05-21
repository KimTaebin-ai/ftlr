# ft_linear_regression

km 값으로 중고차 가격을 예측하는 단순 선형회귀 모델.
경사하강법(gradient descent)으로 직접 학습한다.

## 요구사항

- Rust 1.70+ (edition 2021)
- `data.csv` (헤더: `km,price`)

## 빌드

```sh
cargo build --release
```

## 실행 방법

세 개의 바이너리가 `src/bin/` 아래에 있다.

### 1. 학습 — `train`

CSV를 읽어 경사하강법으로 θ₀, θ₁을 구하고 `theta.json`에 저장한다.

```sh
cargo run --bin train -- data.csv
```

출력 예시:

```
theta0: 8499.59964940584
theta1: -0.021448963586739037
saved to theta.json
scatter saved to scatter.png
```

내부 동작:

1. `data.csv` 파싱
2. x, y 각각 min-max [0, 1] 정규화
3. 경사하강법으로 학습
4. 학습된 θ′를 원 스케일(km/price)의 θ로 역변환
5. `theta.json`에 JSON으로 저장
6. 학습 데이터 산점도를 `scatter.png`로 저장

> 정규화하지 않으면 헤시안 λ_max ≈ 1.3 × 10¹⁰이라 학습률을 `1e-10` 수준으로 낮춰야 발산을 막을 수 있다 (그래도 수렴이 느림). 정규화 후엔 일반적인 `α = 0.1`로 빠르게 수렴.

### 2. 예측 — `predict`

km 값을 입력 받아 예측 가격을 출력한다.

```sh
cargo run --bin predict -- 100000
# 6353.800889944063
```

`theta.json`이 없으면 (θ₀, θ₁) = (0, 0)으로 동작하여 0을 반환 (학습 전 동작 규약).

`data.csv`를 두 번째 인자로 함께 주면 **산점도 + 회귀선 + 예측 지점**을 `regression.png`로 저장:

```sh
cargo run --bin predict -- 100000 data.csv
# 6353.800889944063
# regression chart saved to regression.png
```

### 3. 정확도 — `precision`

학습된 모델을 데이터셋에 대해 평가. R²와 RMSE를 출력.

```sh
cargo run --bin precision -- data.csv
```

출력 예시:

```
R^2  : 0.732975    (1=완벽, 0=평균예측 수준, <0=평균보다 못함)
RMSE : 667.5667   (price 단위. 예측이 평균 ±668 만큼 빗나감)
```

- **R²**: 결정계수. y 분산 중 모델이 설명하는 비율. 회귀 정확도의 표준 지표.
- **RMSE**: √평균제곱오차. y와 같은 단위라서 "평균 얼마나 빗나가나"로 해석.

## 테스트

```sh
cargo test
```

`tests/converges_to_ols.rs`가 합성 데이터에서 **경사하강법 결과가 OLS closed-form 해와 일치**하는지 검증 (옵티마이저/정규화/역정규화 전체 파이프라인 가드).

## 디렉토리 구조

```
src/
├── lib.rs                        # 모듈 노출
├── bin/
│   ├── train.rs                  # 학습 바이너리
│   ├── predict.rs                # 예측 바이너리
│   └── precision.rs              # 정확도 평가 바이너리
├── model/
│   └── linear_regression.rs      # gradient_descent, normal_equation, Model
├── metrics/
│   └── regression.rs             # mse, rmse, r_squared
├── parse/
│   └── dataset.rs                # CSV 파싱 (csv 크레이트)
├── preprocess/
│   └── normalize.rs              # MinMax, denormalize_thetas
└── viz/
    └── plot.rs                   # scatter / regression 차트 (plotters 크레이트)
tests/
└── converges_to_ols.rs           # GD vs OLS 회귀 테스트
```

## 의존성

- `serde`, `serde_json` — `theta.json` 직렬화
- `csv` — RFC 4180 준수 CSV 파싱
- `plotters` — `scatter.png` / `regression.png` 시각화 (PNG)

## 알고리즘 요약

가설:

```
ŷ = θ₀ + θ₁ · x
```

손실함수 (MSE 형태, 2로 나누어 미분 단순화):

```
J(θ) = (1 / 2m) · Σ (ŷᵢ − yᵢ)²
```

그래디언트:

```
∂J/∂θ₀ = (1/m) · Σ (ŷᵢ − yᵢ)
∂J/∂θ₁ = (1/m) · Σ (ŷᵢ − yᵢ) · xᵢ
```

업데이트 (simultaneous):

```
tmp0 = α · ∂J/∂θ₀
tmp1 = α · ∂J/∂θ₁
θ₀ ← θ₀ − tmp0
θ₁ ← θ₁ − tmp1
```

두 그래디언트를 **동시에** 계산한 뒤 동시에 반영해야 한다 (θ₁ 업데이트에 갱신된 θ₀가 끼어들면 안 됨).
