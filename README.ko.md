# ft_linear_regression

km(주행거리)로 중고차 가격을 예측하는 단순 선형회귀 모델입니다.
ML 프레임워크 없이 경사하강법(gradient descent)으로 직접 학습합니다.

> English version: [README.md](README.md)

## 요구사항

- `data.csv` (헤더: `km,price`)

## 빌드

```sh
cargo build --release
```

## 실행 방법

`src/bin/` 내에 세 개의 바이너리 파일이 존재합니다.

### 1. 학습 — `train`

CSV를 읽어 경사하강법으로 θ₀, θ₁을 구한 뒤 `theta.json`에 저장합니다.

```sh
cargo run --bin train -- data.csv
```

출력 예시:

```
theta0: 8499.59964940584
theta1: -0.021448963586739037
saved to theta.json
scatter saved to scatter.png
regression saved to regression.png
```

내부 동작:

1. `data.csv` 파싱
2. x, y 각각 min-max [0, 1] 정규화
3. 정규화 공간에서 경사하강법으로 학습
4. 학습된 θ′를 원 스케일(km/price)의 θ로 역변환
5. `theta.json`에 pretty JSON으로 저장
6. 학습 데이터 산점도를 `scatter.png`로, 산점도 + 회귀선을 `regression.png`로 저장

> 정규화하지 않으면 손실함수가 심하게 ill-conditioned 되어 (이 데이터셋 기준 헤시안 λ_max ≈ 1.3 × 10¹⁰) 학습률을 `1e-10` 수준까지 낮춰야 발산을 막을 수 있고, 그래도 수렴이 매우 느립니다. 정규화 후에는 일반적인 `α = 0.1`로도 빠르게 수렴합니다.

### 2. 예측 — `predict`

km 값을 입력받아 예측 가격을 출력합니다.

```sh
cargo run --bin predict -- 100000
# 6353.800889944063
```

`theta.json`이 없으면 과제 규약에 따라 `(θ₀, θ₁) = (0, 0)`으로 동작하여 항상 0을 반환합니다.

두 번째 인자로 CSV 경로를 함께 주면 **산점도 + 회귀선 + 예측 지점**을 `regression.png`로 저장합니다.

```sh
cargo run --bin predict -- 100000 data.csv
# 6353.800889944063
# regression chart saved to regression.png
```

### 3. 정확도 — `precision`

학습된 모델을 데이터셋에 대해 평가합니다. R²와 RMSE를 출력합니다.

```sh
cargo run --bin precision -- data.csv
```

출력 예시:

```
R^2  : 0.732975    (1=완벽, 0=평균예측 수준, <0=평균보다 못함)
RMSE : 667.5667   (price 단위. 예측이 평균 ±668 만큼 빗나감)
```

- **R²**: 결정계수. y 분산 중 모델이 설명하는 비율로, 회귀 정확도의 표준 지표입니다.
- **RMSE**: √평균제곱오차. y와 같은 단위라서 "평균적으로 얼마나 빗나가는지"로 바로 해석할 수 있습니다.

학습 데이터로 측정한 in-sample error이므로, 미지 데이터에 대한 일반화 성능과는 다릅니다.

## 테스트

```sh
cargo test
```

각 모듈의 단위 테스트와 함께 [tests/converges_to_ols.rs](tests/converges_to_ols.rs)의 통합 테스트가 실행됩니다. 합성 데이터에서 **경사하강법 결과가 OLS closed-form 해에 수렴**하는지 검증하여, 옵티마이저 / 정규화 / 역정규화 전체 파이프라인을 가드합니다.

## 디렉토리 구조

```
src/
├── lib.rs                       # 모듈 루트
├── bin/
│   ├── train.rs                 # 학습 바이너리
│   ├── predict.rs               # 예측 바이너리
│   └── precision.rs             # 정확도 평가 바이너리
├── model/
│   ├── mod.rs
│   └── linear_regression.rs     # gradient_descent, normal_equation, Model
├── preprocess/
│   ├── mod.rs
│   └── normalize.rs             # MinMax, denormalize_thetas
├── utils/
│   ├── mod.rs                   # THETA_PATH 상수
│   ├── dataset.rs               # CSV 파싱 (csv 크레이트)
│   └── metrics.rs               # mse, rmse, r_squared
└── viz/
    ├── mod.rs
    └── plot.rs                  # scatter / regression 차트 (plotters 크레이트)
tests/
└── converges_to_ols.rs          # GD vs OLS 회귀 테스트
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

손실함수 (MSE 형태, 미분이 깔끔하도록 1/2 계수):

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

두 그래디언트를 **동시에** 계산한 뒤 동시에 반영해야 합니다. θ₁ 업데이트에 갱신된 θ₀가 적용되면 더 이상 순수 경사하강법이 아닙니다.

# 최적 θ 찾기

선형회귀가 "데이터에 직선 맞추기"라는 추상적 문제를 어떻게 구체적인 최적화 문제로 바꾸고,
그 문제를 실제로 어떻게 푸는지에 대한 문서입니다.

## 문제 정의

데이터 $(x_i, y_i)$, $i = 1, \ldots, m$가 주어졌을 때,

$$\hat{y}_i = \theta_0 + \theta_1 \cdot x_i$$

이 모든 $i$에 대해 $y_i$에 "최대한 가까워지도록" 하는 $(\theta_0, \theta_1)$을 찾는 문제입니다.

이걸 구체화하려면 세 가지가 필요합니다:

1. 가설(hypothesis) 선택 — 어떤 함수족에서 찾을 것인가
2. 손실(loss) 정의 — "가깝다"를 단일 숫자로 표현
3. 그 손실을 최소화하는 θ를 찾기 — **최적값(optimal value)**

이 문서는 3번에 대한 것이지만, 1, 2번이 "최적"의 의미 자체를 결정하므로 모두 짚고 갑니다.

## 1. 가설

직선 함수족으로 한정합니다:

$$h_\theta(x) = \theta_0 + \theta_1 x$$

2개 모수의 함수족이며, 최적화는 $(\theta_0, \theta_1)$ 평면에서 일어납니다.

## 2. 손실

### 왜 제곱 오차인가

샘플당 오차는 $\hat{y}_i - y_i$입니다. 이걸 그대로 쓰면 두 가지 문제가 있습니다:

- **부호 상쇄**: 큰 양의 오차와 큰 음의 오차가 합쳐져 0이 될 수 있음
- **종합**: $m$개 숫자가 아니라 단일 숫자가 필요함

부호를 없애려면 비음수 함수가 필요합니다. 표준 후보는 $|\cdot|$ (L1)과 $(\cdot)^2$ (L2)
두 가지이며, L2를 선택하는 이유는:

1. **모든 점에서 미분 가능** — 경사하강법이 요구. $|x|$는 0에서 꺾임
2. **θ에 대해 강볼록** — 전역 최솟값이 정확히 하나, 평탄 구간 없음
3. **큰 오차에 더 큰 페널티** — 제곱이 이상치의 영향을 증폭
4. **가우시안 잡음 가정 하의 MLE** — §3 참고. 이게 진짜 깊은 이유

데이터셋 전체에 대해 합한 뒤 평균을 내면:

$$J(\theta_0, \theta_1) = \frac{1}{2m} \sum_{i=1}^{m} (\hat{y}_i - y_i)^2$$

$\frac{1}{m}$은 데이터 크기에 따라 손실이 변하지 않게 하여, $m$이 바뀔 때 학습률을
다시 튜닝할 필요가 없도록 합니다. $\frac{1}{2}$은 관례로, $(\cdot)^2$을 미분할 때
떨어지는 2를 상쇄해 그래디언트를 깔끔하게 만듭니다. $J$ 앞에 양의 상수를 곱해도
$\arg\min$은 변하지 않습니다.

## 3. 통계적으로 왜 제곱 오차인가

제곱 오차를 쓰는 더 깊은 이유: **가우시안 잡음 모델 하의 최대우도 추정량(MLE)**이기
때문입니다.

데이터가 다음 모형에서 생성되었다고 가정:

$$y_i = \theta_0 + \theta_1 x_i + \varepsilon_i, \quad \varepsilon_i \overset{\text{i.i.d.}}{\sim} \mathcal{N}(0, \sigma^2)$$

그러면 $y_i \mid x_i \sim \mathcal{N}(\theta_0 + \theta_1 x_i, \sigma^2)$이고, 우도는

$$L(\theta) = \prod_{i=1}^{m} \frac{1}{\sigma\sqrt{2\pi}} \exp\left(-\frac{(y_i - \theta_0 - \theta_1 x_i)^2}{2\sigma^2}\right)$$

로그를 취하면

$$\ell(\theta) = -\frac{m}{2}\log(2\pi\sigma^2) - \frac{1}{2\sigma^2}\sum_{i=1}^{m}(y_i - \theta_0 - \theta_1 x_i)^2$$

첫 항은 θ와 무관. $\frac{1}{2\sigma^2}$은 양의 상수. 따라서

$$\arg\max_\theta \ell(\theta) = \arg\min_\theta \sum_{i=1}^{m}(y_i - \theta_0 - \theta_1 x_i)^2$$

가우시안 우도를 최대화하는 것은 **곧** 제곱 오차를 최소화하는 것입니다. 최소제곱법은
임의로 선택한 손실이 아니라, **등분산 가우시안 잡음 모델 하의 MLE**입니다.

(잡음 분포가 바뀌면 손실도 바뀝니다: 라플라스 → L1 손실, 베르누이 → cross-entropy.
손실 함수는 단순한 편의가 아니라 잡음에 대한 모델링 결정입니다.)

## 4. 최적값의 존재와 유일성

애초에 최적값이 존재하는가? 유일한가?

$J(\theta_0, \theta_1)$은 θ의 아핀 함수들의 제곱합이므로 θ에 대한 볼록 이차함수입니다.
헤시안은

$$H = \frac{1}{m} X^\top X, \quad X = \begin{bmatrix} 1 & x_1 \\ \vdots & \vdots \\ 1 & x_m \end{bmatrix}$$

$X^\top X$는 항상 양의 준정부호(positive semi-definite)이며, **$X$의 열이 선형독립**일
때 양의 정부호(positive definite)입니다. 이 2열 $X$의 경우, $x_i$들이 모두 같지만
않으면 성립합니다.

비퇴화 데이터셋이라면 $H \succ 0$이고, $J$는 강볼록이므로:

- 정지점(stationary point)이 존재하고, 유일하며, 전역 최솟값
- 페르마 조건 $\nabla J(\theta^*) = 0$은 필요조건이자 **충분조건**

이게 $\nabla J = 0$을 직접 풀어 정답으로 신뢰할 수 있는 이유입니다.

## 5. 그래디언트

각 모수에 대해 연쇄법칙으로 미분:

$$\frac{\partial J}{\partial \theta_0} = \frac{1}{m}\sum_{i=1}^{m}(\hat{y}_i - y_i)$$

$$\frac{\partial J}{\partial \theta_1} = \frac{1}{m}\sum_{i=1}^{m}(\hat{y}_i - y_i) \cdot x_i$$

행렬 형태로는

$$\nabla J(\theta) = \frac{1}{m} X^\top (X\theta - y)$$

$\partial J / \partial \theta_1$에 $x_i$가 곱해진 것은 연쇄법칙 때문입니다 —
내부 함수 $\hat{y}_i = \theta_0 + \theta_1 x_i$를 $\theta_1$로 편미분하면 $x_i$가
남습니다. 직관적으로는, $x = 0$에서 멀리 떨어진 샘플일수록 기울기에 더 큰
영향(leverage)을 미친다는 의미입니다.

## 6. 경로 A — 닫힌 해 (정규방정식)

그래디언트를 0으로 놓으면

$$\frac{1}{m} X^\top (X\theta - y) = 0 \quad \Longrightarrow \quad X^\top X \theta = X^\top y$$

이를 **정규방정식(normal equation)**이라 합니다. $X^\top X$가 가역이면 (비퇴화
데이터셋이면 항상 성립),

$$\hat{\theta} = (X^\top X)^{-1} X^\top y$$

단일 행렬 연산으로 끝나며, 부동소수점 오차를 제외하면 정확한 해입니다. 학습률도, 반복
횟수도, 수렴 디버깅도 없습니다. 단순 선형회귀의 경우 행렬 없이도 유도할 수 있습니다:

$$\hat{\theta}_1 = \frac{\sum (x_i - \bar{x})(y_i - \bar{y})}{\sum (x_i - \bar{x})^2}, \qquad \hat{\theta}_0 = \bar{y} - \hat{\theta}_1 \bar{x}$$

행렬 버전을 $p = 2$로 특수화하면 같은 식이 나옵니다.

이 프로젝트의 [linear_regression.rs](src/model/linear_regression.rs)에 `normal_equation`이
구현되어 있고, [converges_to_ols.rs](tests/converges_to_ols.rs)의 통합 테스트가 합성
데이터에서 GD가 이 닫힌 해로 수렴하는지 검증합니다.

### 그럼 왜 항상 정규방정식을 안 쓰는가?

이 문제에서는 정규방정식이 GD보다 빠르고 정확합니다. 그럼에도 GD를 쓰는 이유:

- 이 프로젝트의 목표는 선형회귀를 어떤 방법으로든 푸는 게 아니라, **경사하강법을
  직접 구현**하는 것
- 정규방정식은 역행렬 계산이 $O(p^3)$. $p = 2$에서는 무시할 만하지만,
  $p = 10^4$ (현대 ML의 흔한 특성 차원)에서는 사실상 불가능
- $X^\top X$가 특이행렬이거나 특이행렬에 가까울 수 있음 (선형종속 특성, $p > m$).
  이 경우 역행렬이 존재하지 않거나 수치적으로 불안정
- 비선형 모델(로지스틱 회귀, 신경망)에는 닫힌 해 자체가 없음. GD는 일반화되지만
  정규방정식은 그렇지 않음

선형회귀는 닫힌 해가 존재하는 매우 드문 모델 중 하나입니다. 여기서 GD를 익히는 것은
이후의 모든 모델에 대한 준비입니다.

## 7. 경로 B — 반복적 (경사하강법)

GD는 $\nabla J = 0$을 대수적으로 풀지 않습니다. 손실 곡면 위를 걸어 내려갑니다:

$$\theta_{t+1} = \theta_t - \alpha \cdot \nabla J(\theta_t)$$

직관은 국소적입니다. 어느 지점에서든 $\nabla J$는 가장 빠르게 **증가**하는 방향을
가리키므로, $-\nabla J$는 가장 빠르게 **감소**하는 방향입니다. 그 방향으로 작은
스텝을 밟고, 다시 계산하고, 반복합니다.

모수별로:

$$\theta_0 \leftarrow \theta_0 - \alpha \cdot \frac{1}{m}\sum_{i=1}^{m}(\hat{y}_i - y_i)$$

$$\theta_1 \leftarrow \theta_1 - \alpha \cdot \frac{1}{m}\sum_{i=1}^{m}(\hat{y}_i - y_i) \cdot x_i$$

"동시 업데이트(simultaneous update)" 규칙에 주의: 두 그래디언트는 **같은**
$(\theta_0, \theta_1)$에서 평가된 후 둘 다 갱신되어야 합니다. 만약 갱신된 $\theta_0$이
$\partial J / \partial \theta_1$ 계산에 들어가면, 결과 벡터는 더 이상 어느 한 점에서의
그래디언트가 아니게 되며 수렴 보장이 깨집니다.

### 왜 이 업데이트가 정당한가

업데이트 규칙은 1차 테일러 전개에서 나옵니다:

$$J(\theta + d) \approx J(\theta) + \nabla J(\theta)^\top d$$

크기가 $\|d\| = \epsilon$로 고정된 스텝에 대해, 코시-슈바르츠 부등식은 $\nabla J(\theta)^\top d$가
$d$가 $\nabla J$와 정반대 방향일 때 최소화됨을 말해줍니다. 즉 $-\nabla J$는 1차 근사의
정확도 안에서 국소적인 최급강하 방향이고, 스텝 크기 $\alpha$가 그 근사를 얼마나
신뢰할지를 조절합니다.

### 수렴

우리의 $J$처럼 강볼록 이차함수의 경우, $\mu = \lambda_{\min}(H)$, $L = \lambda_{\max}(H)$라 하면:

- $0 < \alpha < 2/L$이면 GD는 $\theta^*$로 수렴
- $\alpha = 2/(\mu + L)$이 plain GD의 이론적 최적 스텝 (수축률 $(L-\mu)/(L+\mu)$); 실용적으로는 $\mu$를 모르는 경우가 많아 $\alpha = 1/L$을 안전한 기본값으로 자주 사용
- $\alpha > 2/L$이면 GD 발산 ($\alpha = 2/L$ 경계에서는 발산하지 않지만 진동하여 수렴하지 않음)

이 손실의 경우 $L = \lambda_{\max}\left(\frac{1}{m} X^\top X\right)$입니다.

원본(정규화 안 한) 자동차 데이터에서는 mileage 값이 수만 단위라서
$L \approx 1.3 \times 10^{10}$입니다. 따라서 발산을 피하려면 $\alpha \lesssim 10^{-10}$
이어야 하고, 그 속도에서 수렴은 사실상 멈춰 있는 수준입니다.

min-max 정규화로 $[0, 1]$로 옮긴 뒤에는 $H$의 고유값이 1 정도 크기가 되어 $\alpha = 0.1$
이 잘 작동합니다. **정규화가 선택이 아닌 이유**가 이것입니다 — 최적화 문제 자체를 잘
조건화(well-conditioned)된 상태로 만듭니다. 최적값의 위치는 변하지 않고, 그 주변
손실 곡면의 기하만 바뀝니다.

## 8. 정규화 전후의 최적값 위치

두 경로 모두 입력으로 들어간 좌표계의 θ를 반환합니다. $x' = (x - x_{min})/(x_{max} - x_{min})$, $y'$도 같은 방식으로 정의된 정규화 공간 $(x', y')$에서 학습하면, 학습된 θ′는 정규화 공간에 머무릅니다. 원본 (km, price) 단위의 예측기를 얻으려면 정규화를 해석적으로 되돌려야 합니다.

$\hat{y}' = \theta_0' + \theta_1' x'$이라면, 대입해서 정리하면

$$\theta_1 = \theta_1' \cdot \frac{y_{max} - y_{min}}{x_{max} - x_{min}}$$

$$\theta_0 = y_{min} + \theta_0' \cdot (y_{max} - y_{min}) - \theta_1 \cdot x_{min}$$

이렇게 하면 $\hat{y} = \theta_0 + \theta_1 x$가 원본 단위에서 예측합니다. 이것이
[`denormalize_thetas`](src/preprocess/normalize.rs)가 하는 일입니다. 이 단계를 잊는
것이 동시 업데이트 위반 다음으로 가장 흔한 버그입니다 — 손실 곡선은 멀쩡해 보여도
`predict`가 엉뚱한 값을 뱉습니다.

## 9. 요약

| 단계          | 선택                 | 이유                                             |
| ------------- | -------------------- | ------------------------------------------------ |
| 가설          | $x$의 선형함수       | 문제 자체가 "직선 맞추기"                        |
| 손실          | 평균 제곱 오차 (MSE) | 미분 가능, 볼록, 가우시안 잡음 하의 MLE          |
| 최적값의 존재 | 강볼록성으로 보장    | 비퇴화 데이터에서 $X^\top X \succ 0$             |
| 풀이 방법     | 경사하강법           | 선형을 넘어 일반화 가능; 과제 요구사항           |
| 실용성        | 정규화 선행          | 안 하면 $\alpha$를 $10^{-10}$ 수준까지 낮춰야 함 |
| 최종 좌표     | θ′ → θ 역변환        | 예측은 원본 단위로 나와야 함                     |

최적값 자체는 데이터와 손실의 선택만으로 완전히 결정됩니다. GD와 정규방정식은 같은
지점에 도달하는 두 가지 다른 경로입니다. 정규방정식은 한 번의 행렬 연산으로 가고,
GD는 $(0, 0)$에서 출발해 `config.iterations` 스텝을 걸어서 갑니다. 이 문제에서 둘은
소수점 몇 자리까지 일치합니다 — 그게 `tests/converges_to_ols.rs`가 검증하는 것입니다.
