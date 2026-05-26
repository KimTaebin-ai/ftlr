# ft_linear_regression

A simple linear-regression model that predicts used-car prices from mileage (km).
Trained from scratch with gradient descent — no ML framework.

> Korean version: [README.ko.md](README.ko.md)

## Requirements

- `data.csv` (header: `km,price`)

## Build

```sh
cargo build --release
```

## Usage

Three binaries live under `src/bin/`.

### 1. Train — `train`

Reads the CSV, runs gradient descent for θ₀ and θ₁, and writes the result to `theta.json`.

```sh
cargo run --bin train -- data.csv
```

Example output:

```
theta0: 8499.59964940584
theta1: -0.021448963586739037
saved to theta.json
scatter saved to scatter.png
regression saved to regression.png
```

What happens internally:

1. Parse `data.csv`.
2. Min-max normalize x and y independently to `[0, 1]`.
3. Run gradient descent in the normalized space.
4. Denormalize θ′ back to the original (km, price) scale.
5. Save θ to `theta.json` (pretty JSON).
6. Save a scatter plot of the training data to `scatter.png`, and the scatter + regression line to `regression.png`.

> Without normalization the loss surface is severely ill-conditioned (Hessian λ_max ≈ 1.3 × 10¹⁰ on this dataset), so the learning rate would have to drop to ~`1e-10` to avoid divergence — and would still converge slowly. After normalization a plain `α = 0.1` converges quickly.

### 2. Predict — `predict`

Takes a km value and prints the predicted price.

```sh
cargo run --bin predict -- 100000
# 6353.800889944063
```

If `theta.json` does not exist, the model falls back to `(θ₀, θ₁) = (0, 0)` per spec, which always returns 0.

Pass a CSV path as the second argument to also draw **scatter + regression line + the highlighted prediction point** to `regression.png`:

```sh
cargo run --bin predict -- 100000 data.csv
# 6353.800889944063
# regression chart saved to regression.png
```

### 3. Precision — `precision`

Evaluates the trained model on a dataset. Prints R² and RMSE.

```sh
cargo run --bin precision -- data.csv
```

Example output:

```
R^2  : 0.732975    (1=perfect, 0=mean baseline, <0=worse than mean)
RMSE : 667.5667   (price units — predictions are off by ~668 on average)
```

- **R²**: coefficient of determination. Share of the variance in y explained by the model. Standard regression-quality metric.
- **RMSE**: √mean-squared-error. Same units as y, so it reads directly as "average prediction error".

This is in-sample error (measured on the training set), not held-out generalization error.

## Tests

```sh
cargo test
```

Runs the unit tests in each module plus the integration test in [tests/converges_to_ols.rs](tests/converges_to_ols.rs), which checks that gradient descent on synthetic data converges to the OLS closed-form solution. That test guards the whole optimizer / normalize / denormalize pipeline.

## Layout

```
src/
├── lib.rs                       # module roots
├── bin/
│   ├── train.rs                 # training binary
│   ├── predict.rs               # prediction binary
│   └── precision.rs             # evaluation binary
├── model/
│   ├── mod.rs
│   └── linear_regression.rs     # gradient_descent, normal_equation, Model
├── preprocess/
│   ├── mod.rs
│   └── normalize.rs             # MinMax, denormalize_thetas
├── utils/
│   ├── mod.rs                   # THETA_PATH constant
│   ├── dataset.rs               # CSV parsing (csv crate)
│   └── metrics.rs               # mse, rmse, r_squared
└── viz/
    ├── mod.rs
    └── plot.rs                  # scatter / regression charts (plotters crate)
tests/
└── converges_to_ols.rs          # GD vs OLS regression test
```

## Dependencies

- `serde`, `serde_json` — serializing `theta.json`
- `csv` — RFC 4180-compliant CSV parsing
- `plotters` — PNG charts (`scatter.png` / `regression.png`)

## Algorithm

Hypothesis:

```
ŷ = θ₀ + θ₁ · x
```

Loss (MSE with a 1/2 factor so the gradient drops the 2):

```
J(θ) = (1 / 2m) · Σ (ŷᵢ − yᵢ)²
```

Gradients:

```
∂J/∂θ₀ = (1/m) · Σ (ŷᵢ − yᵢ)
∂J/∂θ₁ = (1/m) · Σ (ŷᵢ − yᵢ) · xᵢ
```

Simultaneous update:

```
tmp0 = α · ∂J/∂θ₀
tmp1 = α · ∂J/∂θ₁
θ₀ ← θ₀ − tmp0
θ₁ ← θ₁ − tmp1
```

Both gradients must be computed **before** either θ is updated — if the new θ₀ leaks into the θ₁ update, the math is no longer plain gradient descent.

# Finding the Optimal θ

How linear regression turns "fit a line to data" into a concrete optimization problem,
and how that problem is actually solved.

## Problem statement

Given pairs $(x_i, y_i)$ for $i = 1, \ldots, m$, find $(\theta_0, \theta_1)$ such that

$$\hat{y}_i = \theta_0 + \theta_1 \cdot x_i$$

is "as close as possible" to $y_i$ across the whole dataset.

To make this concrete we need to do three things:

1. Pick a hypothesis (the family of functions we'll search over).
2. Define a loss (what "close" means as a single number).
3. Find the θ that minimizes that loss — the **optimal value**.

This document is about step 3, but steps 1 and 2 determine what "optimal" means in the
first place, so we walk through them first.

## 1. Hypothesis

We restrict ourselves to lines:

$$h_\theta(x) = \theta_0 + \theta_1 x$$

This is a 2-parameter family. The optimization will live in the $(\theta_0, \theta_1)$ plane.

## 2. Loss

### Why squared error

Per-sample error is $\hat{y}_i - y_i$. Two issues with using it directly:

- **Sign cancels**: large positive errors and large negative errors can sum to zero.
- **Aggregation**: we need one number for the whole dataset, not $m$ numbers.

To kill the sign we need a non-negative function. The two standard choices are
$|\cdot|$ (L1) and $(\cdot)^2$ (L2). We pick L2 because:

1. **Differentiable everywhere** — gradient descent needs this. $|x|$ has a kink at 0.
2. **Strictly convex in θ** — exactly one global minimum, no plateaus.
3. **Penalizes large errors more** — squaring amplifies outliers' influence.
4. **MLE under Gaussian noise** — see §3. This is the deep reason.

Aggregating over the dataset and averaging:

$$J(\theta_0, \theta_1) = \frac{1}{2m} \sum_{i=1}^{m} (\hat{y}_i - y_i)^2$$

The $\frac{1}{m}$ makes the loss invariant to dataset size (so the learning rate doesn't
need re-tuning when $m$ changes). The $\frac{1}{2}$ is a convention: it cancels the 2
that falls out when differentiating $(\cdot)^2$, leaving a clean gradient. Any positive
constant in front of $J$ leaves $\arg\min$ unchanged.

## 3. Why squared error, statistically

The deeper reason to use squared loss: it's the **maximum-likelihood estimator** under
a Gaussian noise model.

Assume the data is generated by

$$y_i = \theta_0 + \theta_1 x_i + \varepsilon_i, \quad \varepsilon_i \overset{\text{i.i.d.}}{\sim} \mathcal{N}(0, \sigma^2)$$

Then $y_i \mid x_i \sim \mathcal{N}(\theta_0 + \theta_1 x_i, \sigma^2)$, and the likelihood is

$$L(\theta) = \prod_{i=1}^{m} \frac{1}{\sigma\sqrt{2\pi}} \exp\left(-\frac{(y_i - \theta_0 - \theta_1 x_i)^2}{2\sigma^2}\right)$$

Taking the log,

$$\ell(\theta) = -\frac{m}{2}\log(2\pi\sigma^2) - \frac{1}{2\sigma^2}\sum_{i=1}^{m}(y_i - \theta_0 - \theta_1 x_i)^2$$

The first term doesn't depend on θ. The $\frac{1}{2\sigma^2}$ is a positive constant. So

$$\arg\max_\theta \ell(\theta) = \arg\min_\theta \sum_{i=1}^{m}(y_i - \theta_0 - \theta_1 x_i)^2$$

Maximizing the Gaussian likelihood **is** minimizing the squared error. Least squares
isn't an arbitrary loss — it's the MLE for the constant-variance Gaussian noise model.

(Change the noise distribution and the loss changes: Laplace noise → L1 loss,
Bernoulli outcomes → cross-entropy. The loss function is a modeling choice about the
noise, not just a convenience.)

## 4. Existence and uniqueness of the optimum

Is there an optimum at all? Is it unique?

$J(\theta_0, \theta_1)$ is a sum of squares of affine functions of θ, so it's a convex
quadratic in θ. Its Hessian is

$$H = \frac{1}{m} X^\top X, \quad X = \begin{bmatrix} 1 & x_1 \\ \vdots & \vdots \\ 1 & x_m \end{bmatrix}$$

$X^\top X$ is positive semi-definite always. It's positive definite (strictly) iff the
columns of $X$ are linearly independent — which for this 2-column $X$ happens iff
the $x_i$ are not all equal.

For any non-degenerate dataset, $H \succ 0$, so $J$ is strictly convex:

- A stationary point exists, is unique, and is the global minimum.
- Fermat's condition $\nabla J(\theta^*) = 0$ is both necessary **and** sufficient.

This is why we can solve $\nabla J = 0$ directly and trust the answer.

## 5. The gradient

Differentiating $J$ with respect to each parameter (chain rule):

$$\frac{\partial J}{\partial \theta_0} = \frac{1}{m}\sum_{i=1}^{m}(\hat{y}_i - y_i)$$

$$\frac{\partial J}{\partial \theta_1} = \frac{1}{m}\sum_{i=1}^{m}(\hat{y}_i - y_i) \cdot x_i$$

In matrix form,

$$\nabla J(\theta) = \frac{1}{m} X^\top (X\theta - y)$$

The $x_i$ factor in $\partial J / \partial \theta_1$ comes from the chain rule:
the inner function $\hat{y}_i = \theta_0 + \theta_1 x_i$ differentiated with respect
to $\theta_1$ leaves $x_i$. Intuitively, samples far from $x = 0$ exert more
leverage on the slope than samples near zero.

## 6. Path A — closed form (Normal Equation)

Set the gradient to zero:

$$\frac{1}{m} X^\top (X\theta - y) = 0 \quad \Longrightarrow \quad X^\top X \theta = X^\top y$$

This is the **normal equation**. When $X^\top X$ is invertible (true for any
non-degenerate dataset),

$$\hat{\theta} = (X^\top X)^{-1} X^\top y$$

A single matrix computation, exact up to floating-point error. No learning rate, no
iteration count, no convergence to debug. For simple linear regression you can also
derive it without matrices:

$$\hat{\theta}_1 = \frac{\sum (x_i - \bar{x})(y_i - \bar{y})}{\sum (x_i - \bar{x})^2}, \qquad \hat{\theta}_0 = \bar{y} - \hat{\theta}_1 \bar{x}$$

These are the same formulas you'd get from the matrix version specialized to $p = 2$.

This project includes `normal_equation` in [linear_regression.rs](src/model/linear_regression.rs)
and the integration test [converges_to_ols.rs](tests/converges_to_ols.rs) checks that
gradient descent converges to this closed-form answer on synthetic data.

### Why not always use this?

For this problem, the normal equation is faster and more accurate than GD. We use GD
anyway because:

- The point of this project is to implement gradient descent, not to solve linear
  regression by any means available.
- Normal equation costs $O(p^3)$ for inversion. Fine at $p = 2$, untenable at
  $p = 10^4$ (modern feature dimensions).
- $X^\top X$ can be singular or near-singular (collinear features, $p > m$),
  in which case the inverse doesn't exist or is numerically unstable.
- Non-linear models (logistic regression, neural networks) have no closed form at all.
  GD generalizes; the normal equation does not.

Linear regression is one of the very few models where a closed form exists. Learning
GD on it is preparation for everything that comes after.

## 7. Path B — iterative (Gradient Descent)

GD doesn't solve $\nabla J = 0$ algebraically. It walks downhill on the loss surface:

$$\theta_{t+1} = \theta_t - \alpha \cdot \nabla J(\theta_t)$$

The intuition is local: at any point, $\nabla J$ points in the direction of steepest
**increase**, so $-\nabla J$ points in the direction of steepest **decrease**. Take a
small step that way, recompute, repeat.

Per-parameter:

$$\theta_0 \leftarrow \theta_0 - \alpha \cdot \frac{1}{m}\sum_{i=1}^{m}(\hat{y}_i - y_i)$$

$$\theta_1 \leftarrow \theta_1 - \alpha \cdot \frac{1}{m}\sum_{i=1}^{m}(\hat{y}_i - y_i) \cdot x_i$$

Note the "simultaneous update" rule: both gradients must be evaluated at the same
$(\theta_0, \theta_1)$ before either is changed. If the new $\theta_0$ leaks into
the computation of $\partial J / \partial \theta_1$, the resulting vector is no
longer the gradient at any single point, and the convergence guarantees go out the
window.

### Why this is justified

The update rule comes from a first-order Taylor expansion:

$$J(\theta + d) \approx J(\theta) + \nabla J(\theta)^\top d$$

For a fixed-norm step $\|d\| = \epsilon$, the Cauchy-Schwarz inequality says the
quantity $\nabla J(\theta)^\top d$ is minimized when $d$ points opposite to $\nabla J$.
So $-\nabla J$ is locally the steepest descent direction, up to the accuracy of the
first-order approximation. The step size $\alpha$ controls how far we trust that
approximation.

### Convergence

For a strictly convex quadratic like our $J$, with $\mu = \lambda_{\min}(H)$ and $L = \lambda_{\max}(H)$:

- If $0 < \alpha < 2/L$, GD converges to $\theta^*$.
- The theoretically optimal step for plain GD is $\alpha = 2/(\mu + L)$, giving contraction rate $(L - \mu)/(L + \mu)$. In practice $\mu$ is often unknown, so $\alpha = 1/L$ is the common safe default.
- If $\alpha > 2/L$, GD diverges. At the boundary $\alpha = 2/L$ it doesn't diverge but oscillates without converging.

Concretely for this loss, $L = \lambda_{\max}\left(\frac{1}{m} X^\top X\right)$.

On the raw (un-normalized) car data, $L \approx 1.3 \times 10^{10}$ because mileage
values are in the tens of thousands. That forces $\alpha \lesssim 10^{-10}$ to avoid
divergence, and at that rate convergence is glacial.

After min-max normalization to $[0, 1]$, the eigenvalues of $H$ are on the order of 1,
and $\alpha = 0.1$ works fine. **This is why normalization is not optional** — it's
what makes the optimization problem well-conditioned. The optimum is unchanged in
position; only the geometry of the loss surface around it changes.

## 8. Where the optimum lives, before and after normalization

Both paths return θ in whatever coordinate system the inputs were in. If we train on
normalized $(x', y')$ with $x' = (x - x_{min})/(x_{max} - x_{min})$ and similarly for
$y'$, the learned θ′ lives in normalized space. To get an estimator on the original
(km, price) scale we need to undo the normalization analytically.

If $\hat{y}' = \theta_0' + \theta_1' x'$, substituting back and rearranging gives

$$\theta_1 = \theta_1' \cdot \frac{y_{max} - y_{min}}{x_{max} - x_{min}}$$

$$\theta_0 = y_{min} + \theta_0' \cdot (y_{max} - y_{min}) - \theta_1 \cdot x_{min}$$

so that $\hat{y} = \theta_0 + \theta_1 x$ predicts in the original units. This is
what [`denormalize_thetas`](src/preprocess/normalize.rs) does. Forgetting this step
is the second-most common bug after non-simultaneous updates — the loss curve will
look great but `predict` will return nonsense values.

## 9. Summary

| Step                 | Choice                         | Why                                              |
| -------------------- | ------------------------------ | ------------------------------------------------ |
| Hypothesis           | Linear in $x$                  | The problem is "fit a line"                      |
| Loss                 | Mean squared error             | Differentiable, convex, MLE under Gaussian noise |
| Existence of optimum | Guaranteed by strict convexity | $X^\top X \succ 0$ on non-degenerate data        |
| Solution method      | Gradient descent               | Generalizes beyond linear; project requirement   |
| Feasibility          | Normalize features first       | Otherwise $\alpha$ has to be ~$10^{-10}$         |
| Final coordinates    | Denormalize θ′ → θ             | Predictions need to be in the original units     |

The optimum itself is determined entirely by the data and the choice of loss. GD and
the normal equation are two different paths to the same point. The normal equation
goes there in one matrix computation; GD walks there from $(0, 0)$ in
`config.iterations` steps. On this problem they agree to several decimals — that's
what `tests/converges_to_ols.rs` verifies.
