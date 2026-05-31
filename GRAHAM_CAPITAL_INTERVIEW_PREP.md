# 🧠 Graham Capital Management — Senior Quant Researcher Interview Prep
### Round 1 | 30 Questions with Detailed Answers | 2025–2026 Edition

> **Synopsis:** Graham Capital Management (~$20B AUM) is a Westport, CT-based macro & quant hedge fund. Their quant research team operates as a *"think tank"* building systematic alpha through signal research, advanced ML, and rigorous statistical modeling. Round 1 interviews are typically 30–45 min calls with a senior quant — expect deep technical probing across probability, stochastic calculus, linear algebra, ML/AI, coding, and sharp brain teasers. Questions below are sourced and synthesized from Glassdoor, Blind, Wall Street Oasis, QuantNet, LeetCode, eFinancialCareers, and OpenQuant (2025–2026 cycles), weighted toward 2026 intelligence.

---

## 📋 Table of Contents

| # | Category | Question |
|---|----------|----------|
| 1 | [🎲 Probability & Statistics](#a-probability--statistics) | [Conditional expectation & Bayes update for signal decay](#q1) |
| 2 | [🎲 Probability & Statistics](#a-probability--statistics) | [Order statistics: expected minimum of N uniform draws](#q2) |
| 3 | [🎲 Probability & Statistics](#a-probability--statistics) | [Sharpe ratio distribution under finite samples](#q3) |
| 4 | [🎲 Probability & Statistics](#a-probability--statistics) | [Multiple testing & FDR in factor discovery](#q4) |
| 5 | [🎲 Probability & Statistics](#a-probability--statistics) | [Copula dependence vs linear correlation](#q5) |
| 6 | [📈 Stochastic Calculus](#b-stochastic-calculus) | [Itô's Lemma applied to log-price dynamics](#q6) |
| 7 | [📈 Stochastic Calculus](#b-stochastic-calculus) | [Girsanov theorem and risk-neutral measure](#q7) |
| 8 | [📈 Stochastic Calculus](#b-stochastic-calculus) | [Ornstein–Uhlenbeck mean reversion & half-life](#q8) |
| 9 | [📈 Stochastic Calculus](#b-stochastic-calculus) | [Feynman–Kac formula and PDE connection](#q9) |
| 10 | [🔢 Linear Algebra](#c-linear-algebra) | [PCA for factor extraction from returns covariance](#q10) |
| 11 | [🔢 Linear Algebra](#c-linear-algebra) | [Eigenvalue stability and Marchenko–Pastur law](#q11) |
| 12 | [🔢 Linear Algebra](#c-linear-algebra) | [Regularization: Ridge vs LASSO in high-dimensional regression](#q12) |
| 13 | [🤖 Machine Learning](#d-machine-learning) | [Gradient boosting for cross-sectional return prediction](#q13) |
| 14 | [🤖 Machine Learning](#d-machine-learning) | [Overfitting in backtest: walk-forward validation design](#q14) |
| 15 | [🤖 Machine Learning](#d-machine-learning) | [Recurrent nets vs transformers for time-series alpha](#q15) |
| 16 | [🤖 Machine Learning](#d-machine-learning) | [Purging and embargoing in financial CV](#q16) |
| 17 | [🧬 Artificial Intelligence](#e-artificial-intelligence) | [LLMs for alternative data signal extraction (2026)](#q17) |
| 18 | [🧬 Artificial Intelligence](#e-artificial-intelligence) | [Reinforcement learning for execution optimization](#q18) |
| 19 | [🧬 Artificial Intelligence](#e-artificial-intelligence) | [Causal inference vs correlation in AI-driven alpha](#q19) |
| 20 | [💻 Coding](#f-coding) | [Vectorized rolling Sharpe ratio in Python/NumPy](#q20) |
| 21 | [💻 Coding](#f-coding) | [Kalman filter implementation for dynamic beta](#q21) |
| 22 | [💻 Coding](#f-coding) | [Cointegration test (Engle–Granger) from scratch](#q22) |
| 23 | [🧩 Reasoning](#g-reasoning) | [Assess signal capacity: when does alpha decay with AUM?](#q23) |
| 24 | [🧩 Reasoning](#g-reasoning) | [Design a macro momentum strategy from scratch](#q24) |
| 25 | [🧩 Reasoning](#g-reasoning) | [Transaction cost model and signal threshold calibration](#q25) |
| 26 | [🎪 Brain Teasers](#h-brain-teasers) | [Expected number of flips to get two consecutive heads](#q26) |
| 27 | [🎪 Brain Teasers](#h-brain-teasers) | [100 prisoners & light switch puzzle](#q27) |
| 28 | [🎪 Brain Teasers](#h-brain-teasers) | [Secretary problem: optimal stopping rule](#q28) |
| 29 | [∑ Mathematical Induction](#i-mathematical-induction) | [Prove sum of first N integers by induction](#q29) |
| 30 | [∑ Mathematical Induction](#i-mathematical-induction) | [Prove geometric series formula by induction](#q30) |

---

## 🎲 A. Probability & Statistics

<a name="q1"></a>
### Q1 — Conditional Expectation & Bayesian Signal Decay
> *"You have an alpha signal with IC = 0.05. Halfway through the holding period you observe a noisy mid-period return. How do you update your position size using Bayes' theorem? What happens to IC as the holding period lengthens?"*
> 🗓️ *Reported: Glassdoor / eFinancialCareers, Q1 2026*

**Answer:**

**Setup.** Let $\alpha$ be the true expected return (unobserved), $s$ the signal, and $r_{mid}$ the mid-period noisy return. Assume a Gaussian linear model:

$$\alpha \sim \mathcal{N}(0, \sigma_\alpha^2), \quad s = \alpha + \epsilon_s, \quad r_{mid} = \alpha + \epsilon_r$$

where $\epsilon_s \sim \mathcal{N}(0, \sigma_s^2)$ and $\epsilon_r \sim \mathcal{N}(0, \sigma_r^2)$ are independent noise terms.

**Bayesian Update.** The posterior mean after observing both $s$ and $r_{mid}$ is:

$$\mathbb{E}[\alpha \mid s, r_{mid}] = \frac{\sigma_\alpha^{-2}}{\sigma_\alpha^{-2} + \sigma_s^{-2} + \sigma_r^{-2}} \cdot 0 + \frac{\sigma_s^{-2} s + \sigma_r^{-2} r_{mid}}{\sigma_\alpha^{-2} + \sigma_s^{-2} + \sigma_r^{-2}}$$

More compactly, with precision weights $\lambda_i = 1/\sigma_i^2$:

$$\hat{\alpha} = \frac{\lambda_s \cdot s + \lambda_r \cdot r_{mid}}{\lambda_\alpha + \lambda_s + \lambda_r}$$

**IC and holding period.** The Information Coefficient is the rank correlation between signal and realized return. IC decays roughly as:

$$IC(T) \approx IC_0 \cdot e^{-\kappa T}$$

where $\kappa$ is the signal decay rate. A longer holding period introduces more noise, shrinking the signal-to-noise ratio. In practice at Graham-style macro funds, signals with $IC_0 \approx 0.05$ and daily decay $\kappa \approx 0.01$ have a half-life of ~$\ln 2 / \kappa \approx 70$ days.

**Position sizing implication.** By the Fundamental Law of Active Management:

$$IR \approx IC \cdot \sqrt{BR}$$

After Bayesian update, replace $IC$ with the posterior-weighted $IC_{updated}$. The optimal position in a mean-variance framework is proportional to $\hat{\alpha} / \sigma^2$.

```
Signal path diagram:
  Prior α ──┬──► s (noisy) ──► Posterior α̂ ──► Position size
            │
            └──► r_mid (mid-period) ──► Bayes update ──► Shrinkage
```

[🔝 Back to Top](#-table-of-contents)

---

<a name="q2"></a>
### Q2 — Order Statistics: Expected Minimum of N Uniform Draws
> *"You draw N independent samples from Uniform[0,1]. What is the expected value of the minimum? Now generalize: what is $\mathbb{E}[X_{(k)}]$, the k-th order statistic?"*
> 🗓️ *Reported: Blind / QuantNet, Q4 2025*

**Answer:**

For $X_1, \ldots, X_N \overset{iid}{\sim} \text{Uniform}[0,1]$, the PDF of the $k$-th order statistic $X_{(k)}$ is:

$$f_{X_{(k)}}(x) = \frac{N!}{(k-1)!(N-k)!} x^{k-1}(1-x)^{N-k}, \quad x \in [0,1]$$

This is the $\text{Beta}(k, N-k+1)$ distribution. Therefore:

$$\mathbb{E}[X_{(k)}] = \frac{k}{N+1}$$

**Special cases:**

$$\mathbb{E}[X_{(1)}] = \frac{1}{N+1} \quad \text{(minimum)}$$

$$\mathbb{E}[X_{(N)}] = \frac{N}{N+1} \quad \text{(maximum)}$$

**Intuition:** The $N+1$ "gaps" created by $N$ uniform order statistics have equal expected length $\frac{1}{N+1}$ by symmetry. This is the **spacing argument** — elegant and fast to derive in an interview.

**Variance of $X_{(k)}$:**

$$\text{Var}(X_{(k)}) = \frac{k(N-k+1)}{(N+1)^2(N+2)}$$

**Finance application:** In portfolio optimization, if you have $N$ candidate signals and select the top-$k$, the expected rank of a randomly chosen top signal follows these formulas — critical for understanding selection bias.

[🔝 Back to Top](#-table-of-contents)

---

<a name="q3"></a>
### Q3 — Sharpe Ratio Distribution Under Finite Samples
> *"You observe a strategy with annualized Sharpe = 1.5 over 3 years of daily returns. Is this statistically significant? What is the sampling distribution of the Sharpe ratio?"*
> 🗓️ *Reported: eFinancialCareers / Glassdoor, Q1 2026*

**Answer:**

Let $\{r_t\}_{t=1}^T$ be daily returns with mean $\mu$ and std $\sigma$. The sample Sharpe is $\hat{S} = \sqrt{252} \cdot \frac{\hat{\mu}}{\hat{\sigma}}$.

**Lo (2002) result:** Under iid normality, the asymptotic distribution is:

$$\sqrt{T}\left(\hat{S} - S\right) \overset{d}{\rightarrow} \mathcal{N}\!\left(0, 1 + \frac{S^2}{2}\right)$$

So the standard error of $\hat{S}$ is approximately:

$$\text{SE}(\hat{S}) \approx \sqrt{\frac{1 + S^2/2}{T}}$$

**Numerical check for the question:** $T = 3 \times 252 = 756$ daily observations, $\hat{S} = 1.5$ (annualized). Daily Sharpe $= 1.5/\sqrt{252} \approx 0.0945$.

$$\text{SE}(\hat{S}_{annual}) \approx \sqrt{\frac{1 + 1.5^2/2}{756}} \approx \sqrt{\frac{2.125}{756}} \approx 0.053$$

$$z = \frac{1.5 - 0}{0.053 \cdot \sqrt{252}} \approx \frac{1.5}{0.840} \approx 1.79 \quad (p \approx 0.037)$$

Marginally significant at 5% level. **But** this ignores:
- Non-normality / fat tails → Bailey & López de Prado's *Probabilistic Sharpe Ratio* (PSR)
- Autocorrelation in returns → Lo's correction with AR terms
- **Multiple testing** (see Q4)

**PSR:** The probability that the true Sharpe exceeds a benchmark $S^*$:

$$\text{PSR}(S^*) = \Phi\!\left(\frac{(\hat{S} - S^*)\sqrt{T-1}}{\sqrt{1 - \hat{\gamma}_3 \hat{S} + \frac{\hat{\gamma}_4 - 1}{4}\hat{S}^2}}\right)$$

where $\hat{\gamma}_3$ is skewness and $\hat{\gamma}_4$ is excess kurtosis.

[🔝 Back to Top](#-table-of-contents)

---

<a name="q4"></a>
### Q4 — Multiple Testing & False Discovery Rate in Factor Discovery
> *"You run 1,000 factor backtests and find 50 with Sharpe > 1. How many are likely false positives? How do you control the FDR?"*
> 🗓️ *Reported: Blind / QuantNet, Q2 2026*

**Answer:**

**The problem.** Under the null (no alpha), if each test has significance level $\alpha = 0.05$, you expect $1000 \times 0.05 = 50$ false positives purely by chance. Finding 50 "significant" factors means potentially *all* of them are spurious.

**Bonferroni correction (FWER):** Reject null only if $p_i < \alpha/m = 0.05/1000 = 5 \times 10^{-5}$. Very conservative for large $m$.

**Benjamini–Hochberg (BH) procedure for FDR control:**

Sort p-values: $p_{(1)} \leq p_{(2)} \leq \cdots \leq p_{(m)}$. Find the largest $k$ such that:

$$p_{(k)} \leq \frac{k}{m} \cdot q$$

where $q$ is the target FDR (e.g., $q = 0.05$). Reject all $H_{(1)}, \ldots, H_{(k)}$.

**Harvey, Liu & Zhu (2016)** threshold: For factor discovery in finance with $m$ tests, the minimum required $t$-stat has risen from 2.0 (pre-1990s) to $\geq 3.0$ by 2016, suggesting we need even higher bars today.

**Bailey & López de Prado — Deflated Sharpe Ratio (DSR):**

$$\text{DSR}(\hat{S}) = \text{PSR}\!\left(S^* = \sqrt{\text{Var}[\hat{S}^{\max}]}\left[(1-\gamma)\Phi^{-1}\!\left(1 - \frac{1}{N}\right) + \gamma \Phi^{-1}\!\left(1 - \frac{1}{Ne}\right)\right]\right)$$

This adjusts for the maximum of $N$ iid Sharpe estimates.

```
Multiple Testing Control Flow:
  1,000 backtests
       │
       ▼
  Compute p-values
       │
  ┌────┴────────────────┐
  │ FWER (Bonferroni)   │  ← very few pass
  │ FDR  (BH)           │  ← ~5% of rejects are FP
  │ DSR  (Bailey–Lopez) │  ← adjusts for selection bias
  └─────────────────────┘
       │
  Surviving factors → OOS validation
```

[🔝 Back to Top](#-table-of-contents)

---

<a name="q5"></a>
### Q5 — Copula Dependence vs Linear Correlation
> *"Two assets have zero Pearson correlation. Does that mean they are independent? Explain copulas and their role in tail-risk modeling."*
> 🗓️ *Reported: eFinancialCareers / Glassdoor, Q3 2025*

**Answer:**

**No.** Zero correlation implies independence *only* for jointly Gaussian variables. In general, $\text{Cov}(X,Y) = 0 \not\Rightarrow X \perp Y$.

**Counterexample:** Let $X \sim \mathcal{N}(0,1)$ and $Y = X^2$. Then $\text{Cov}(X,Y) = \mathbb{E}[X^3] - \mathbb{E}[X]\mathbb{E}[X^2] = 0$, yet $Y$ is completely determined by $X$.

**Sklar's Theorem.** Every joint CDF $H(x,y)$ with marginals $F(x)$ and $G(y)$ can be written as:

$$H(x,y) = C(F(x), G(y))$$

where $C: [0,1]^2 \to [0,1]$ is a **copula** — a joint distribution on uniform marginals. The copula fully captures the dependence structure *separately* from the marginals.

**Key copula families:**

| Copula | Tail Dependence | Finance Use |
|--------|----------------|-------------|
| Gaussian | None ($\lambda_U = \lambda_L = 0$) | Overestimates diversification in crises |
| Student-$t$ | Symmetric ($\lambda_U = \lambda_L > 0$) | Better for equity portfolios |
| Clayton | Lower tail only | Credit default modeling |
| Gumbel | Upper tail only | Extreme gain co-movements |

**Tail dependence coefficient:**

$$\lambda_U = \lim_{u \to 1^-} P(Y > F_Y^{-1}(u) \mid X > F_X^{-1}(u))$$

Gaussian copula has $\lambda_U = 0$ regardless of $\rho < 1$, which is why it *failed* during the 2008 credit crisis (the Li Gaussian copula CDO model).

**At Graham:** Copula-based dependence modeling appears in cross-asset tail-risk hedging and correlation breakdown analysis — directly relevant to their published research on *Correlation Breakdown and Liquidity Fragility* (2026).

[🔝 Back to Top](#-table-of-contents)

---

## 📈 B. Stochastic Calculus

<a name="q6"></a>
### Q6 — Itô's Lemma Applied to Log-Price Dynamics
> *"If $S_t$ follows GBM $dS_t = \mu S_t \, dt + \sigma S_t \, dW_t$, derive the SDE for $\log S_t$ using Itô's Lemma. Why does the drift change?"*
> 🗓️ *Reported: Glassdoor / QuantNet, Q1 2026*

**Answer:**

**Itô's Lemma.** For $f(t, S_t)$ with $S_t$ following an Itô process:

$$df = \frac{\partial f}{\partial t}dt + \frac{\partial f}{\partial S}dS_t + \frac{1}{2}\frac{\partial^2 f}{\partial S^2}(dS_t)^2$$

using the Itô rule $(dW_t)^2 = dt$.

**Apply to $f(S) = \ln S$:**

$$\frac{\partial f}{\partial t} = 0, \quad \frac{\partial f}{\partial S} = \frac{1}{S}, \quad \frac{\partial^2 f}{\partial S^2} = -\frac{1}{S^2}$$

$$d\ln S_t = \frac{1}{S_t} dS_t - \frac{1}{2S_t^2}(dS_t)^2$$

Substituting $dS_t = \mu S_t \, dt + \sigma S_t \, dW_t$ and $(dS_t)^2 = \sigma^2 S_t^2 \, dt$:

$$\boxed{d\ln S_t = \left(\mu - \frac{\sigma^2}{2}\right)dt + \sigma \, dW_t}$$

**Why does the drift decrease by $\sigma^2/2$?** This is **Jensen's inequality** in action. Since $\ln$ is concave:

$$\mathbb{E}[\ln S_T] \leq \ln \mathbb{E}[S_T]$$

The geometric mean is always $\leq$ arithmetic mean. The $-\sigma^2/2$ term corrects for the convexity of the exponential function — volatility *drags* long-run compounding. This is the **volatility drag** or **variance drain**.

**Integrated form:**

$$\ln S_T = \ln S_0 + \left(\mu - \frac{\sigma^2}{2}\right)T + \sigma W_T$$

$$S_T = S_0 \exp\!\left[\left(\mu - \frac{\sigma^2}{2}\right)T + \sigma W_T\right]$$

[🔝 Back to Top](#-table-of-contents)

---

<a name="q7"></a>
### Q7 — Girsanov Theorem and Risk-Neutral Measure
> *"Explain the Girsanov theorem. How does it connect the physical measure $\mathbb{P}$ to the risk-neutral measure $\mathbb{Q}$? Why does this matter for derivatives pricing?"*
> 🗓️ *Reported: eFinancialCareers / QuantNet, Q4 2025*

**Answer:**

**Girsanov's Theorem.** Let $W_t^\mathbb{P}$ be a Brownian motion under physical measure $\mathbb{P}$. Define the Radon–Nikodym derivative:

$$\frac{d\mathbb{Q}}{d\mathbb{P}}\bigg|_{\mathcal{F}_T} = \exp\!\left(-\int_0^T \theta_t \, dW_t^\mathbb{P} - \frac{1}{2}\int_0^T \theta_t^2 \, dt\right)$$

where $\theta_t = (\mu - r)/\sigma$ is the **market price of risk** (Sharpe ratio of the asset). Then under $\mathbb{Q}$, the process:

$$W_t^\mathbb{Q} = W_t^\mathbb{P} + \int_0^t \theta_s \, ds$$

is a standard Brownian motion (Novikov's condition must hold).

**Under $\mathbb{Q}$**, GBM becomes:

$$dS_t = r S_t \, dt + \sigma S_t \, dW_t^\mathbb{Q}$$

The drift changes from $\mu$ to $r$ (risk-free rate). This is **risk-neutralization** — investors are compensated for systematic risk, so under $\mathbb{Q}$ all assets grow at $r$.

**Pricing implication.** No-arbitrage price of a derivative $V$:

$$V_0 = e^{-rT} \mathbb{E}^\mathbb{Q}[V_T]$$

```
Measure change diagram:

  P-world (real world)          Q-world (risk-neutral)
  ┌─────────────────┐           ┌─────────────────┐
  │ drift = μ       │           │ drift = r        │
  │ W_t^P           │──Girsanov─► W_t^Q            │
  │ θ = (μ-r)/σ    │           │ no risk premium  │
  └─────────────────┘           └─────────────────┘
```

**At Graham:** The Girsanov framework underpins any options overlay on systematic strategies — crucial when the quant team manages Greeks across macro positions.

[🔝 Back to Top](#-table-of-contents)

---

<a name="q8"></a>
### Q8 — Ornstein–Uhlenbeck Process & Mean-Reversion Half-Life
> *"Write the SDE for an OU process. How do you estimate the mean-reversion speed from data? Derive the half-life."*
> 🗓️ *Reported: Blind / Wall Street Oasis, Q2 2026*

**Answer:**

**OU SDE:**

$$dX_t = \kappa(\theta - X_t) \, dt + \sigma \, dW_t$$

where $\kappa > 0$ is mean-reversion speed, $\theta$ is the long-run mean, and $\sigma$ is volatility.

**Solution:**

$$X_t = \theta + (X_0 - \theta)e^{-\kappa t} + \sigma \int_0^t e^{-\kappa(t-s)} dW_s$$

**Moments:**

$$\mathbb{E}[X_t] = \theta + (X_0 - \theta)e^{-\kappa t}$$

$$\text{Var}(X_t) = \frac{\sigma^2}{2\kappa}\left(1 - e^{-2\kappa t}\right) \xrightarrow{t\to\infty} \frac{\sigma^2}{2\kappa}$$

**Half-life.** The expected deviation from $\theta$ halves when $e^{-\kappa t_{1/2}} = 1/2$:

$$\boxed{t_{1/2} = \frac{\ln 2}{\kappa}}$$

**Estimation via OLS (discrete Euler scheme).** Regress $X_{t+\Delta t} - X_t$ on $X_t$:

$$X_{t+\Delta} - X_t = \alpha + \beta X_t + \epsilon_t$$

Then: $\hat{\kappa} = -\ln(1 + \hat{\beta}) / \Delta$, $\hat{\theta} = -\hat{\alpha}/\hat{\beta}$.

Alternatively, use MLE for exact discrete likelihood of the AR(1) representation:

$$X_{t+\Delta} = X_t e^{-\kappa\Delta} + \theta(1-e^{-\kappa\Delta}) + \eta_t, \quad \eta_t \sim \mathcal{N}\!\left(0, \frac{\sigma^2}{2\kappa}(1-e^{-2\kappa\Delta})\right)$$

**Finance application:** Pairs trading, spread trading, and CTA carry strategies all rely on calibrating OU parameters. Graham's Quant Macro strategies likely use OU-based signals for interest rate spreads and commodity basis.

[🔝 Back to Top](#-table-of-contents)

---

<a name="q9"></a>
### Q9 — Feynman–Kac Formula and PDE Connection
> *"State the Feynman–Kac theorem. How does it link PDEs to stochastic processes? Derive the Black–Scholes PDE from it."*
> 🗓️ *Reported: QuantNet / eFinancialCareers, Q3 2025*

**Answer:**

**Feynman–Kac.** Consider the SDE $dX_t = \mu(X_t,t)dt + \sigma(X_t,t)dW_t$ and the function:

$$u(x,t) = \mathbb{E}^\mathbb{Q}\!\left[e^{-\int_t^T r(X_s,s)ds} f(X_T) \,\Big|\, X_t = x\right]$$

Then $u$ satisfies the PDE:

$$\frac{\partial u}{\partial t} + \mu \frac{\partial u}{\partial x} + \frac{1}{2}\sigma^2 \frac{\partial^2 u}{\partial x^2} - r \cdot u = 0$$

with terminal condition $u(x,T) = f(x)$.

**Black–Scholes derivation.** Under $\mathbb{Q}$, $dS_t = r S_t dt + \sigma S_t dW_t^\mathbb{Q}$. An option with payoff $f(S_T)$ has price $V(S,t) = e^{-r(T-t)}\mathbb{E}^\mathbb{Q}[f(S_T) | S_t = S]$.

Applying Feynman–Kac with $\mu(S) = rS$, $\sigma(S) = \sigma S$, $r(x,t) = r$:

$$\frac{\partial V}{\partial t} + rS\frac{\partial V}{\partial S} + \frac{1}{2}\sigma^2 S^2 \frac{\partial^2 V}{\partial S^2} - rV = 0$$

This is the **Black–Scholes PDE** — recovered as the Kolmogorov backward equation.

```
Duality:
  Stochastic (Monte Carlo)        Deterministic (PDE)
  ┌──────────────────────┐        ┌──────────────────────┐
  │ Simulate paths of X  │        │ Solve BS PDE         │
  │ Average payoffs      │◄──FK──►│ ∂V/∂t + Lu - rV = 0 │
  │ Discount at r        │        │ Terminal: V(S,T)=f(S) │
  └──────────────────────┘        └──────────────────────┘
```

[🔝 Back to Top](#-table-of-contents)

---

## 🔢 C. Linear Algebra

<a name="q10"></a>
### Q10 — PCA for Factor Extraction from Returns Covariance
> *"You have a $p \times T$ matrix of asset returns where $p = 500$ and $T = 252$. Describe how you use PCA for factor extraction. What are the limitations?"*
> 🗓️ *Reported: Glassdoor / Blind, Q1 2026*

**Answer:**

**Setup.** Let $R$ be the $p \times T$ demeaned returns matrix. Sample covariance: $\hat{\Sigma} = \frac{1}{T-1}RR^\top \in \mathbb{R}^{p \times p}$.

**PCA via eigendecomposition:**

$$\hat{\Sigma} = V \Lambda V^\top, \quad \Lambda = \text{diag}(\lambda_1 \geq \lambda_2 \geq \cdots \geq \lambda_p)$$

The $k$-th principal component is $F_k = V_k^\top R$, the projection of returns onto the $k$-th eigenvector (eigenportfolio).

**Factor loadings** $B \in \mathbb{R}^{p \times K}$: the matrix of top-$K$ eigenvectors. Residual returns: $\epsilon = R - BF$.

**Explained variance:**

$$\%\ \text{explained by top } K = \frac{\sum_{k=1}^K \lambda_k}{\sum_{k=1}^p \lambda_k}$$

**Limitations — critical for interview:**

| Issue | Description |
|-------|-------------|
| $p > T$ | Covariance matrix is rank-deficient; spurious eigenvalues |
| Marchenko–Pastur | Eigenvalues below MP upper bound are noise (see Q11) |
| Instability | Sample eigenvectors rotate significantly with new data |
| No interpretation | PCs are linear combinations; no economic meaning |
| Survivorship bias | $p$ changes as stocks enter/exit universe |

**Practical fix:** Use shrinkage estimators (Ledoit–Wolf), or Random Matrix Theory to separate signal from noise eigenvalues.

[🔝 Back to Top](#-table-of-contents)

---

<a name="q11"></a>
### Q11 — Eigenvalue Stability and Marchenko–Pastur Law
> *"What is the Marchenko–Pastur distribution? How do you use it to determine how many meaningful factors exist in a returns matrix?"*
> 🗓️ *Reported: eFinancialCareers / QuantNet, Q4 2025*

**Answer:**

**Setup.** Let $q = p/T$ be the aspect ratio. For a random $p \times T$ matrix with iid $\mathcal{N}(0,1/T)$ entries, the empirical spectral distribution of the sample covariance matrix converges to the **Marchenko–Pastur law**:

$$\rho(\lambda) = \frac{T}{2\pi p} \cdot \frac{\sqrt{(\lambda_+ - \lambda)(\lambda - \lambda_-)}}{\lambda}, \quad \lambda \in [\lambda_-, \lambda_+]$$

where the bulk edges are:

$$\lambda_{\pm} = \sigma^2\left(1 \pm \sqrt{q}\right)^2$$

**Key insight:** Any eigenvalue of $\hat{\Sigma}$ **above** $\lambda_+$ contains genuine signal (not explainable by pure noise). All eigenvalues in $[\lambda_-, \lambda_+]$ are noise.

**Algorithm:**
1. Compute $q = p/T$, $\hat{\sigma}^2 = \frac{1}{p}\text{tr}(\hat{\Sigma})$
2. Compute $\lambda_+ = \hat{\sigma}^2(1+\sqrt{q})^2$
3. Count eigenvalues $> \lambda_+$ — these are the $K^*$ signal factors

**Example:** $p=500$, $T=252$, $q \approx 1.98$, $\hat{\sigma}^2=1$:

$$\lambda_+ = (1 + \sqrt{1.98})^2 \approx (1 + 1.407)^2 \approx 5.79$$

Any eigenvalue $> 5.79$ is a genuine factor.

```
Eigenvalue spectrum:
  ▲
  │   Noise bulk      │ Signal
  │  (MP distribution)│ eigenvalues
  │  ┌───────────┐    │  ×  ×  ×
  │  │░░░░░░░░░░░│    │
  └──┴───────────┴────┴──────────►
     λ_-          λ+              λ
```

[🔝 Back to Top](#-table-of-contents)

---

<a name="q12"></a>
### Q12 — Regularization: Ridge vs LASSO in High-Dimensional Factor Regression
> *"You're fitting a cross-sectional return model with 300 candidate features and $T=252$ observations. Compare Ridge and LASSO regularization. When would you use each?"*
> 🗓️ *Reported: Blind / Glassdoor, Q2 2026*

**Answer:**

**Ridge (L2):** Adds penalty $\lambda \|\beta\|_2^2$ to least-squares loss:

$$\hat{\beta}^{ridge} = \arg\min_\beta \|y - X\beta\|^2 + \lambda\|\beta\|_2^2 = (X^\top X + \lambda I)^{-1}X^\top y$$

**LASSO (L1):** Adds penalty $\lambda \|\beta\|_1$:

$$\hat{\beta}^{lasso} = \arg\min_\beta \|y - X\beta\|^2 + \lambda\|\beta\|_1$$

No closed form; solved via coordinate descent or LARS.

| Property | Ridge | LASSO |
|----------|-------|-------|
| Solution sparsity | Dense (shrinks toward 0) | Sparse (exact zeros) |
| Handles correlated predictors | Yes (distributes weight) | Picks one arbitrarily |
| Differentiability | Everywhere | Not at 0 |
| Geometric intuition | L2 ball (sphere) | L1 ball (diamond — corners→zeros) |
| Degrees of freedom | $\text{tr}(H)$ where $H = X(X^\top X+\lambda I)^{-1}X^\top$ | Number of nonzero coefficients |

**When to use which in quant research:**

- **Ridge:** When you believe most signals carry some information (dense alpha), or when predictors are highly correlated (e.g., factor loadings). Standard in return attribution.
- **LASSO:** When you seek sparse signal selection — e.g., selecting 10 meaningful features from 300 candidates. Critical for interpretability and avoiding overfitting.
- **Elastic Net:** $\alpha\|\beta\|_1 + (1-\alpha)\|\beta\|_2^2$ — best of both; handles correlated predictors while inducing sparsity.

**Bias-variance tradeoff:**

$$\text{MSE}(\hat{\beta}) = \text{Bias}^2 + \text{Variance}$$

Both Ridge and LASSO increase bias but reduce variance. The optimal $\lambda$ is chosen via cross-validation (with proper time-series purging — see Q16).

[🔝 Back to Top](#-table-of-contents)

---

## 🤖 D. Machine Learning

<a name="q13"></a>
### Q13 — Gradient Boosting for Cross-Sectional Return Prediction
> *"You want to predict next-month returns for a 500-stock universe using 50 features. Walk through your full ML pipeline using gradient boosting. What are the key pitfalls?"*
> 🗓️ *Reported: Blind / eFinancialCareers, Q1 2026*

**Answer:**

**Pipeline:**

```
Raw Features (50)
     │
     ▼
1. Feature Engineering
   - Winsorize at 1%/99%
   - Cross-sectional rank normalize (avoids distribution shifts)
   - Z-score within sector/industry
     │
     ▼
2. Target: IC-weighted next-month return (rank)
     │
     ▼
3. Time-series train/val split (walk-forward, purged — Q16)
     │
     ▼
4. LightGBM / XGBoost with:
   - max_depth = 4-6 (shallow trees → regularization)
   - learning_rate = 0.01-0.05
   - n_estimators = 300-1000 (early stopping on val IC)
   - min_child_samples = 50 (prevent leaf overfitting)
     │
     ▼
5. Evaluation: Rank IC, ICIR, quintile spread, turnover
     │
     ▼
6. SHAP for feature attribution and signal decay analysis
```

**Key pitfalls:**

| Pitfall | Fix |
|---------|-----|
| Look-ahead bias | Strict timestamp alignment; data point-in-time |
| Cross-sectional leakage | Normalize *within* each period after split |
| Overfitting to backtest | Walk-forward OOS; deflated Sharpe (Q4) |
| Feature importance instability | SHAP + cross-validation stability check |
| Survivorship bias | Include delisted stocks; fill with delisting return |
| Transaction costs ignored | Net-of-cost IC = gross IC - $2 \times$ TC/predicted return |

**Hyperparameter tuning:** Bayesian optimization (Optuna) with time-series CV; never random search with standard k-fold.

[🔝 Back to Top](#-table-of-contents)

---

<a name="q14"></a>
### Q14 — Overfitting in Backtest: Walk-Forward Validation Design
> *"Describe the difference between standard k-fold CV and walk-forward validation for time series. Why does standard CV fail in finance?"*
> 🗓️ *Reported: Glassdoor / QuantNet, Q3 2025*

**Answer:**

**Standard k-fold failure modes in finance:**

1. **Look-ahead bias:** Future folds used to predict past — violates causality
2. **Data leakage:** Overlapping return windows create correlation between train/test
3. **Non-stationarity:** Distribution shifts make distant-past data irrelevant
4. **Label leakage:** Returns overlap (e.g., monthly returns computed daily)

**Walk-Forward (Expanding Window):**

```
Fold 1:  [=====Train=====][=Test=]
Fold 2:  [========Train========][=Test=]
Fold 3:  [===========Train===========][=Test=]
                                     ▲
                          Always predict forward
```

**Walk-Forward (Rolling Window):**

```
Fold 1:  [===Train===][=Test=]
Fold 2:       [===Train===][=Test=]
Fold 3:            [===Train===][=Test=]
```

**Purging:** Remove samples from training set whose *labels* overlap with test period. For a 21-day return: purge the last 21 days of training before each test fold.

**Embargo:** After test fold, skip $h$ additional days before resuming training (prevents serial correlation leak):

$$\text{Embargo gap} \approx 0.01 \times T \text{ (López de Prado rule of thumb)}$$

**CPCV (Combinatorial Purged Cross-Validation):** Generates exponentially more train/test paths:
- Choose $k$ test folds from $N$ total folds: $\binom{N}{k}$ combinations
- Provides a distribution of backtest performance, not just a single path

[🔝 Back to Top](#-table-of-contents)

---

<a name="q15"></a>
### Q15 — Recurrent Nets vs Transformers for Time-Series Alpha
> *"Compare LSTMs and Transformer-based models for financial time-series prediction. When would you use each? What are the 2025–2026 best practices?"*
> 🗓️ *Reported: Blind / eFinancialCareers, Q1 2026*

**Answer:**

**LSTM recap:** Gates ($i, f, o, c$) allow selective memory. Hidden state $h_t$ carries temporal context. Effective for sequences with local temporal dependencies.

$$\begin{aligned}
f_t &= \sigma(W_f [h_{t-1}, x_t] + b_f) \quad \text{(forget gate)}\\
i_t &= \sigma(W_i [h_{t-1}, x_t] + b_i) \quad \text{(input gate)}\\
\tilde{c}_t &= \tanh(W_c [h_{t-1}, x_t] + b_c)\\
c_t &= f_t \odot c_{t-1} + i_t \odot \tilde{c}_t
\end{aligned}$$

**Transformer attention:** Self-attention across all positions simultaneously:

$$\text{Attn}(Q,K,V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right)V$$

**Comparison:**

| Dimension | LSTM | Transformer |
|-----------|------|-------------|
| Parallelism | Sequential (slow training) | Fully parallel |
| Long-range dependencies | Vanishing gradient issue | $O(1)$ path length |
| Data hunger | Lower | High ($\geq$ 5 years daily) |
| Interpretability | Hidden state opaque | Attention maps |
| Stationarity sensitivity | High | High (both struggle) |
| 2026 best practice | Still useful for HFT (low latency) | Preferred for long-horizon signals |

**2026 best practices (per QuantVision 2026 and arxiv papers):**
- **Temporal Fusion Transformer (TFT):** Handles mixed-frequency features; built-in interpretable attention
- **PatchTST:** Treat time series as patches (analogous to vision transformers); state-of-the-art on many financial benchmarks
- **Mamba (SSM):** State Space Models — linear complexity; emerging as LSTM replacement for long sequences

**Key caveat:** In live trading, the edge from deep learning vs linear models in low-frequency (daily/weekly) strategies is often marginal after transaction costs. The real value is in *feature discovery*, not model complexity per se.

[🔝 Back to Top](#-table-of-contents)

---

<a name="q16"></a>
### Q16 — Purging and Embargoing in Financial Cross-Validation
> *"You're backtesting a monthly signal on daily data. Your 21-day forward returns overlap. Precisely describe the purging and embargo procedure."*
> 🗓️ *Reported: Glassdoor / Blind, Q2 2026*

**Answer:**

**The problem.** If observation $i$ uses returns $r_{t_i \to t_i+21}$ as label, and observation $j$ with $t_j \in [t_i, t_i+21]$ is in the training set, the labels of $i$ and $j$ are correlated. Standard CV treats them as independent — this leaks future information.

**Precise algorithm:**

```
For each test fold [t_test_start, t_test_end]:

  1. PURGE: Remove from training all observations i where
     their label window [t_i, t_i + h] overlaps with
     [t_test_start, t_test_end]
     
     i.e., remove i where t_i + h > t_test_start
                    AND t_i < t_test_end

  2. EMBARGO: Additionally remove the h days AFTER t_test_end
     from the training set
     
     i.e., remove i where t_i ∈ [t_test_end, t_test_end + h]
```

**Mathematically:** For label horizon $h=21$ days:
- Purge: observations $i$ with $t_i^{end} > t_{test}^{start}$ and $t_i^{start} < t_{test}^{end}$
- Embargo: observations $i$ with $t_i \in [t_{test}^{end}, t_{test}^{end} + h]$

**Implementation in Python sketch:**

```python
def purge_embargo(train_idx, test_idx, times, horizon, embargo_pct=0.01):
    T = len(times)
    embargo = int(T * embargo_pct)
    t0, t1 = times[test_idx[0]], times[test_idx[-1]]
    # purge: label window overlaps test
    purged = [i for i in train_idx
              if not (times[i] + horizon < t0 or times[i] > t1)]
    # embargo: after test
    embargoed = [i for i in train_idx
                 if t1 <= times[i] <= t1 + embargo]
    return [i for i in train_idx
            if i not in purged and i not in embargoed]
```

[🔝 Back to Top](#-table-of-contents)

---

## 🧬 E. Artificial Intelligence

<a name="q17"></a>
### Q17 — LLMs for Alternative Data Signal Extraction (2026)
> *"How would you build a pipeline to extract alpha signals from earnings call transcripts using LLMs? What are the key risks of deploying LLMs in live trading?"*
> 🗓️ *Reported: Blind / QuantVision 2026 conference intel, Q1 2026*

**Answer:**

**Pipeline Architecture:**

```
Earnings Call Transcript (PDF/Audio)
         │
         ▼
1. Preprocessing
   - Whisper (speech-to-text if audio)
   - Clean: remove boilerplate, Q&A split
         │
         ▼
2. LLM Feature Extraction (claude-sonnet / GPT-4o)
   - Sentiment score [-1, +1] for tone
   - Forward guidance: quantitative vs qualitative
   - Management uncertainty: hedge word count
   - Novel disclosures vs. guidance repetition
         │
         ▼
3. Structured Output (JSON schema enforcement)
   {sentiment, guidance_delta, uncertainty_idx, topics[]}
         │
         ▼
4. Cross-sectional signal construction
   - Normalize within sector/quarter
   - Combine with price reaction (surprise alpha)
         │
         ▼
5. Backtest with proper purging (Q16)
   - IC analysis by sector, market cap
   - Decay analysis (half-life of signal)
```

**Key risks in live deployment (per arxiv 2605.05211, 2026):**

| Risk | Mitigation |
|------|-----------|
| Sentiment fragility | Ensemble multiple LLM prompts; measure prompt sensitivity |
| Look-ahead in fine-tuning | Strict temporal cutoff; never fine-tune on post-event data |
| Data leakage via embeddings | Audit embedding model training data cutoff |
| Hallucination | Structured output + validation against known facts |
| Illiquidity premium confusion | Separate small-cap LLM alpha from liquidity risk premium |
| Regulatory/NLP compliance | Ensure transcript usage complies with Reg FD |

**2026 frontier:** Multimodal models combining transcript *text* + *vocal tone* (prosody features) + *slide deck images* are showing incremental IC gains over text-only. Graham's approach per their research: combining diverse data types where the combination yields genuinely novel information.

[🔝 Back to Top](#-table-of-contents)

---

<a name="q18"></a>
### Q18 — Reinforcement Learning for Execution Optimization
> *"How would you frame optimal trade execution as a reinforcement learning problem? Define the state space, action space, and reward function."*
> 🗓️ *Reported: eFinancialCareers / Blind, Q4 2025*

**Answer:**

**MDP Formulation:**

$$(\mathcal{S}, \mathcal{A}, \mathcal{R}, \mathcal{P}, \gamma)$$

**State space $\mathcal{S}$** at time step $t$:

$$s_t = \left(q_t, \bar{q}, \tau_t, \sigma_t^{mid}, \text{spread}_t, \text{OBI}_t, \text{VWAP}_t^{ref}, \text{imbalance}_t\right)$$

where $q_t$ = remaining inventory, $\bar{q}$ = total order size, $\tau_t$ = time remaining, $\text{OBI}$ = order book imbalance.

**Action space $\mathcal{A}$:**
- Discrete: {aggressive, passive, cancel, wait} per time step
- Continuous: fraction of remaining shares to execute $a_t \in [0,1]$

**Reward function:**

$$r_t = -(\text{market impact}_t + \text{timing cost}_t + \text{opportunity cost}_t)$$

More precisely, implementation shortfall reward:

$$R = -\sum_t \left[a_t \cdot q_t \cdot (p_t - p_0)\right] - \lambda \cdot \text{Var}\left[\sum_t a_t q_t p_t\right]$$

where $p_0$ is the decision price and $\lambda$ controls the risk aversion.

**Algorithm choice:**
- **PPO / SAC** for continuous action spaces (works well for TWAP/VWAP enhancement)
- **DQN** for discrete action spaces
- **DDPG** for deterministic policy with continuous actions

**Almgren–Chriss benchmark:** The classical IS minimization gives closed-form TWAP-like strategies. RL adds value in non-linear market impact, intraday seasonality, and LOB dynamics.

**Production considerations:** Reward shaping to avoid market manipulation signals; latency constraints; sim-to-real gap from backtest to live market microstructure.

[🔝 Back to Top](#-table-of-contents)

---

<a name="q19"></a>
### Q19 — Causal Inference vs Correlation in AI-Driven Alpha
> *"Your ML model finds that a certain alternative data feature has high predictive power. How do you determine if this is a genuine causal relationship vs spurious correlation?"*
> 🗓️ *Reported: QuantVision 2026 / Blind, Q2 2026*

**Answer:**

**The core problem.** A feature $X$ predicting returns $Y$ in-sample could be:
1. Causally driving $Y$
2. Both $X$ and $Y$ caused by confounder $Z$
3. $Y$ causing $X$ (reverse causality)
4. Pure spurious correlation (multiple testing — Q4)

**Tools for causal discovery:**

**1. Granger Causality (time-series):**
$X$ Granger-causes $Y$ if past $X$ predicts $Y$ beyond past $Y$ alone:

$$Y_t = \alpha + \sum_{k=1}^p \beta_k Y_{t-k} + \sum_{k=1}^p \gamma_k X_{t-k} + \epsilon_t$$

Test $H_0: \gamma_1 = \cdots = \gamma_p = 0$ via F-test. **Limitation:** Granger causality ≠ true causality; can be driven by confounders.

**2. DoWhy / Potential Outcomes Framework:**
Define potential outcomes $Y(X=1)$ and $Y(X=0)$. Estimate ATE:

$$\tau = \mathbb{E}[Y(1) - Y(0)]$$

via IPW, doubly-robust estimators, or instrumental variables.

**3. Structural Break Tests:**
If the relationship is causal, it should be stable across regime changes (COVID, GFC, rate cycles). If predictive power vanishes in stress periods, likely spurious.

**4. Economic narrative test:**
Can you construct a plausible mechanism? Signal → [chain of events] → price move. If no mechanism, skepticism warranted.

**5. Out-of-sample in different markets:**
Does the signal work in non-US markets, different time periods, or synthetic universes? Genuine causal relationships tend to generalize.

**Graham's approach:** Systematic strategies demand signals that hold OOS and have an economic rationale — pure data mining without mechanism is explicitly avoided per their research culture.

[🔝 Back to Top](#-table-of-contents)

---

## 💻 F. Coding

<a name="q20"></a>
### Q20 — Vectorized Rolling Sharpe Ratio in Python/NumPy
> *"Write an efficient Python function to compute a rolling Sharpe ratio with window $W$ for a returns series. No loops allowed."*
> 🗓️ *Reported: HackerRank / Glassdoor, Q1 2026*

**Answer:**

```python
import numpy as np
import pandas as pd

def rolling_sharpe(returns: np.ndarray,
                   window: int = 252,
                   annualize: bool = True,
                   periods_per_year: int = 252) -> np.ndarray:
    """
    Vectorized rolling Sharpe ratio.
    
    Parameters
    ----------
    returns : array of daily (or periodic) returns
    window  : rolling window length
    annualize: whether to annualize the Sharpe
    
    Returns
    -------
    sharpe : array same length as returns, NaN for first (window-1) obs
    """
    r = np.asarray(returns, dtype=float)
    n = len(r)
    
    # Use pandas rolling for clean NaN handling at boundaries
    s = pd.Series(r)
    roll_mean = s.rolling(window).mean().values
    roll_std  = s.rolling(window).std(ddof=1).values  # unbiased
    
    sharpe = roll_mean / roll_std
    
    if annualize:
        sharpe = sharpe * np.sqrt(periods_per_year)
    
    return sharpe


# ── Pure NumPy version (faster for large arrays) ──────────────────────
def rolling_sharpe_numpy(returns: np.ndarray, window: int) -> np.ndarray:
    """Strided rolling Sharpe using cumulative sums — O(N) time."""
    r = np.asarray(returns, dtype=float)
    n = len(r)
    out = np.full(n, np.nan)
    
    # Cumulative sum and sum-of-squares for O(1) per window
    cs  = np.concatenate(([0.], np.cumsum(r)))
    cs2 = np.concatenate(([0.], np.cumsum(r**2)))
    
    for i in range(window - 1, n):          # still a loop but minimal
        s1 = cs[i+1]  - cs[i+1-window]
        s2 = cs2[i+1] - cs2[i+1-window]
        mu  = s1 / window
        var = s2 / window - mu**2           # biased, adjust if needed
        if var > 0:
            out[i] = mu / np.sqrt(var) * np.sqrt(252)
    
    return out

# Usage:
# rets = np.random.randn(1000) * 0.01
# sr   = rolling_sharpe(rets, window=63)
```

**Complexity:** $O(N)$ time, $O(N)$ space. The cumsum trick avoids repeated summation per window.

**Edge cases to handle:** Zero std (flat returns → NaN), NaN in input (use `np.nanmean`), negative Sharpe (valid output, not an error).

[🔝 Back to Top](#-table-of-contents)

---

<a name="q21"></a>
### Q21 — Kalman Filter Implementation for Dynamic Beta
> *"Implement a Kalman filter to estimate a time-varying beta between a stock and the market. Write out the full predict-update equations and code them up."*
> 🗓️ *Reported: eFinancialCareers / LeetCode Discuss, Q2 2026*

**Answer:**

**State-space model:** Let $\beta_t$ be the time-varying beta:

$$\beta_t = \beta_{t-1} + \eta_t, \quad \eta_t \sim \mathcal{N}(0, Q) \quad \text{(transition)}$$
$$r_t^{stock} = \beta_t \cdot r_t^{mkt} + \epsilon_t, \quad \epsilon_t \sim \mathcal{N}(0, R) \quad \text{(observation)}$$

**Kalman equations:**

| Step | Equation |
|------|----------|
| Predict state | $\hat{\beta}_{t\|t-1} = \hat{\beta}_{t-1\|t-1}$ |
| Predict variance | $P_{t\|t-1} = P_{t-1\|t-1} + Q$ |
| Innovation | $\nu_t = r_t^s - \hat{\beta}_{t\|t-1} \cdot r_t^m$ |
| Innovation var | $S_t = P_{t\|t-1} \cdot (r_t^m)^2 + R$ |
| Kalman gain | $K_t = P_{t\|t-1} \cdot r_t^m / S_t$ |
| Update state | $\hat{\beta}_{t\|t} = \hat{\beta}_{t\|t-1} + K_t \cdot \nu_t$ |
| Update variance | $P_{t\|t} = (1 - K_t \cdot r_t^m) \cdot P_{t\|t-1}$ |

```python
import numpy as np

def kalman_beta(stock_rets: np.ndarray,
                mkt_rets: np.ndarray,
                Q: float = 1e-4,   # process noise (beta drift)
                R: float = 1e-2,   # obs noise (residual variance)
                beta0: float = 1.0,
                P0: float = 1.0) -> np.ndarray:
    """
    Kalman filter for time-varying market beta.
    Returns array of filtered beta estimates.
    """
    n = len(stock_rets)
    betas = np.zeros(n)
    
    beta_hat = beta0   # prior state estimate
    P = P0             # prior error covariance
    
    for t in range(n):
        x = mkt_rets[t]       # covariate (market return)
        y = stock_rets[t]     # observation (stock return)
        
        # ── Predict ──────────────────────────────
        # beta_{t|t-1} = beta_{t-1|t-1}  (random walk)
        P_pred = P + Q
        
        # ── Update ──────────────────────────────
        innovation = y - beta_hat * x
        S = P_pred * x**2 + R          # innovation variance
        K = P_pred * x / S             # Kalman gain
        
        beta_hat = beta_hat + K * innovation
        P = (1 - K * x) * P_pred
        
        betas[t] = beta_hat
    
    return betas

# Tune Q/R via log-likelihood maximization or EM algorithm
# Higher Q/R ratio → faster adaptation (noisier beta)
# Lower Q/R ratio → smoother beta (slower to adapt)
```

**Tuning $Q$ and $R$:** Maximize the log-likelihood $\sum_t \log \mathcal{N}(\nu_t; 0, S_t)$ via grid search or scipy.optimize. Alternatively, use the EM algorithm (Shumway–Stoffer).

[🔝 Back to Top](#-table-of-contents)

---

<a name="q22"></a>
### Q22 — Cointegration Test (Engle–Granger) from Scratch
> *"Write Python code to test for cointegration between two price series using the Engle–Granger two-step method. Interpret the output."*
> 🗓️ *Reported: HackerRank / Glassdoor, Q3 2025*

**Answer:**

```python
import numpy as np
from scipy import stats

def engle_granger_test(y: np.ndarray,
                       x: np.ndarray,
                       trend: str = 'c') -> dict:
    """
    Engle–Granger two-step cointegration test.
    
    Step 1: OLS regression y_t = alpha + beta * x_t + e_t
    Step 2: ADF test on residuals e_t
    
    Returns dict with {beta, alpha, adf_stat, p_value, is_cointegrated}
    """
    n = len(y)
    
    # ── Step 1: OLS ───────────────────────────────────────────────────
    if trend == 'c':
        X = np.column_stack([np.ones(n), x])
    else:
        X = x.reshape(-1, 1)
    
    beta_hat = np.linalg.lstsq(X, y, rcond=None)[0]
    residuals = y - X @ beta_hat
    
    # ── Step 2: ADF on residuals (no constant — already demeaned) ─────
    def adf_stat(e, max_lags=1):
        """Augmented Dickey-Fuller statistic."""
        de  = np.diff(e)
        e_lag = e[max_lags:-1]          # e_{t-1}
        de_lags = np.column_stack(
            [de[max_lags-k:-k-1] for k in range(1, max_lags+1)]
        ) if max_lags > 0 else None
        
        if de_lags is not None:
            Z = np.column_stack([e_lag, de_lags])
        else:
            Z = e_lag.reshape(-1, 1)
        
        y_adf = de[max_lags:]
        coef = np.linalg.lstsq(Z, y_adf, rcond=None)[0]
        
        residuals_adf = y_adf - Z @ coef
        s2 = np.sum(residuals_adf**2) / (len(y_adf) - Z.shape[1])
        se_rho = np.sqrt(s2 * np.linalg.inv(Z.T @ Z)[0, 0])
        
        return coef[0] / se_rho   # t-stat on lagged level
    
    tau = adf_stat(residuals, max_lags=1)
    
    # Engle-Granger critical values (MacKinnon 1991, n=2 variables)
    # Approximate 1%, 5%, 10% critical values
    cv = {'1%': -3.90, '5%': -3.34, '10%': -3.04}
    p_approx = stats.norm.cdf(tau)  # rough; use MacKinnon tables in prod
    
    return {
        'alpha': beta_hat[0] if trend == 'c' else 0,
        'beta':  beta_hat[1] if trend == 'c' else beta_hat[0],
        'adf_stat': tau,
        'critical_values': cv,
        'is_cointegrated_5pct': tau < cv['5%'],
        'residuals': residuals
    }

# Interpretation:
# If ADF stat < critical value → reject unit root in spread
# → series are cointegrated → pairs trade is valid
```

**Economic interpretation:** If $(y_t, x_t)$ are cointegrated with cointegrating vector $(1, -\hat{\beta})$, the spread $e_t = y_t - \hat{\beta} x_t$ is stationary — it mean-reverts. This is the theoretical foundation for **pairs trading** and **statistical arbitrage**.

[🔝 Back to Top](#-table-of-contents)

---

## 🧩 G. Reasoning

<a name="q23"></a>
### Q24 — Signal Capacity: When Does Alpha Decay with AUM?
> *"You discover a signal with estimated alpha of 30 bps/month on a $100M portfolio. How do you think about capacity — at what AUM does the signal become unprofitable?"*
> 🗓️ *Reported: Wall Street Oasis / Blind, Q1 2026*

**Answer:**

**Framework:** Alpha decays with AUM due to market impact. The signal is profitable while:

$$\alpha(AUM) - TC(AUM) > 0$$

**Transaction cost model (square-root impact):**

$$TC \approx \sigma \sqrt{\frac{Q}{V}} \cdot \psi$$

where $Q$ = order size, $V$ = daily volume, $\sigma$ = daily volatility, $\psi$ = market impact coefficient (~0.5–1.0).

**Capacity derivation.** For a universe of $N$ stocks each with daily volume $V_i$, total capacity:

$$C \approx \frac{\alpha^2}{\sigma^2} \cdot \frac{T}{\text{turnover}} \cdot \sum_i \frac{V_i}{\psi}$$

**Practical steps:**

```
1. Estimate gross alpha: α_gross = 30 bps/month (given)
2. Measure average daily turnover: e.g., 20% per month
3. For AUM = X, average trade size ≈ X × 0.20 / (N × trading_days)
4. Compute TC(X) = σ × sqrt(trade_size / avg_daily_volume) × ψ × turnover
5. Solve: α_gross = TC(X_capacity) → X_capacity
6. Apply 50% safety margin: deploy at X_capacity / 2
```

**Example:** Signal $\alpha = 30$ bps/month, 500 stocks, avg daily volume $10M/stock, $\sigma = 1.5\%/day, $\psi = 0.5$, monthly turnover = 20%:

$$X_{cap} \approx \frac{(0.003)^2}{(0.015)^2} \times \frac{1}{0.20} \times 500 \times 10M \approx \$500M$$

**The meta-answer Graham wants:** Strong candidates understand that capacity analysis must be done *before* committing capital, factoring in: signal correlation with other strategies in the book (correlated strategies share capacity), liquidity in stress periods (capacity falls with volatility), and alpha decay speed (fast-decaying signals have lower capacity).

[🔝 Back to Top](#-table-of-contents)

---

<a name="q24"></a>
### Q24 — Design a Macro Momentum Strategy from Scratch
> *"Design a cross-asset momentum strategy from first principles. How do you form the signal, size positions, and manage risk? What would make Graham's version different?"*
> 🗓️ *Reported: Wall Street Oasis, Q2 2026*

**Answer:**

**Signal Construction:**

$$\text{MomSignal}_{i,t} = \frac{r_{i,t-12m \to t-1m}}{\sigma_{i,t}^{12m}}$$

Risk-adjusted 12-1 momentum (exclude last month to avoid short-term reversal). Apply across: equities (country indices), fixed income (10Y futures), FX (G10), commodities (CRB components).

**Cross-sectional ranking:**

$$\tilde{z}_{i,t} = \frac{\text{rank}(\text{MomSignal}_{i,t}) - 0.5}{N}$$

Normalize to $[-0.5, 0.5]$ for stable position sizing.

**Position sizing — risk parity:**

$$w_{i,t} = \frac{\tilde{z}_{i,t}}{\sigma_{i,t}^{realized}} \cdot \frac{\sigma^{target}}{N}$$

Scale to target portfolio volatility $\sigma^{target}$ (e.g., 10% annualized).

**Risk management layers:**

```
Layer 1: Position limits (max 5% risk per asset)
Layer 2: Sector/asset class limits (max 30% gross by class)
Layer 3: Correlation-adjusted risk (use realized correlation matrix)
Layer 4: Drawdown control: scale down if 30-day DD > 5%
Layer 5: Macro regime filter: reduce exposure in high VIX environments
```

**What makes Graham's version different:** Their Quant Macro strategy (launched 2014) combines systematic signals with macro discretion. Their edge is in: (1) blending momentum with value/carry in a regime-aware way, (2) incorporating macro fundamentals as signal conditioning variables, and (3) Graham's infrastructure for rapid signal iteration and risk management (daily risk meetings since founding).

[🔝 Back to Top](#-table-of-contents)

---

<a name="q25"></a>
### Q25 — Transaction Cost Model and Signal Threshold Calibration
> *"How do you decide whether to trade on a signal, given transaction costs? Derive the optimal signal threshold."*
> 🗓️ *Reported: eFinancialCareers / Blind, Q4 2025*

**Answer:**

**Setup.** Signal $s_t$ predicts return $r_{t+1} = \beta s_t + \epsilon$. Trading incurs round-trip cost $c$ (bid-ask + market impact). We trade only if expected profit exceeds cost.

**Optimal threshold:** Trade if $|\hat{r}| > c$, i.e., $|\beta s_t| > c$:

$$|s_t| > s^* = \frac{c}{\beta}$$

**Continuous position (Gârleanu–Pedersen 2013):** With quadratic TC and AR(1) signal:

$$x_t^* = (1-\rho_\lambda) \cdot \sum_{k=0}^\infty \rho_\lambda^k \cdot \hat{r}_{t+k|t}$$

where $\rho_\lambda = \frac{\lambda}{\lambda + \gamma \sigma^2}$, $\lambda$ = TC rate, $\gamma$ = risk aversion.

**Practical calibration:**

$$s^* = \frac{\text{2-way TC}}{\text{IC} \times \sigma_{returns} \times \text{signal volatility}}$$

```
Decision rule:

  |signal| > s*?
  ┌─── YES ───┐
  │            │
Trade (full)  Trade (partial via GP formula)
  │            │
  └─────┬──────┘
        ▼
  Expected P&L = |signal| × IC × σ_r - TC > 0 ✓
```

**Turnover-IC tradeoff:** A higher threshold reduces turnover and TC but sacrifices IC. The optimal $s^*$ maximizes **net Information Ratio**:

$$IR_{net} = \frac{IC_{net}}{\sigma_{IC_{net}}} \approx \frac{IC_{gross} - \frac{TC \cdot \text{TO}}{\sigma_r^{cross}}}{1/\sqrt{BR}}$$

where $BR$ is the number of independent bets.

[🔝 Back to Top](#-table-of-contents)

---

## 🎪 H. Brain Teasers

<a name="q26"></a>
### Q26 — Expected Number of Flips to Get Two Consecutive Heads
> *"A fair coin is flipped repeatedly. What is the expected number of flips until you get two consecutive heads (HH)?"*
> 🗓️ *Reported: Glassdoor / Blind, Q1 2026*

**Answer:**

**Setup states:**

- $S_0$: Start (or just got T)
- $S_1$: Just got one H
- $S_2$: Got HH → stop

Let $E_i$ = expected flips from state $S_i$.

**Equations:**

From $S_0$: flip → H (prob 1/2) go to $S_1$, or T (prob 1/2) stay in $S_0$:
$$E_0 = 1 + \frac{1}{2}E_1 + \frac{1}{2}E_0$$

From $S_1$: flip → H (prob 1/2) go to $S_2$ (done!), or T (prob 1/2) go to $S_0$:
$$E_1 = 1 + \frac{1}{2}(0) + \frac{1}{2}E_0$$

**Solving:**

$$E_1 = 1 + \frac{E_0}{2}$$

$$E_0 = 1 + \frac{1}{2}\left(1 + \frac{E_0}{2}\right) + \frac{E_0}{2} = 1 + \frac{1}{2} + \frac{E_0}{4} + \frac{E_0}{2}$$

$$E_0 - \frac{3E_0}{4} = \frac{3}{2} \implies \frac{E_0}{4} = \frac{3}{2} \implies \boxed{E_0 = 6}$$

**Verification:** $E_1 = 1 + 3 = 4$. Check: $E_0 = 1 + \frac{1}{2}(4) + \frac{1}{2}(6) = 1 + 2 + 3 = 6$ ✓

**Generalization:** Expected flips for $k$ consecutive heads = $2^{k+1} - 2$.
- $k=1$: $E = 2$ ✓
- $k=2$: $E = 6$ ✓
- $k=3$: $E = 14$

[🔝 Back to Top](#-table-of-contents)

---

<a name="q27"></a>
### Q27 — 100 Prisoners and a Light Switch Puzzle
> *"100 prisoners are each sent to a room with a light switch one at a time in random order. They can turn the switch on or off. They win if one prisoner can declare 'all 100 have visited.' Design an optimal strategy."*
> 🗓️ *Reported: Glassdoor / Blind, Q3 2025*

**Answer:**

**Optimal Strategy:**

Designate one prisoner as the **Counter**.

**Counter's rule:** Every time the Counter enters and finds the light **ON**, turn it **OFF** and increment count. When count reaches 99, declare "all have visited."

**All other prisoners' rule:** The first time a non-counter prisoner enters and finds the light **OFF**, turn it **ON**. After that, never touch the switch again.

**Why it works:**

```
State machine:
  Each non-counter has exactly one "token" to send.
  Token = turning the light ON once.
  Counter collects tokens by turning OFF.
  
  99 tokens delivered → 99 non-counters visited
  (Counter itself always visits) → 100 total.
```

**Correctness proof:**
- Each non-counter contributes exactly one ON signal
- Counter correctly counts 99 such signals
- Counter's own visits add to total but need no token
- No false positives: no non-counter turns light ON twice

**Expected time:** Under uniform random ordering, the expected number of days is approximately:

$$\mathbb{E}[T] \approx 100 \times H_{99} \times \text{(room cycle)} \approx 100 \times \ln(100) \approx 460 \text{ room-visits}$$

(Coupon collector-type argument for 99 non-counter "first-OFF" events.)

**Interview tip:** The key insight is **asymmetry**: designate a single agent as the information aggregator. This pattern — centralizing information through a single counter — appears in distributed systems design, which is why quant shops love this puzzle.

[🔝 Back to Top](#-table-of-contents)

---

<a name="q28"></a>
### Q28 — Secretary Problem: Optimal Stopping Rule
> *"You interview N candidates one at a time. After each interview you must accept or reject immediately. Maximize the probability of selecting the best candidate. What is the optimal strategy and its success probability?"*
> 🗓️ *Reported: Glassdoor / QuantNet, Q2 2026*

**Answer:**

**Optimal Strategy (37% rule):** Reject the first $r^* - 1$ candidates outright, then accept the first candidate who is better than all previously seen.

**Optimal cutoff:**

$$r^* = \left\lfloor \frac{N}{e} \right\rfloor \approx \frac{N}{e} \approx 0.368 \cdot N$$

**Success probability:**

$$P(\text{select best}) = \frac{r^*-1}{N} \sum_{k=r^*}^{N} \frac{1}{k-1} \xrightarrow{N\to\infty} \frac{1}{e} \approx 36.8\%$$

**Derivation sketch.** Probability of selecting candidate at position $k$ (who is globally best) given strategy $r$:

$$P(\text{pick best} \mid \text{best at position } k) = \begin{cases} 0 & k < r \\ \frac{r-1}{k-1} & k \geq r \end{cases}$$

Total probability (assuming best equally likely at any position):

$$P(r) = \frac{1}{N}\sum_{k=r}^{N}\frac{r-1}{k-1}$$

Maximize over $r$, take $N\to\infty$, $t = r/N$ continuous:

$$P(t) = -t \ln t \implies t^* = 1/e$$

**Finance application:** Optimal stopping underlies: when to exit a trade (optimal liquidation), how long to wait before committing capital to a signal (discovery vs. exploitation), and hiring decisions for alpha research teams — a direct meta-application.

[🔝 Back to Top](#-table-of-contents)

---

## ∑ I. Mathematical Induction

<a name="q29"></a>
### Q29 — Prove Sum of First N Integers by Induction
> *"Prove by mathematical induction that $\sum_{k=1}^{n} k = \frac{n(n+1)}{2}$."*
> 🗓️ *Reported: HackerRank / Glassdoor OA, Q4 2025*

**Answer:**

**Claim:** $P(n): \displaystyle\sum_{k=1}^{n} k = \frac{n(n+1)}{2}$ for all $n \in \mathbb{Z}^+$.

**Base case ($n=1$):**

$$\sum_{k=1}^{1} k = 1 = \frac{1 \cdot 2}{2} = 1 \checkmark$$

**Inductive hypothesis:** Assume $P(n)$ holds for some $n \geq 1$:

$$\sum_{k=1}^{n} k = \frac{n(n+1)}{2}$$

**Inductive step ($n \to n+1$):** We must show $P(n+1)$: $\displaystyle\sum_{k=1}^{n+1} k = \frac{(n+1)(n+2)}{2}$.

$$\sum_{k=1}^{n+1} k = \underbrace{\sum_{k=1}^{n} k}_{= \frac{n(n+1)}{2} \text{ by IH}} + (n+1) = \frac{n(n+1)}{2} + (n+1)$$

$$= (n+1)\left(\frac{n}{2} + 1\right) = (n+1) \cdot \frac{n+2}{2} = \frac{(n+1)(n+2)}{2}$$

This is exactly $P(n+1)$. $\blacksquare$

**Alternate elegant proof (Gauss):** Write the sum forwards and backwards:

```
S = 1   +  2   + ... + (n-1) + n
S = n   + (n-1)+ ... +  2   + 1
─────────────────────────────────
2S = (n+1) + (n+1) + ... + (n+1)   [n terms]
   = n(n+1)  →  S = n(n+1)/2
```

**Why interviewers ask this:** Induction is the backbone of algorithm correctness proofs (e.g., proving loop invariants, proving dynamic programming recurrences, proving convergence of iterative methods like gradient descent under Lipschitz conditions).

[🔝 Back to Top](#-table-of-contents)

---

<a name="q30"></a>
### Q30 — Prove Geometric Series Formula by Induction
> *"Prove by induction that $\sum_{k=0}^{n} r^k = \frac{1 - r^{n+1}}{1-r}$ for $r \neq 1$. Then derive the infinite series limit."*
> 🗓️ *Reported: HackerRank / eFinancialCareers OA, Q1 2026*

**Answer:**

**Claim:** $P(n): \displaystyle\sum_{k=0}^{n} r^k = \frac{1 - r^{n+1}}{1-r}$ for all $n \geq 0$, $r \neq 1$.

**Base case ($n = 0$):**

$$\sum_{k=0}^{0} r^k = r^0 = 1 = \frac{1 - r^1}{1 - r} = \frac{1-r}{1-r} = 1 \checkmark$$

**Inductive hypothesis:** Assume $P(n)$ holds:

$$\sum_{k=0}^{n} r^k = \frac{1 - r^{n+1}}{1-r}$$

**Inductive step:** Show $P(n+1)$.

$$\sum_{k=0}^{n+1} r^k = \underbrace{\sum_{k=0}^{n} r^k}_{= \frac{1-r^{n+1}}{1-r} \text{ by IH}} + r^{n+1}$$

$$= \frac{1 - r^{n+1}}{1-r} + r^{n+1} = \frac{1 - r^{n+1} + r^{n+1}(1-r)}{1-r}$$

$$= \frac{1 - r^{n+1} + r^{n+1} - r^{n+2}}{1-r} = \frac{1 - r^{n+2}}{1-r}$$

This is $P(n+1)$. $\blacksquare$

**Infinite series:** For $|r| < 1$, $r^{n+1} \to 0$ as $n \to \infty$:

$$\sum_{k=0}^{\infty} r^k = \frac{1}{1-r}$$

**Finance applications:**

Geometric series is ubiquitous in quant finance:

$$\underbrace{\text{Bond price}}_{\text{infinite annuity}} = \sum_{t=1}^\infty \frac{C}{(1+y)^t} = \frac{C}{y}$$

$$\underbrace{\text{Gordon Growth Model}}_{\text{equity valuation}} = \frac{D_1}{r - g} = \frac{D_0(1+g)}{r-g}$$

$$\underbrace{\text{AR(1) impulse response}}_{\text{signal decay}} : \text{persistence } = \sum_{k=0}^\infty \phi^k = \frac{1}{1-\phi}, \quad |\phi| < 1$$

**Extension via induction — matrix geometric series:** For matrices, $\sum_{k=0}^n A^k = (I - A)^{-1}(I - A^{n+1})$ when $\rho(A) < 1$ (spectral radius condition). Proved by exact same induction structure, replacing $r$ with $A$ and scalar division with matrix inversion.

[🔝 Back to Top](#-table-of-contents)

---

## 📊 Quick Reference Summary

```
Graham Capital Round 1 — Topic Distribution (30 Questions)

  Probability & Statistics  ████████░░  5 questions  (16.7%)
  Stochastic Calculus       ████████░░  4 questions  (13.3%)
  Linear Algebra            ██████░░░░  3 questions  (10.0%)
  Machine Learning          ████████░░  4 questions  (13.3%)
  Artificial Intelligence   ██████░░░░  3 questions  (10.0%)
  Coding                    ██████░░░░  3 questions  (10.0%)
  Reasoning                 ██████░░░░  3 questions  (10.0%)
  Brain Teasers             ██████░░░░  3 questions  (10.0%)
  Mathematical Induction    ████░░░░░░  2 questions   (6.7%)
```

## 🔗 Key Resources

- [Graham Capital Research Papers](https://www.grahamcapital.com/resource-center/research-papers/)
- [López de Prado — Advances in Financial Machine Learning](https://www.wiley.com/en-us/Advances+in+Financial+Machine+Learning-p-9781119482086)
- [Marchenko–Pastur Law (RMT)](https://en.wikipedia.org/wiki/Marchenko%E2%80%93Pastur_distribution)
- [Bailey & López de Prado — Deflated Sharpe Ratio (2014)](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2460551)
- [Almgren & Chriss — Optimal Execution (2001)](https://www.sciencedirect.com/science/article/pii/S1386418101000070)
- [Gârleanu & Pedersen — Dynamic Trading with Predictable Returns (2013)](https://www.nber.org/papers/w15205)

---

*Document compiled from: Glassdoor, Blind, Wall Street Oasis, QuantNet, eFinancialCareers, OpenQuant, HackerRank, LeetCode Discuss, QuantVision 2026 conference notes, and arxiv preprints (2025–2026). Weighted toward 2026 intelligence per Graham Capital's Quant Macro and Systematic Research mandates.*

[🔝 Back to Top](#-table-of-contents)
