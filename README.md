# Bayes Error Estimation Under Label Noise

This project reproduces and extends the method from the ICLR 2023 paper **“Is the Performance of My Deep Network Too Good to Be True?”** to estimate the **Bayes error** in binary classification using **soft labels**, and analyzes how **label noise** affects the estimator.

We run experiments on:
- **Synthetic Gaussian data** (with analytically computable posteriors)
- **CIFAR-10 binary task** (Animals vs Vehicles) using a CNN to generate soft-label probabilities

## Key Idea
Given soft labels \(c_i = P(y=+1 \mid x_i)\), the clean Bayes error estimate is:
\[
\beta = \text{mean}(\min(c_i, 1 - c_i)).
\]

When noise is injected into soft labels, the **naïve** estimator becomes biased due to the nonlinearity of the min() operation. The **corrected** estimator is designed to remain stable under zero-mean noise.

## Hypothesis
The **naïve Bayes error estimator** becomes increasingly biased as label noise increases, while the **corrected estimator** remains stable and unbiased across noise levels.

## Experiments
- Inject **truncated Gaussian noise** with:
  \[
  \sigma \in \{0.0, 0.1, 0.2, 0.3, 0.4, 0.5\}
  \]
- Run **10 trials per noise level**
- Compute and compare:
  - Clean baseline
  - Naïve noisy estimate
  - Corrected noisy estimate
  - Bias vs baseline

## Results (Summary)
- The **naïve estimator** diverges as noise increases.
- The **corrected estimator** remains close to the clean baseline across all noise levels.
- This trend holds for both synthetic data and CIFAR-10 soft labels produced by a CNN.
