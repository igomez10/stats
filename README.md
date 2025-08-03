# stats

A Go package for working with probability distributions, statistical functions, and numerical methods.

## Features

- Probability Mass Functions (PMF) for Binomial and Poisson distributions
- Probability Density Functions (PDF) for Normal, Student's t, and Exponential distributions
- Numerical integration and differentiation utilities
- Maximum Likelihood Estimation (MLE) for Normal and Poisson distributions
- Utility functions for combinations, factorials, normalization, and more

## Main Functions

### Probability Distributions

- `NewBinomialPMF(numberOfTrials int, probSuccess float64) *PMF`  
  Create a binomial probability mass function.
- `NewPoissonPMF(lambda float64, numEvents int) *PMF`  
  Create a Poisson probability mass function.
- `NewNormalPDF(mean, stdDev, rangeMin, rangeMax float64) *PDF`  
  Create a normal probability density function.
- `NewStudentTDistributionPDF(degreesOfFreedom, rangeMin, rangeMax float64) *PDF`  
  Create a Student's t-distribution PDF.
- `NewExponentialPDF(lambda, rangeMin, rangeMax float64) *PDF`  
  Create an exponential probability density function.

### Utility Functions

- `Combination(totalItems, takenItems int) int`  
  Compute the number of combinations (n choose k).
- `Factorial(n int) int`  
  Compute the factorial of n.
- `Normalize(x, mean, stdDev float64) float64`  
  Compute the z-score for a value.
- `Integrate(from, to, step float64, fx func(float64) float64) float64`  
  Numerically integrate a function over a range.
- `Derivate(x, step float64, fx func(float64) float64) func(float64) float64`  
  Return a function that computes the numerical derivative.
- `IntegrateUntilValue(from, toMaxValue, targetValue, step float64, fx func(float64) float64) float64`  
  Integrate until a target value is reached.

### Maximum Likelihood Estimation (MLE)

- `GetMaximumLikelihoodEstimation(data []float64) float64`  
  Compute the MLE (mean) for a normal distribution.
- `GetMaximumLikelihoodEstimationPoisson(data []float64) float64`  
  Compute the MLE (mean) for the rate parameter (lambda) of a Poisson distribution.

## Example Usage

```go
// Binomial PMF
pmf := NewBinomialPMF(10, 0.5)

// Normal PDF
pdf := NewNormalPDF(0, 1, -3, 3)

// MLE for normal
mean := GetMaximumLikelihoodEstimation([]float64{1, 2, 3, 4, 5})

// MLE for Poisson
lambda := GetMaximumLikelihoodEstimationPoisson([]float64{2, 3, 4, 2, 3})
```

## License

MIT
