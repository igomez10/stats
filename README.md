# stats

A Go library for statistical computations, probability distributions, and related mathematical functions.

## Features

- Probability Mass Functions (PMF) and Probability Density Functions (PDF)
- Binomial, Poisson, Normal, Exponential distributions
- Confidence intervals for means and proportions
- Z-score and T-score calculations
- Numerical integration and differentiation
- Marginal distributions for joint PDFs
- Utility functions for statistics (sum, average, stdev, etc.)

## Exposed Methods

### Distributions

- **NewPMF**: Create and manipulate a probability mass function.
- **NewBinomialPMF(n, p)**: Binomial distribution PMF for `n` trials and success probability `p`.
- **NewPoissonPMF(lambda, maxNumEvents)**: Poisson distribution PMF for rate `lambda`.
- **NewExponentialPDF(lambda, rangeMin, rangeMax)**: Exponential distribution PDF.
- **NewNormalPDF(mean, stdDev, rangeMin, rangeMax)**: Normal (Gaussian) distribution PDF.

### Probability and Statistics

- **GetMeanConfidenceIntervalForNormalDistribution(sampleMean, sampleStdDev, sampleSize, confidenceLevel, from, to, step)**  
  Calculate confidence interval for the mean.
- **GetProportionConfidenceInterval(successProbability, sampleSize, confidenceLevel, from, to, step)**  
  Confidence interval for a proportion.
- **GetStudentTStatistic(sampleMean, populationMean, sampleStandardDeviation, sampleSize)**  
  Student's T statistic.
- **GetZScore(sampleMean, populationMean, sampleStandardDeviation, sampleSize)**  
  Z-score for hypothesis testing.
- **GetTScore(degreesOfFreedom, confidenceLevel, from, to, step)**  
  T-score for given degrees of freedom and confidence level.
- **GetExponentialDistributionFunction(lambda)**  
  Exponential CDF.

### Numerical Methods

- **Integrate(from, to, step, func)**  
  Numerical integration using the supplied function.
- **Derivate(fx, step)**  
  Numerical derivative of a function.
- **FindCriticalPoint(fx, from, to, step)**  
  Find points where the derivative is zero (local minima/maxima).

### Joint and Marginal Distributions

- **JointPDF.GetMarginalX(step)**  
  Get the marginal distribution in X by integrating over Y.
- **JointPDF.GetMarginalY(step)**  
  Get the marginal distribution in Y by integrating over X.

### Utility Functions

- **sum([]float64)**: Sum of array elements.
- **average([]float64)**: Mean of array elements.
- **stdev([]float64)**: Standard deviation.
- **Normalize(x, mean, stdDev)**: Calculates z-score.

## Usage

Import the package and use the available methods to perform statistical calculations or probability simulations.

```go
pmf := NewBinomialPMF(10, 0.5)
pmf.Print(os.Stdout)

// Normal distribution example
pdf := NewNormalPDF(0, 1, -5, 5)
value := pdf.ValueAt(0)
```

## Testing

The repository includes a comprehensive test suite (`main_test.go`) that demonstrates the usage and expected output of all major methods.

---

Feel free to open issues or contribute with improvements!
