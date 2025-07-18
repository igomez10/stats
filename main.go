package stats

import (
	"math"
)

// Combination calculates the number of combinations of n items taken r at a time
func Combination(totalItems, takenItems int) int {
	if takenItems > totalItems || takenItems < 0 {
		return 0
	}
	if takenItems == 0 || takenItems == totalItems {
		return 1
	}

	// Use the property C(n, r) = C(n, n-r) to minimize calculations
	if takenItems > totalItems-takenItems {
		takenItems = totalItems - takenItems
	}

	numerator := 1
	denominator := 1
	for i := 0; i < takenItems; i++ {
		numerator *= (totalItems - i)
		denominator *= (i + 1)
	}
	return numerator / denominator
}

// Factorial calculates Factorial of n
func Factorial(n int) int {
	if n <= 1 {
		return 1
	}
	result := 1
	for i := 2; i <= n; i++ {
		result *= i
	}
	return result
}

// NewCDFFromPMF creates a CDF from a PDF
func NewCDFFromPDF(pdf *PDF) *CDF {
	cdf := NewCDF()

	for x := pdf.rangeMin; x <= pdf.rangeMax; x += 0.01 {
		cdf.Set(x, pdf.function(x))
	}

	return cdf
}

// NewPoissonPMF creates a Poisson PMF
// Poisson is always PMF
// lambda is the average rate of occurrence
// numEvents is the maximum number of events to consider
// The PMF is defined for k = 0, 1, ..., numEvents
// The total number of outcomes is numEvents + 1 (from 0 to numEvents)
// The PMF is normalized so that the sum of probabilities equals 1
func NewPoissonPMF(lambda float64, numEvents int) *PMF {
	pmf := NewPMF()
	for currentNumEvents := 0; currentNumEvents <= numEvents; currentNumEvents++ {
		// Calculate Poisson probability: e^(-λ) * λ^k / k!
		prob := math.Exp(-lambda) * math.Pow(lambda, float64(currentNumEvents)) / float64(Factorial(currentNumEvents))
		pmf.Set(float64(currentNumEvents), prob)
	}

	return pmf
}

// NewBinomialPMF creates a binomial PMF
// A binomial PMF is defined by the number of trials (n) and the probability of success (p)
// It calculates the probability of getting k successes in n trials
// using the formula: P(X=k) = C(n, k) * p^k * (1-p)^(n-k)
// where C(n, k) is the binomial coefficient "n choose k"
// The PMF is defined for k = 0, 1, ..., n
// The total number of outcomes is n+1 (from 0 to n)
// The PMF is normalized so that the sum of probabilities equals 1
func NewBinomialPMF(numberOfTrials int, probSuccess float64) *PMF {
	pmf := NewPMF()
	for i := 0; i <= numberOfTrials; i++ {
		// Calculate binomial coefficient C(n, k)
		coeff := Combination(numberOfTrials, i)
		// Calculate probability: C(n,k) * p^k * (1-p)^(n-k)
		prob := float64(coeff) * math.Pow(probSuccess, float64(i)) * math.Pow(1-probSuccess, float64(numberOfTrials-i))
		pmf.Set(float64(i), prob)
	}

	return pmf
}

func GetNormalDistributionFunction(mean, stdDev float64) func(float64) float64 {
	return func(x float64) float64 {
		// fx = (1 / (stdDev * math.Sqrt(2*math.Pi))) * exp(-0.5 * ((x - mean) / stdDev) ^ 2)
		return (1 / (stdDev * math.Sqrt(2*math.Pi))) * math.Exp(-0.5*math.Pow((x-mean)/stdDev, 2))
	}
}

func NewNormalPDF(mean, stdDev, rangeMin, rangeMax float64) *PDF {
	if stdDev <= 0 {
		panic("Standard deviation must be positive")
	}
	if rangeMin >= rangeMax {
		panic("Invalid range: rangeMin must be less than rangeMax")
	}

	normalPDF := NewPDF(
		GetNormalDistributionFunction(mean, stdDev),
		rangeMin,
		rangeMax,
	)

	return normalPDF
}

// Normalize normalizes a value x based on the mean and standard deviation
// It returns the z-score, which is the number of standard deviations away from the mean
// z = (x - mean) / stdDev
// This is useful for standardizing values in a normal distribution
// It transforms the value into a standard normal variable (mean = 0, stdDev = 1)
func Normalize(x float64, mean, stdDev float64) float64 {
	return (x - mean) / stdDev
}
