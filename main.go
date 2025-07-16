package stats

import (
	"math"
)

// NewCDFFromPMF creates a CDF from a PDF
func NewCDFFromPDF(pdf *PDF) *CDF {
	cdf := NewCDF()

	for x := pdf.rangeMin; x <= pdf.rangeMax; x += 0.01 {
		cdf.Set(x, pdf.function(x))
	}

	return cdf
}

// CreatePoissonPMF creates a Poisson PMF
// maxK is the maximum value of k for which the PMF is defined
// k is the number of events in a fixed interval
func CreatePoissonPMF(lambda float64, maxNumEvents int) *PMF {
	pmf := NewPMF()

	for currentNumEvents := 0; currentNumEvents <= maxNumEvents; currentNumEvents++ {
		// Calculate Poisson probability: e^(-λ) * λ^k / k!
		prob := math.Exp(-lambda) * math.Pow(lambda, float64(currentNumEvents)) / float64(factorial(currentNumEvents))
		pmf.Set(float64(currentNumEvents), prob)
	}

	return pmf
}

// CreateBinomialPMF creates a binomial PMF
// A binomial PMF is defined by the number of trials (n) and the probability of success (p)
// It calculates the probability of getting k successes in n trials
// using the formula: P(X=k) = C(n, k) * p^k * (1-p)^(n-k)
// where C(n, k) is the binomial coefficient "n choose k"
// The PMF is defined for k = 0, 1, ..., n
// The total number of outcomes is n+1 (from 0 to n)
// The PMF is normalized so that the sum of probabilities equals 1
func CreateBinomialPMF(numberOfTrials int, probSuccess float64) *PMF {
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
