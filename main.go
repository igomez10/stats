package stats

import (
	"fmt"
	"math"
	"sort"
)

// PMF represents a Probability Mass Function
type PMF struct {
	values        map[float64]float64
	orderedValues []float64 // To maintain order of insertion
}

// NewPMF creates a new PMF
func NewPMF() *PMF {
	return &PMF{
		values:        make(map[float64]float64),
		orderedValues: []float64{},
	}
}

// GetSpace (also called range) returns the set of values with non-zero probability
func (p *PMF) GetSpace() []float64 {
	var space []float64
	for value := range p.values {
		if p.values[value] > 0 {
			space = append(space, float64(value))
		}
	}
	return space
}

// Set sets the probability for a given value
func (p *PMF) Set(value float64, prob float64) {
	if prob < 0 || prob > 1 {
		panic("Probability must be between 0 and 1")
	}

	// validatate that the sum of all probabilities does not exceed 1
	if prob+p.TotalSumProbabilities()-1 > 0.01 {
		panic("Total probability cannot exceed 1")
	}

	p.values[value] = prob
	p.orderedValues = append(p.orderedValues, value)
	// sort the ordered values based on the keys
	sort.Float64s(p.orderedValues)
}

// Get returns the probability for a given value
func (p *PMF) Get(value float64) float64 {
	return p.values[value]
}

// TotalSumProbabilities returns the sum of all probabilities
// or to get the total probability mass
// It should be 1 for a valid PMF
func (p *PMF) TotalSumProbabilities() float64 {
	var total float64
	for _, prob := range p.values {
		total += prob
	}
	return total
}

// Mean calculates the expected value (mean) of the PMF
func (p *PMF) Mean() float64 {
	var mean float64
	for value, prob := range p.values {
		mean += float64(value) * prob
	}
	return mean
}

// Variance calculates the variance of the PMF
func (p *PMF) Variance() float64 {
	mean := p.Mean()
	var variance float64
	for value, prob := range p.values {
		diff := float64(value) - mean
		variance += diff * diff * prob
	}
	return variance
}

// StdDev calculates the standard deviation
func (p *PMF) StdDev() float64 {
	return math.Sqrt(p.Variance())
}

// Values returns all values with non-zero probability
func (p *PMF) Values() []float64 {
	var values []float64
	for value := range p.values {
		values = append(values, value)
	}
	return values
}

// Print displays the PMF in a readable format
func (p *PMF) Print() {
	fmt.Println("PMF:")
	for value, prob := range p.values {
		fmt.Printf("  P(X=%f) = %.4f\n", value, prob)
	}
	fmt.Printf("Total: %.4f\n", p.TotalSumProbabilities())
	fmt.Printf("Mean: %.4f\n", p.Mean())
	fmt.Printf("Std Dev: %.4f\n", p.StdDev())
}

// factorial calculates factorial of n
func factorial(n int) int {
	if n <= 1 {
		return 1
	}
	result := 1
	for i := 2; i <= n; i++ {
		result *= i
	}
	return result
}

type PDF struct {
	function func(float64) float64
	rangeMin float64
	rangeMax float64
}

func ValidatePDF(function func(float64) float64, min, max float64) {
	// Check if the function is non-negative in the range
	for x := min; x <= max; x += 0.01 {
		if function(x) < 0 {
			panic("Function must be non-negative in the specified range but got a negative value: " + fmt.Sprintf("f(%f) = %f", x, function(x)))
		}
	}
	// Check if the function integrates to 1 over the range
	total := 0.0
	for x := min; x <= max; x += 0.01 {
		piece := function(x) * 0.01 // Approximate integral using Riemann sum
		total += piece
	}
	if math.Abs(total-1.0) > 0.01 {
		panic("Function must integrate to 1 over the specified range: " + fmt.Sprintf("integral from %f to %f = %f", min, max, total))
	}
}

func NewPDF(function func(float64) float64, min, max float64) *PDF {
	if min >= max {
		panic("Invalid range: min must be less than max")
	}
	if function == nil {
		panic("Function cannot be nil")
	}

	ValidatePDF(function, min, max)

	return &PDF{
		function: function,
		rangeMin: min,
		rangeMax: max,
	}
}

type CDF struct {
	values        map[float64]float64
	orderedValues []float64 // To maintain order of insertion
}

// NewCDF creates a new CDF
func NewCDF() *CDF {
	return &CDF{
		values: make(map[float64]float64),
	}
}

// Set sets the cumulative probability for a given value
func (c *CDF) Set(value float64, prob float64) {
	if prob < 0 || prob > 1 {
		panic("Probability must be between 0 and 1")
	}
	c.values[value] = prob
	c.orderedValues = append(c.orderedValues, value)
	// sort the ordered values based on the keys
	sort.Float64s(c.orderedValues)

	// Ensure cumulative probabilities are non-decreasing
	if len(c.orderedValues) > 1 {
		for i := 1; i < len(c.orderedValues); i++ {
			if c.values[c.orderedValues[i]] < c.values[c.orderedValues[i-1]] {
				panic("Cumulative probabilities must be non-decreasing")
			}
		}
	}
}

// Get returns the cumulative probability for a given value
func (c *CDF) Get(value float64) float64 {
	return c.values[value]
}

// GetSpace returns the set of values with non-zero cumulative probability
func (c *CDF) GetSpace() []float64 {
	var space []float64
	for value := range c.values {
		if c.values[value] > 0 {
			space = append(space, value)
		}
	}
	return space
}

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
		coeff := binomialCoeff(numberOfTrials, i)
		// Calculate probability: C(n,k) * p^k * (1-p)^(n-k)
		prob := float64(coeff) * math.Pow(probSuccess, float64(i)) * math.Pow(1-probSuccess, float64(numberOfTrials-i))
		pmf.Set(float64(i), prob)
	}

	return pmf
}

// binomialCoeff calculates binomial coefficient C(n, k)
// the binomial coefficient is the number of ways to choose k successes in n trials
func binomialCoeff(numTrials int, numSuccesses int) int {
	if numSuccesses > numTrials || numSuccesses < 0 {
		return 0
	}
	if numSuccesses == 0 || numSuccesses == numTrials {
		return 1
	}

	// Use the property C(n,k) = C(n,n-k) to minimize calculations
	if numSuccesses > numTrials-numSuccesses {
		numSuccesses = numTrials - numSuccesses
	}

	result := 1
	for i := 0; i < numSuccesses; i++ {
		result = result * (numTrials - i) / (i + 1)
	}
	return result
}
