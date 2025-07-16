package stats

import (
	"fmt"
	"io"
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
func (p *PMF) Print(out io.Writer) {
	fmt.Println("PMF:")
	for _, value := range p.orderedValues {
		fmt.Fprintf(out, "  P(X=%f) = %.4f\n", value, p.Get(value))
	}
	fmt.Fprintf(out, "Total Sum Probabilities: %.4f\n", p.TotalSumProbabilities())
	fmt.Fprintf(out, "Mean: %.4f\n", p.Mean())
	fmt.Fprintf(out, "Std Dev: %.4f\n", p.StdDev())
}
