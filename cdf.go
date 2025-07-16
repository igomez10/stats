package stats

import "sort"

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
