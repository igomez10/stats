package stats

import "math/rand/v2"

// GenerateNormalSamples generates n samples from a normal distribution
// with the specified mean and standard deviation.
func GenerateNormalSamples(mean, stddev float64, n int) []float64 {
	samples := make([]float64, n)
	for i := range samples {
		samples[i] = rand.NormFloat64()*stddev + mean
	}
	return samples
}
