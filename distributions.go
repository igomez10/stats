package stats

import "math/rand/v2"

func GenerateNormalSamples(mean, stddev float64, n int) []float64 {
	samples := make([]float64, n)
	for i := range n {
		samples[i] = rand.NormFloat64()*stddev + mean
	}
	return samples
}
