package stats

import "math/rand/v2"

type Generator struct {
	Random *rand.Rand
}

// GenerateNormalSamples generates n samples from a normal distribution
// with the specified mean and standard deviation.
func (g *Generator) GenerateNormalSamples(mean, stddev float64, n int) []float64 {
	samples := make([]float64, n)
	for i := range samples {
		samples[i] = g.Random.NormFloat64()*stddev + mean
	}
	return samples
}

func (g *Generator) GetRandomSample(data []float64, sampleSize int) []float64 {
	if sampleSize > len(data) {
		panic("sample size cannot be larger than data size")
	}
	sample := make([]float64, sampleSize)
	dataLen := len(data)
	for i := range sampleSize {
		index := g.Random.IntN(dataLen)
		sample[i] = data[index]
	}
	return sample
}
