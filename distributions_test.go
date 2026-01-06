package stats_test

import (
	"math"
	"math/rand/v2"
	"testing"

	"github.com/igomez10/stats"
	"github.com/igomez10/stats/pkg"
)

func TestGenerateNormalSamples(t *testing.T) {
	tests := []struct {
		name string // description of this test case
		// Named input parameters for target function.
		mean   float64
		stddev float64
		n      int
	}{
		{
			name:   "Generate 5 samples with mean 0 and stddev 1",
			mean:   0,
			stddev: 1,
			n:      10000000,
		},
		{
			name:   "Generate 3 samples with mean 10 and stddev 2",
			mean:   10,
			stddev: 2,
			n:      30000000,
		},
		{
			name:   "Generate 100 samples",
			mean:   5,
			stddev: 0.5,
			n:      10000000,
		},
		{
			name:   "Generate 1000 samples",
			mean:   -3,
			stddev: 1.5,
			n:      10000000,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			src := rand.NewPCG(0, 0)
			generator := stats.Generator{
				Random: rand.New(src),
			}
			stats := &generator
			got := stats.GenerateNormalSamples(tt.mean, tt.stddev, tt.n)
			if len(got) != tt.n {
				t.Errorf("GenerateNormalSamples() length = %v, want %v", len(got), tt.n)
			}
			mean := pkg.GetMean(got)
			variance := pkg.GetPopulationVariance(got)
			stddev := math.Sqrt(variance)
			if math.Abs(mean-tt.mean) > 0.01 {
				t.Errorf("GenerateNormalSamples() mean = %v, want approx %v", mean, tt.mean)
			}
			if math.Abs(stddev-tt.stddev) > 0.01 {
				t.Errorf("GenerateNormalSamples() stddev = %v, want approx %v", stddev, tt.stddev)
			}
		})
	}
}

func TestGetRandomSample(t *testing.T) {
	tests := []struct {
		name string // description of this test case
		// Named input parameters for target function.
		data       []float64
		sampleSize int
		want       []float64
	}{
		{
			name:       "Sample 3 from 5 elements",
			data:       []float64{1, 2, 3, 4, 5},
			sampleSize: 3,
			want:       nil, // TODO: fill in expected output
		},
		{
			name:       "Sample 0 from 5 elements",
			data:       []float64{1, 2, 3, 4, 5},
			sampleSize: 0,
			want:       []float64{},
		},
		{
			name:       "Sample 5 from 5 elements",
			data:       []float64{1, 2, 3, 4, 5},
			sampleSize: 5,
			want:       nil, // TODO: fill in expected output
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			src := rand.NewPCG(0, 0)
			generator := stats.Generator{
				Random: rand.New(src),
			}
			stats := &generator
			got := stats.GetRandomSample(tt.data, tt.sampleSize)
			if len(got) != tt.sampleSize {
				t.Errorf("GetRandomSample() length = %v, want %v", len(got), tt.sampleSize)
			}
			// Note: Further checks can be added here to validate the contents of 'got' if needed.
		})
	}
}
