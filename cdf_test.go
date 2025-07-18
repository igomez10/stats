package stats

import (
	"testing"
)

func TestNewCDFFromPMF(t *testing.T) {
	type args struct {
		pmf *PMF
	}
	tests := []struct {
		name string
		args args
		want *CDF
	}{
		{
			name: "Test PMF to CDF conversion",
			args: args{
				pmf: &PMF{
					values: map[float64]float64{
						1: 0.2,
						2: 0.3,
						3: 0.5,
					},
					orderedValues: []float64{1, 2, 3},
				},
			},
			want: &CDF{
				values: map[float64]float64{
					1: 0.2,
					2: 0.5,
					3: 1.0,
				},
			},
		},
		{
			name: "Test empty PMF",
			args: args{
				pmf: &PMF{
					values:        map[float64]float64{},
					orderedValues: []float64{},
				},
			},
			want: &CDF{
				values: map[float64]float64{},
			},
		},
		{
			name: "Test PMF with single value",
			args: args{
				pmf: &PMF{
					values: map[float64]float64{
						1: 1.0,
					},
					orderedValues: []float64{1},
				},
			},
			want: &CDF{
				values: map[float64]float64{
					1: 1.0,
				},
			},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := NewCDFFromPMF(tt.args.pmf)
			if len(got.values) != len(tt.want.values) {
				t.Errorf("NewCDFFromPMF() = %v, want %v", got.values, tt.want.values)
			}
			for k, v := range tt.want.values {
				if got.Get(k) != v {
					t.Errorf("NewCDFFromPMF() = %v, want %v for key %v", got.Get(k), v, k)
				}
			}
		})
	}
}
