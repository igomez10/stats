package pkg

import (
	"math"
	"testing"
)

func Test_mean(t *testing.T) {
	type args struct {
		a []float64
	}
	tests := []struct {
		name string
		args args
		want float64
	}{
		{"case1", args{[]float64{1, 2, 3}}, 2.0},
		{"case2", args{[]float64{4, 5, 6}}, 5.0},
		{"case3", args{[]float64{7, 8, 9}}, 8.0},
		{"case4", args{[]float64{10, 11, 12}}, 11.0},
		{"case5", args{[]float64{1, 100}}, 50.5},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := GetMean(tt.args.a); got != tt.want {
				t.Errorf("GetMean() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestGetSampleVariance(t *testing.T) {
	tests := []struct {
		name string // description of this test case
		// Named input parameters for target function.
		a    []float64
		want float64
	}{
		{"case1", []float64{1, 2, 3}, 1.0},
		{"case2", []float64{4, 5, 6}, 1.0},
		{"case3", []float64{7, 8, 9}, 1.0},
		{"case4", []float64{10, 11, 12}, 1.0},
		{"case5", []float64{1, 100}, 4900.5},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := GetSampleVariance(tt.a)
			if math.Abs(got-tt.want) > 1e-9 {
				t.Errorf("GetVariance() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestNormalizeObservations(t *testing.T) {
	tests := []struct {
		name         string
		observations [][]float64
		want         [][]float64
	}{
		{
			name: "simple case",
			observations: [][]float64{
				{1},
				{2},
				{3},
			},
			want: [][]float64{
				{-1},
				{0},
				{1},
			},
		},
		{
			name: "multiple observations",
			observations: [][]float64{
				{1, 2, 3},
				{1, 2, 3},
			},
			want: [][]float64{
				{0, 0, 0},
				{0, 0, 0},
			},
		},
		{
			name: "multiple observations different values",
			observations: [][]float64{
				{1, 1, 1},
				{2, 2, 2},
			},
			want: [][]float64{
				{-0.7071067811, -0.7071067811, -0.7071067811},
				{0.7071067811, 0.7071067811, 0.7071067811},
			},
		},
		{
			name: "complex case",
			observations: [][]float64{
				{1, 2},
				{2, 1},
				{3, 0},
			},
			want: [][]float64{
				{-1, 1},
				{0, 0},
				{1, -1},
			},
		},
		{
			name:         "empty observations",
			observations: [][]float64{},
			want:         [][]float64{},
		},
		{
			name: "3 observations with 3 features",
			observations: [][]float64{
				{1, 2, 3},
				{4, 5, 6},
				{7, 8, 9},
			},
			want: [][]float64{
				{-1, -1, -1},
				{0, 0, 0},
				{1, 1, 1},
			},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			before := [][]float64{}
			for i := 0; i < len(tt.observations); i++ {
				before = append(before, make([]float64, len(tt.observations[i])))
			}
			CopyMatrix(before, tt.observations)
			got := NormalizeObservations(tt.observations)
			for i := range got {
				for j := range got[i] {
					if math.Abs(got[i][j]-tt.want[i][j]) > 1e-9 {
						t.Errorf("NormalizeObservations() = %v, want %v", got, tt.want)
						return
					}
				}
			}
			for i := range tt.observations {
				for j := range tt.observations[i] {
					if tt.observations[i][j] != before[i][j] {
						t.Errorf("NormalizeObservations() modified the input observations")
						return
					}
				}
			}
		})
	}
}

func Test_normalize(t *testing.T) {
	tests := []struct {
		name string // description of this test case
		// Named input parameters for target function.
		arr  []float64
		want []float64
	}{
		{"case1", []float64{1, 2, 3}, []float64{-1.2247448714, 0, 1.2247448714}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := normalize(tt.arr)
			for i := range got {
				if math.Abs(got[i]-tt.want[i]) > 1e-9 {
					t.Errorf("normalize() = %v, want %v", got, tt.want)
				}
			}
		})
	}
}

func TestGetPopulationVariance(t *testing.T) {
	tests := []struct {
		name string // description of this test case
		// Named input parameters for target function.
		a    []float64
		want float64
	}{
		{"case1", []float64{1, 2, 3}, 0.6666666667},
		{"case2", []float64{4, 5, 6}, 0.6666666667},
		{"case3", []float64{7, 8, 9}, 0.6666666667},
		{"case4", []float64{10, 11, 12}, 0.6666666667},
		{"case5", []float64{1, 100}, 2450.25},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := GetPopulationVariance(tt.a)
			if math.Abs(got-tt.want) > 1e-9 {
				t.Errorf("GetPopulationVariance() = %v, want %v", got, tt.want)
			}
		})
	}
}
