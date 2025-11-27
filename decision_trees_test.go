package stats

import (
	"math"
	"testing"
)

func TestGetGiniImpurity(t *testing.T) {
	tests := []struct {
		name   string
		labels []string
		want   float64
	}{
		{
			name:   "Single class",
			labels: []string{"A", "A", "A"},
			want:   0.0,
		},
		{
			name:   "Two classes",
			labels: []string{"A", "A", "B", "B"},
			want:   0.5,
		},
		{
			name:   "Three classes",
			labels: []string{"A", "A", "B", "B", "C"},
			want:   0.639999999999,
		},
		{
			name:   "Empty labels",
			labels: []string{},
			want:   0.0,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := GetGiniImpurity(tt.labels)
			if math.Abs(got-tt.want) > 1e-9 {
				t.Errorf("GetGiniImpurity() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestGetBestSplit(t *testing.T) {
	tests := []struct {
		name string // description of this test case
		// Named input parameters for target function.
		labels         []string
		minSamplesLeaf int
		want           int
		wantErr        error
	}{
		{
			name:           "no labels",
			labels:         []string{},
			minSamplesLeaf: 1,
			want:           0,
			wantErr:        errEmptySplits,
		},
		{
			name:           "invalid min nodes",
			labels:         []string{"a", "b"},
			minSamplesLeaf: 3,
			want:           0,
			wantErr:        errInvalidSamplesPerLeaf,
		},
		{
			name:           "invalid min nodes",
			labels:         []string{"a", "b"},
			minSamplesLeaf: 3,
			want:           0,
			wantErr:        errInvalidSamplesPerLeaf,
		},
		{
			name:           "ex1 ", //-----|
			labels:         []string{"a", "b"},
			minSamplesLeaf: 1,
			want:           1,
			wantErr:        nil,
		},
		{
			name:           "ex2 ", //-----|
			labels:         []string{"a", "b", "b"},
			minSamplesLeaf: 1,
			want:           1,
			wantErr:        nil,
		},
		{
			name:           "ex3 ", //----------|
			labels:         []string{"a", "a", "b"},
			minSamplesLeaf: 1,
			want:           2,
			wantErr:        nil,
		},
		{
			name:           "ex4 ", //----------|
			labels:         []string{"a", "a", "b", "b"},
			minSamplesLeaf: 2,
			want:           2,
			wantErr:        nil,
		},
		{
			name:           "ex5 ", //----------|
			labels:         []string{"a", "a", "b", "a", "b"},
			minSamplesLeaf: 2,
			want:           2,
			wantErr:        nil,
		},
		{
			name:           "ex7 ", //---------------|
			labels:         []string{"a", "b", "a", "b", "b", "b"},
			minSamplesLeaf: 2,
			want:           3,
			wantErr:        nil,
		},
		{
			name:           "ex8 ", //---------------|
			labels:         []string{"a", "b", "a", "b", "b", "b"},
			minSamplesLeaf: 3,
			want:           3,
			wantErr:        nil,
		},
		{
			name:           "ex9 min samples > 2*len input ",
			labels:         []string{"a", "b", "a", "b", "b", "b"},
			minSamplesLeaf: 4,
			want:           0,
			wantErr:        errInvalidSamplesPerLeaf,
		},
		{
			name:           "ex10 ", //---------------|
			labels:         []string{"a", "a", "a", "b", "b", "b", "b", "b"},
			minSamplesLeaf: 2,
			want:           3,
			wantErr:        nil,
		},
		{
			name: "ex11 left not totally pure",
			//---------------------------------------|
			labels:         []string{"a", "b", "a", "b", "b", "b", "b", "b"},
			minSamplesLeaf: 2,
			want:           3,
			wantErr:        nil,
		},
		{
			name: "ex11 right not totally pure",
			//---------------------------------------|
			labels:         []string{"a", "b", "a", "b", "b", "b", "b", "a"},
			minSamplesLeaf: 2,
			want:           3,
			wantErr:        nil,
		},
		{
			name: "ex12 left is more pure",
			//---------------------------------------|
			labels:         []string{"a", "a", "a", "b", "b", "b", "b", "a"},
			minSamplesLeaf: 2,
			want:           3,
			wantErr:        nil,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := GetBestSplit(tt.labels, tt.minSamplesLeaf)
			if tt.wantErr != err {
				t.Error("unexpected err")
			}

			if got != tt.want {
				t.Errorf("unexpected split for %+v got %d wanted %d", tt.labels, got, tt.want)
			}
		})
	}
}
