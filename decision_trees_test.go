package stats_test

import (
	"math"
	"testing"

	"github.com/igomez10/stats"
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
			got := stats.GetGiniImpurity(tt.labels)
			if math.Abs(got-tt.want) > 1e-9 {
				t.Errorf("GetGiniImpurity() = %v, want %v", got, tt.want)
			}
		})
	}
}
