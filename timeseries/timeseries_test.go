package timeseries

import (
	"math"
	"testing"
)

// estimate trend
func TestGetTrend(t *testing.T) {
	type testcase struct {
		name     string
		data     []float64
		window   int
		expected []float64
	}

	testcases := []testcase{
		{
			name:     "simple moving average",
			data:     []float64{1, 2, 3, 4, 5, 6, 7, 8, 9, 10},
			window:   3,
			expected: []float64{math.NaN(), 2, 3, 4, 5, 6, 7, 8, 9, math.NaN()},
		},
		{
			name:     "simple moving average window size 5",
			data:     []float64{1, 2, 3, 4, 5, 6, 7, 8, 9, 10},
			window:   5,
			expected: []float64{math.NaN(), math.NaN(), 3, 4, 5, 6, 7, 8, math.NaN(), math.NaN()},
		},
		{
			name:     "even window size 4",
			data:     []float64{1, 2, 3, 4, 5, 6, 7, 8, 9, 10},
			window:   4,
			expected: []float64{math.NaN(), math.NaN(), 3, 4, 5, 6, 7, 8, math.NaN(), math.NaN()},
		},
	}

	for _, tc := range testcases {
		t.Run(tc.name, func(t *testing.T) {
			result := GetTrend(tc.data, tc.window)
			for i := range result {
				if diff := math.Abs(result[i] - tc.expected[i]); diff > 0.01 {
					t.Errorf("index %d", i)
					t.Errorf("value %f", tc.expected[i])
					t.Errorf("diff, %f", diff)
					t.Errorf("value %f", result[i])
					t.Error("expected:", tc.expected)
					t.Error("got:", result)
				}
			}
		})
	}
}

// estimate seasonality
// estimate residuals
func TestGetSeasonality(t *testing.T) {
	type testcase struct {
		name     string
		data     []float64
		window   int
		expected []float64
	}

	testcases := []testcase{
		{
			name:   "periodic data with linear trend window 4",
			data:   []float64{2, 3, 5, 2, 3, 4, 6, 3, 4, 5, 7, 4},
			window: 4,
			expected: []float64{
				-0.625, 0.125, 1.875, -1.375,
				-0.625, 0.125, 1.875, -1.375,
				-0.625, 0.125, 1.875, -1.375,
			},
		},
		{
			name:     "constant data has zero seasonality",
			data:     []float64{5, 5, 5, 5, 5, 5, 5, 5, 5},
			window:   3,
			expected: []float64{0, 0, 0, 0, 0, 0, 0, 0, 0},
		},
		{
			name:     "linear data has zero seasonality",
			data:     []float64{1, 2, 3, 4, 5, 6, 7, 8, 9, 10},
			window:   3,
			expected: []float64{0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
		},
	}

	for _, tc := range testcases {
		t.Run(tc.name, func(t *testing.T) {
			result := GetSeasonality(tc.data, tc.window)
			for i := range result {
				if diff := math.Abs(result[i] - tc.expected[i]); diff > 0.01 {
					t.Errorf("index %d", i)
					t.Errorf("value %f", tc.expected[i])
					t.Errorf("diff, %f", diff)
					t.Errorf("value %f", result[i])
					t.Error("expected:", tc.expected)
					t.Error("got:", result)
				}
			}
		})
	}
}
