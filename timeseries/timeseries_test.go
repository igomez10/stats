package timeseries

import (
	"math"
	"testing"
)

// estimate trend
func TestGetTrend(t *testing.T) {
	type testcase struct {
		name      string
		data      []float64
		window    int
		useCenter bool
		expected  []float64
	}

	testcases := []testcase{
		{
			name:      "simple moving average",
			data:      []float64{1, 2, 3, 4, 5, 6, 7, 8, 9, 10},
			window:    3,
			useCenter: true,
			expected:  []float64{math.NaN(), 2, 3, 4, 5, 6, 7, 8, 9, math.NaN()},
		},
		{
			name:      "simple moving average window size 5",
			data:      []float64{1, 2, 3, 4, 5, 6, 7, 8, 9, 10},
			window:    5,
			useCenter: true,
			expected:  []float64{math.NaN(), math.NaN(), 3, 4, 5, 6, 7, 8, math.NaN(), math.NaN()},
		},
		{
			name:      "even window size 4",
			data:      []float64{1, 2, 3, 4, 5, 6, 7, 8, 9, 10},
			window:    4,
			useCenter: true,
			expected:  []float64{math.NaN(), math.NaN(), 3, 4, 5, 6, 7, 8, math.NaN(), math.NaN()},
		},
		{
			name:      "simple moving average without centering",
			data:      []float64{1, 2, 3, 4, 5, 6, 7, 8, 9, 10},
			window:    3,
			useCenter: false,
			expected:  []float64{math.NaN(), math.NaN(), 2, 3, 4, 5, 6, 7, 8, 9},
		},
		{
			name:      "simple moving average without centering window size 5",
			data:      []float64{1, 2, 3, 4, 5, 6, 7, 8, 9, 10},
			window:    5,
			useCenter: false,
			expected:  []float64{math.NaN(), math.NaN(), math.NaN(), math.NaN(), 3, 4, 5, 6, 7, 8},
		},
	}

	for _, tc := range testcases {
		t.Run(tc.name, func(t *testing.T) {
			result := GetTrend(tc.data, tc.window, tc.useCenter)
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
			trend := GetTrend(tc.data, tc.window, true)
			result := GetSeasonality(tc.data, trend, tc.window)
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

func Test_detrend(t *testing.T) {
	tests := []struct {
		name string // description of this test case
		// Named input parameters for target function.
		data  []float64
		trend []float64
		want  []float64
	}{
		{
			name:  "detrend simple linear data",
			data:  []float64{1, 2, 3, 4, 5},
			trend: []float64{1, 2, 3, 4, 5},
			want:  []float64{0, 0, 0, 0, 0},
		},
		{
			name:  "detrend data with linear trend and seasonality",
			data:  []float64{2, 3, 5, 2, 3},
			trend: []float64{1, 2, 3, 4, 5},
			want:  []float64{1, 1, 2, -2, -2},
		},
		{
			name:  "detrend data with missing trend values",
			data:  []float64{2, 3, 5, 2, 3},
			trend: []float64{math.NaN(), 2, math.NaN(), 4, math.NaN()},
			want:  []float64{math.NaN(), 1, math.NaN(), -2, math.NaN()},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := detrend(tt.data, tt.trend)
			for i := range got {
				if diff := math.Abs(got[i] - tt.want[i]); diff > 0.01 {
					t.Errorf("index %d", i)
					t.Errorf("value %f", tt.want[i])
					t.Errorf("diff, %f", diff)
					t.Errorf("value %f", got[i])
					t.Error("expected:", tt.want)
					t.Error("got:", got)
				}
			}
		})
	}
}

func TestDecomposeAdditive(t *testing.T) {
	tests := []struct {
		name string // description of this test case
		// Named input parameters for target function.
		data       []float64
		windowSize int
		useCenter  bool
		want       DecompositionResult
	}{
		{
			name:       "decompose simple linear data",
			data:       []float64{1, 2, 3, 4, 5},
			windowSize: 3,
			useCenter:  true,
			want: DecompositionResult{
				Trend:       []float64{math.NaN(), 2, 3, 4, math.NaN()},
				Seasonality: []float64{0, 0, 0, 0, 0},
				Residuals:   []float64{math.NaN(), 0, 0, 0, math.NaN()},
			},
		},
		{
			name:       "decompose data with linear trend and seasonality",
			data:       []float64{2, 3, 5, 2, 3},
			windowSize: 3,
			useCenter:  true,
			want: DecompositionResult{
				Trend:       []float64{math.NaN(), 10.0 / 3, 10.0 / 3, 10.0 / 3, math.NaN()},
				Seasonality: []float64{-4.0 / 3, -1.0 / 3, 5.0 / 3, -4.0 / 3, -1.0 / 3},
				Residuals:   []float64{math.NaN(), 0, 0, 0, math.NaN()},
			},
		},
		{
			// trend is constant 5, seasonal averages all 0, residuals all 0
			name:       "decompose constant data",
			data:       []float64{5, 5, 5, 5, 5, 5, 5},
			windowSize: 3,
			useCenter:  true,
			want: DecompositionResult{
				Trend:       []float64{math.NaN(), 5, 5, 5, 5, 5, math.NaN()},
				Seasonality: []float64{0, 0, 0, 0, 0, 0, 0},
				Residuals:   []float64{math.NaN(), 0, 0, 0, 0, 0, math.NaN()},
			},
		},
		{
			// underlying trend 1..9, seasonal pattern [+1,-2,+1] repeated;
			// data[i] = trend[i] + seasonal[i%3]
			// MA trend recovers exact integer values; seasonal averages match perfectly
			name:       "decompose multiple full periods with clear seasonality",
			data:       []float64{2, 0, 4, 5, 3, 7, 8, 6, 10},
			windowSize: 3,
			useCenter:  true,
			want: DecompositionResult{
				Trend:       []float64{math.NaN(), 2, 3, 4, 5, 6, 7, 8, math.NaN()},
				Seasonality: []float64{1, -2, 1, 1, -2, 1, 1, -2, 1},
				Residuals:   []float64{math.NaN(), 0, 0, 0, 0, 0, 0, 0, math.NaN()},
			},
		},
		{
			// trailing (non-centered) MA: all windows average to 3, detrended
			// pattern [+2, 0, -2] averaged per position gives seasonality [0, -2, +2]
			name:       "decompose periodic data with non-centered moving average",
			data:       []float64{3, 1, 5, 3, 1, 5, 3, 1, 5},
			windowSize: 3,
			useCenter:  false,
			want: DecompositionResult{
				Trend:       []float64{math.NaN(), math.NaN(), 3, 3, 3, 3, 3, 3, 3},
				Seasonality: []float64{0, -2, 2, 0, -2, 2, 0, -2, 2},
				Residuals:   []float64{math.NaN(), math.NaN(), 0, 0, 0, 0, 0, 0, 0},
			},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := DecomposeAdditive(tt.data, tt.windowSize, tt.useCenter)
			for i := range got.Trend {
				if diff := math.Abs(got.Trend[i] - tt.want.Trend[i]); diff > 0.01 {
					t.Errorf("index %d", i)
					t.Errorf("value %f", tt.want.Trend[i])
					t.Errorf("diff, %f", diff)
					t.Errorf("value %f", got.Trend[i])
					t.Error("expected:", tt.want.Trend)
					t.Error("got:", got.Trend)
				}
			}
			for i := range got.Seasonality {
				if diff := math.Abs(got.Seasonality[i] - tt.want.Seasonality[i]); diff > 0.01 {
					t.Errorf("index %d", i)
					t.Errorf("value %f", tt.want.Seasonality[i])
					t.Errorf("diff, %f", diff)
					t.Errorf("value %f", got.Seasonality[i])
					t.Error("expected:", tt.want.Seasonality)
					t.Error("got:", got.Seasonality)
				}
			}
			for i := range got.Residuals {
				if diff := math.Abs(got.Residuals[i] - tt.want.Residuals[i]); diff > 0.01 {
					t.Errorf("index %d", i)
					t.Errorf("value %f", tt.want.Residuals[i])
					t.Errorf("diff, %f", diff)
					t.Errorf("value %f", got.Residuals[i])
					t.Error("expected:", tt.want.Residuals)
					t.Error("got:", got.Residuals)
				}
			}
		})
	}
}
