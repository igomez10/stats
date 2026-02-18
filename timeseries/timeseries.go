package timeseries

import (
	"math"

	"github.com/igomez10/stats/pkg"
)

// GetTrend returns the trend of the given data
// The trend is calculated using a centered moving average with
// the specified window size. The first and last floor(window/2) values will be NaN.
func GetTrend(data []float64, window int, useCenter bool) []float64 {
	if len(data) < window {
		panic("invalid window size cannot be larger than data length")
	}
	res := make([]float64, len(data))
	for i := range res {
		res[i] = math.NaN()
	}

	if !useCenter {
		for i := window - 1; i < len(data); i++ {
			counter := 0.0
			for j := 0; j < window; j++ {
				counter += data[i-j]
			}
			res[i] = counter / float64(window)
		}
		return res
	}

	half := window / 2
	even := window%2 == 0

	for i := half; i < len(data)-half; i++ {
		counter := data[i]
		for j := 1; j <= half; j++ {
			counter += data[i-j]
			counter += data[i+j]
		}
		if even {
			counter -= 0.5 * data[i-half]
			counter -= 0.5 * data[i+half]
		}
		res[i] = counter / float64(window)
	}
	return res
}

// GetSeasonality returns the seasonal component of the given data using additive decomposition
func GetSeasonality(detrended []float64, windowSize int, dtype DecompositionType) []float64 {
	// Average each seasonal position
	seasonalAvg := make([]float64, windowSize)
	counts := make([]int, windowSize)
	for i := range detrended {
		if math.IsNaN(detrended[i]) {
			continue
		}

		indexInWindow := i % windowSize
		seasonalAvg[indexInWindow] += detrended[i]
		counts[indexInWindow]++
	}
	for k := 0; k < windowSize; k++ {
		if counts[k] > 0 {
			seasonalAvg[k] /= float64(counts[k])
		}
	}

	switch dtype {
	case AdditiveDecomposition:
		// Center the seasonal component by subtracting the mean of the seasonal averages
		mean := pkg.GetMean(seasonalAvg)
		for i := range seasonalAvg {
			seasonalAvg[i] -= mean
		}
	case MultiplicativeDecomposition:
		// Center the seasonal component by dividing by the mean of the seasonal averages
		mean := pkg.GetMean(seasonalAvg)
		for i := range seasonalAvg {
			if mean != 0 {
				seasonalAvg[i] /= mean
			} else {
				seasonalAvg[i] = math.NaN()
			}
		}
	}

	// Tile
	res := make([]float64, len(detrended))
	for i := range res {
		res[i] = seasonalAvg[i%windowSize]
	}
	return res
}

// Additive or Multiplicative
type DecompositionType int

const (
	AdditiveDecomposition DecompositionType = iota
	MultiplicativeDecomposition
)

// detrend returns the detrended data by subtracting the trend from the original data
func detrend(data []float64, trend []float64, dtype DecompositionType) []float64 {
	detrended := make([]float64, len(data))
	for i := range data {
		if math.IsNaN(trend[i]) {
			detrended[i] = math.NaN()
		} else {
			switch dtype {
			case AdditiveDecomposition:
				// detrended is the original data minus the trend
				detrended[i] = data[i] - trend[i]
			case MultiplicativeDecomposition:
				// detrended is the original data divided by the trend
				detrended[i] = data[i] / trend[i]
			}
		}
	}
	return detrended
}

// DecompositionResult holds the result of the additive and multiplicative decomposition of a time series
type DecompositionResult struct {
	Trend       []float64
	Seasonality []float64
	Residuals   []float64
}

// DecomposeAdditive performs additive decomposition of the given time series data using the specified window size for trend estimation
func DecomposeAdditive(data []float64, windowSize int, useCenter bool) DecompositionResult {
	trend := GetTrend(data, windowSize, useCenter)
	detrended := detrend(data, trend, AdditiveDecomposition)
	seasonality := GetSeasonality(detrended, windowSize, AdditiveDecomposition)
	residuals := make([]float64, len(data))

	for i := range data {
		if math.IsNaN(detrended[i]) {
			residuals[i] = math.NaN()
		} else {
			residuals[i] = detrended[i] - seasonality[i]
		}
	}
	return DecompositionResult{
		Trend:       trend,
		Seasonality: seasonality,
		Residuals:   residuals,
	}
}

func DecomposeMultiplicative(data []float64, windowSize int, useCenter bool) DecompositionResult {
	trend := GetTrend(data, windowSize, useCenter)
	residuals := make([]float64, len(data))
	detrended := detrend(data, trend, MultiplicativeDecomposition)
	seasonality := GetSeasonality(detrended, windowSize, MultiplicativeDecomposition)
	for i := range data {
		if math.IsNaN(detrended[i]) {
			seasonality[i] = math.NaN()
			residuals[i] = math.NaN()
		} else {
			residuals[i] = detrended[i] / seasonality[i]
		}
	}
	return DecompositionResult{
		Trend:       trend,
		Seasonality: seasonality,
		Residuals:   residuals,
	}
}
