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
func GetSeasonality(originalData, trend []float64, windowSize int) []float64 {
	// Detrend
	detrended := make([]float64, len(originalData))
	for i := range originalData {
		if math.IsNaN(trend[i]) {
			detrended[i] = math.NaN()
		} else {
			detrended[i] = originalData[i] - trend[i]
		}
	}

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

	// Normalize so seasonal averages sum to zero
	grandMean := pkg.GetMean(seasonalAvg)
	for k := 0; k < windowSize; k++ {
		seasonalAvg[k] -= grandMean
	}

	// Tile
	res := make([]float64, len(originalData))
	for i := range res {
		res[i] = seasonalAvg[i%windowSize]
	}
	return res
}

// detrend returns the detrended data by subtracting the trend from the original data
func detrend(data []float64, trend []float64) []float64 {
	detrended := make([]float64, len(data))
	for i := range data {
		if math.IsNaN(trend[i]) {
			detrended[i] = math.NaN()
		} else {
			detrended[i] = data[i] - trend[i]
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
	seasonality := GetSeasonality(data, trend, windowSize)
	residuals := make([]float64, len(data))

	detrended := detrend(data, trend)
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
