package timeseries

import "math"

// GetTrend returns the trend of the given data
func GetTrend(data []float64, window int) []float64 {
	if len(data) < window {
		panic("invalid window size cannot be larger than data length")
	}
	res := make([]float64, len(data))
	for i := range res {
		res[i] = math.NaN()
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
