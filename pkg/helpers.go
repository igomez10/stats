package pkg

// GetMean is the average of a slice
func GetMean(a []float64) float64 {
	if len(a) == 0 {
		return 0
	}
	return GetSum(a) / float64(len(a))
}

func GetSum(a []float64) float64 {
	s := 0.0
	for _, v := range a {
		s += v
	}
	return s
}

// dot is the dot product of two slices
func dot(a, b []float64) float64 {
	s := 0.0
	for i := range a {
		s += a[i] * b[i]
	}
	return s
}

// sampleVar is the sample variance of a slice
func sampleVar(a []float64) float64 {
	n := float64(len(a))
	if n < 2 {
		return 0
	}
	m := GetMean(a)
	ss := 0.0
	for _, v := range a {
		d := v - m
		ss += d * d
	}
	return ss / (n - 1)
}

func GetMin(a []float64) float64 {
	if len(a) == 0 {
		return 0
	}
	min := a[0]
	for _, v := range a {
		if v < min {
			min = v
		}
	}
	return min
}

func GetMax(a []float64) float64 {
	if len(a) == 0 {
		return 0
	}
	max := a[0]
	for _, v := range a {
		if v > max {
			max = v
		}
	}
	return max
}

func GetVariance(a []float64) float64 {
	mean := GetMean(a)
	varianceSum := 0.0
	for _, v := range a {
		diff := v - mean
		varianceSum += diff * diff
	}
	return varianceSum / float64(len(a))
}
