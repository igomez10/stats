package linearregression

import (
	"fmt"
	"math"
)

type Model struct {
	B0 float64 // intercept
	B1 float64 // slope
}

func (m Model) GetIntercept() float64 {
	return m.B0
}

func (m Model) GetSlope() float64 {
	return m.B1
}

func FitSimpleLR(x, y []float64) (Model, error) {
	if len(x) != len(y) || len(x) == 0 {
		return Model{}, fmt.Errorf("x and y must have same nonzero length")
	}
	sampleSize := float64(len(x))

	// lets find sumX and sumY
	var sumX, sumY float64
	for i := range x {
		sumX += x[i]
		sumY += y[i]
	}
	meanX := sumX / sampleSize
	meanY := sumY / sampleSize

	// Lets find sumOfSquaredDiffX and sumOfSquaredDiffXY
	var sumOfSquaredDiffX, sumOfSquaredDiffXY float64
	for i := range x {
		diffXi := x[i] - meanX
		diffYi := y[i] - meanY
		sumOfSquaredDiffX += diffXi * diffXi
		sumOfSquaredDiffXY += diffXi * diffYi
	}
	if sumOfSquaredDiffX == 0 {
		return Model{}, fmt.Errorf("zero variance in x")
	}

	// find b1
	slopeCoefficient := sumOfSquaredDiffXY / sumOfSquaredDiffX
	// find b0
	intercept := meanY - slopeCoefficient*meanX

	return Model{B0: intercept, B1: slopeCoefficient}, nil
}

func (m Model) Predict(xInput float64) float64 {
	return m.B0 + m.B1*xInput
}

// SSX is the sum of squares of x
// ∑(xi - x̄)²
func GetSSX(x []float64) float64 {
	// meanX is the mean of x
	meanX := mean(x)
	ssx := 0.0
	for _, xi := range x {
		diffXi := xi - meanX
		ssx += diffXi * diffXi
	}
	return ssx
}

// GetSSXY is the sum of products of deviations of x and y
// ∑(xi - x̄)(yi - ȳ)
func GetSSXY(x, y []float64) float64 {
	meanX := mean(x)
	meanY := mean(y)
	ssxy := 0.0
	for i := range x {
		diffXi := x[i] - meanX // xi - x̄
		diffYi := y[i] - meanY // yi - ȳ
		ssxy += diffXi * diffYi
	}
	return ssxy
}

// helpers
func sum(a []float64) float64 {
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

// sumSquares is the sum of squares of a slice
func sumSquares(a []float64) float64 {
	s := 0.0
	for _, v := range a {
		s += v * v
	}
	return s
}

// mean is the average of a slice
func mean(a []float64) float64 {
	if len(a) == 0 {
		return 0
	}
	return sum(a) / float64(len(a))
}

// sampleVar is the sample variance of a slice
func sampleVar(a []float64) float64 {
	n := float64(len(a))
	if n < 2 {
		return 0
	}
	m := mean(a)
	ss := 0.0
	for _, v := range a {
		d := v - m
		ss += d * d
	}
	return ss / (n - 1)
}

// norm2 is the Euclidean norm (L2 norm) of a slice
func norm2(a []float64) float64 {
	return math.Sqrt(sumSquares(a))
}

// aliased notations for Sum Squares Total
func GetSST(x, y []float64) float64 {
	return GetSumSquaresTotal(x, y)
}
func GetTSS(x, y []float64) float64 {
	return GetSumSquaresTotal(x, y)
}

// GetSumSquaresTotal Measures the total variability of the dataset
func GetSumSquaresTotal(x, y []float64) float64 {
	res := 0.0
	meanY := mean(y)
	for i := range y {
		errI := y[i] - meanY
		res += errI * errI
	}
	return res
}

// GetSumSquaresRegression Measures the explained variability by your line
func GetSumSquaresRegression(x, y []float64) float64 {
	sum := 0.0
	meanY := mean(y)
	model, err := FitSimpleLR(x, y)
	if err != nil {
		panic(err)
	}

	for i := range y {
		predictedYi := model.Predict(x[i])
		errI := predictedYi - meanY
		sum += errI * errI
	}

	return sum
}

var commonNotation = map[string]string{
	"SST": "Total Sum of Squares",
	"SSR": "Sum of Squares Regression",
	"SSE": "Sum of Squares Error",
}

func GetSSR(x, y []float64) float64 {
	return GetSumSquaresRegression(x, y)
}

func GetESS(x, y []float64) float64 {
	return GetSumSquaresRegression(x, y)
}

func GetRSS(x, y []float64) float64 {
	return GetSumSquaresError(x, y)
}

func GetSSE(x, y []float64) float64 {
	return GetSumSquaresError(x, y)
}

// GetSumSquaresError Measures the unexplained variability by the regression
// The difference between our predicted and the actual values
func GetSumSquaresError(x, y []float64) float64 {
	sum := 0.0
	model, err := FitSimpleLR(x, y)
	if err != nil {
		panic(err)
	}

	for i := range y {
		predictedYi := model.Predict(x[i])
		actualYi := y[i]

		errI := predictedYi - actualYi
		sum += errI * errI
	}

	return sum
}
