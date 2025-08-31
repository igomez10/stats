package linearregression

import (
	"fmt"
	"math"
	"stats/pkg"
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

func CreateSLRModel(x, y []float64) (Model, error) {
	if len(x) != len(y) || len(x) == 0 {
		return Model{}, fmt.Errorf("x and y must have same nonzero length")
	}

	meanX := pkg.GetMean(x)
	meanY := pkg.GetMean(y)

	ssx := GetSSX(x)
	if ssx == 0 {
		return Model{}, fmt.Errorf("zero variance in x")
	}
	ssxy := GetSSXY(x, y)

	// find b1
	slopeCoefficient := ssxy / ssx
	// find b0
	intercept := meanY - slopeCoefficient*meanX

	return Model{B0: intercept, B1: slopeCoefficient}, nil
}

// Predict will predict y at x
func (m Model) Predict(xInput float64) float64 {
	return m.B0 + m.B1*xInput
}

// SSX is the sum of squares of x
// ∑(xi - x̄)²
func GetSSX(x []float64) float64 {
	meanX := pkg.GetMean(x)
	sumSoFar := 0.0

	for _, xi := range x {
		diffXi := xi - meanX
		sumSoFar += diffXi * diffXi
	}

	return sumSoFar
}

// GetSSXY is the sum of products of deviations of x and y
// ∑(xi - x̄)(yi - ȳ)
// This is also just the covariance multiplied by n-1
func GetSSXY(x, y []float64) float64 {
	if len(x) != len(y) {
		panic("Incompatible slice lengths")
	}

	meanX := pkg.GetMean(x)
	meanY := pkg.GetMean(y)
	sumSoFar := 0.0

	for i := range x {
		diffXi := x[i] - meanX // xi - x̄
		diffYi := y[i] - meanY // yi - ȳ
		sumSoFar += diffXi * diffYi
	}

	return sumSoFar
}

// sumSquares is the sum of squares of a slice
func sumSquares(a []float64) float64 {
	s := 0.0
	for _, v := range a {
		s += v * v
	}
	return s
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
	meanY := pkg.GetMean(y)
	for i := range y {
		errI := y[i] - meanY
		res += errI * errI
	}
	return res
}

// GetSumSquaresRegression Measures the explained variability by your line
func GetSumSquaresRegression(x, y []float64) float64 {
	sum := 0.0
	meanY := pkg.GetMean(y)
	model, err := CreateSLRModel(x, y)
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
	model, err := CreateSLRModel(x, y)
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

// norm2 is the Euclidean norm (L2 norm) of a slice
func norm2(a []float64) float64 {
	return math.Sqrt(sumSquares(a))
}
