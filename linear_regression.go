package stats

import (
	"fmt"
	"math"

	"github.com/igomez10/stats/pkg"

	"github.com/igomez10/linearalgebra"
)

type InferenceModel interface {
	Predict(xInput []float64) float64
}

type SimpleModel struct {
	B0 float64 // intercept
	B1 float64 // slope
}

func (m SimpleModel) GetIntercept() float64 {
	return m.B0
}

func (m SimpleModel) GetSlope() float64 {
	return m.B1
}

// GetCoefficientDetermination returns the R2 of the model
// This will tell us how much variance is explained by the model.
// We can use this to judge if our model is good or not
func GetCoefficientDetermination(x, y []float64) float64 {
	sse := GetSSE(x, y)
	sst := GetSSTSimple(x, y)
	return 1 - (sse / sst)
}

// CreateSLRModelWithOLS creates a simple linear regression model using ordinary least squares
func CreateSLRModelWithOLS(x, y []float64) (SimpleModel, error) {
	if len(x) != len(y) || len(x) == 0 {
		return SimpleModel{}, fmt.Errorf("x and y must have same nonzero length")
	}

	meanX := pkg.GetMean(x)
	meanY := pkg.GetMean(y)

	ssx := GetSSXSimple(x)
	if ssx == 0 {
		return SimpleModel{}, fmt.Errorf("zero variance in x")
	}
	ssxy := GetSSXYSimple(x, y)

	// Find b0 and b1 by the method of least squares
	// find b1
	slopeCoefficient := ssxy / ssx
	// find b0
	intercept := meanY - slopeCoefficient*meanX

	return SimpleModel{B0: intercept, B1: slopeCoefficient}, nil
}

func GetLogLikelihoodFunctionLinearRegression(x, y []float64) func(b0, b1, sigma2 float64) float64 {
	return func(b0, b1, sigma2 float64) float64 {
		if sigma2 <= 0 {
			return math.Inf(-1)
		}

		c := 0.0
		c += float64((-len(x) / 2) * int(math.Log(2*math.Pi)))
		c += float64((-len(x) / 2) * int(math.Log(sigma2)))
		counter := 0.0
		for i := range x {
			counter += math.Pow(y[i]-b0-b1*x[i], 2.0)
		}
		c += float64(-1/(2*sigma2)) * counter

		return c
	}
}

// Predict will predict y at x
func (m SimpleModel) Predict(xInput float64) float64 {
	return m.B0 + m.B1*xInput
}

// SSX is the sum of squares of x
// ∑(xi - x̄)²
func GetSSXSimple(x []float64) float64 {
	meanX := pkg.GetMean(x)
	sumSoFar := 0.0

	for _, xi := range x {
		diffXi := xi - meanX
		sumSoFar += diffXi * diffXi
	}

	return sumSoFar
}

// GetSSXYSimple is the sum of products of deviations of x and y
// ∑(xi - x̄)(yi - ȳ)
// This is also just the covariance multiplied by n-1
func GetSSXYSimple(x, y []float64) float64 {
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
func GetSSTSimple(x, y []float64) float64 {
	return GetSumSquaresTotalSimple(x, y)
}
func GetTSSSimple(x, y []float64) float64 {
	return GetSumSquaresTotalSimple(x, y)
}

// GetSumSquaresTotalSimple Measures the total variability of the dataset
func GetSumSquaresTotalSimple(x, y []float64) float64 {
	res := 0.0
	meanY := pkg.GetMean(y)
	for i := range y {
		errI := y[i] - meanY
		res += errI * errI
	}
	return res
}

// GetSumSquaresRegressionSimple Measures the explained variability by your line
func GetSumSquaresRegressionSimple(x, y []float64) float64 {
	sum := 0.0
	meanY := pkg.GetMean(y)
	model, err := CreateSLRModelWithOLS(x, y)
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
	return GetSumSquaresRegressionSimple(x, y)
}

func GetESS(x, y []float64) float64 {
	return GetSumSquaresRegressionSimple(x, y)
}

func GetRSS(x, y []float64) float64 {
	return GetSumSquaresErrorSimple(x, y)
}

func GetSSE(x, y []float64) float64 {
	return GetSumSquaresErrorSimple(x, y)
}

// GetSumSquaresErrorSimple Measures the unexplained variability by the regression
// The difference between our predicted and the actual values
func GetSumSquaresErrorSimple(x []float64, y []float64) float64 {
	sum := 0.0
	model, err := CreateSLRModelWithOLS(x, y)
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

// GetMSESimple is the mean squared error
func GetMSESimple(x, y []float64) float64 {
	return GetSumSquaresErrorSimple(x, y) / (float64(len(y)) - 2)
}

func GetVarianceB1(x, y []float64) float64 {
	return GetVariance(x) / GetSSXSimple(x)
}

// GetStandardErrorB1 returns the standard error for the slope
func GetStandardErrorB1(x, y []float64) float64 {
	mse := GetMSESimple(x, y)
	ssx := GetSSXSimple(x)
	return math.Sqrt(mse / ssx)
}

// GetSlopeFromSSXYAndSSX calculates the slope from the sum of squares
// Slope can be thought as a relation between the covariance of x,y and the variance of x
// This follows the method of least squares
func GetSlopeFromSSXYAndSSX(x, y []float64) float64 {
	ssx := GetSSXSimple(x)
	ssxy := GetSSXYSimple(x, y)
	return ssxy / ssx
}

// GetInterceptFromSlopeAndMeans calculates the intercept from the slope and means
// This follows the method of least squares
func GetInterceptFromSlopeAndMeans(slope float64, meanX, meanY float64) float64 {
	return meanY - slope*meanX // B0 = ȳ - B1*x̄
}

// GetDesignMatrix creates the design matrix for multiple linear regression
// The design matrix includes a column of ones for the intercept term
// Each row corresponds to an observation, and each column corresponds to a feature
// For example, if we have observations = [[x11, x12], [x21, x22]], the design matrix will be:
// [
//
//	[1, x11, x12],
//	[1, x21, x22],
//	[1, x31, x32]
//
// The first column of ones allows us to estimate the intercept term in the regression model.
// ]
func GetDesignMatrix(observations [][]float64) [][]float64 {
	designMatrix := make([][]float64, len(observations))
	for i := range observations {
		designMatrix[i] = make([]float64, len(observations[0])+1)
		designMatrix[i][0] = 1 // intercept term
		for j := range observations[i] {
			designMatrix[i][j+1] = observations[i][j]
		}
	}
	return designMatrix
}

type MultiLinearModel struct {
	Betas []float64
}

func (m MultiLinearModel) Predict(xInput []float64) float64 {
	if len(xInput)+1 != len(m.Betas) {
		panic("Incompatible input length")
	}

	betasNoIntercept := m.Betas[1:]
	yHat := linearalgebra.DotProduct([][]float64{xInput}, linearalgebra.TransposeMatrix([][]float64{betasNoIntercept}))
	return yHat[0][0] + m.Betas[0]
}

// GetMSE computes the Mean Squared Error of the multi linear regression model
// against the provided observations and actual outputs
func (m MultiLinearModel) GetMSE(x [][]float64, y []float64) float64 {
	if len(x) != len(y) {
		panic("Incompatible lengths between observations and actual output")
	}

	sumSquaresError := 0.0
	for i := range y {
		yiHat := m.Predict(x[i])
		errI := yiHat - y[i]
		sumSquaresError += errI * errI
	}

	mse := sumSquaresError / float64(len(y)-len(m.Betas))

	return mse
}

// CreateLRModelWithOLS creates a multi linear regression model using ordinary least squares
func CreateLRModelWithOLS(observations [][]float64, actualOutput []float64) (MultiLinearModel, error) {
	for i := range observations {
		if len(observations[i]) != len(observations[0]) {
			return MultiLinearModel{}, fmt.Errorf("unexpected length in observation")
		}
	}

	designMatrix := GetDesignMatrix(observations)

	Xt := linearalgebra.CopyMatrix(designMatrix)
	Xt = linearalgebra.TransposeMatrix(Xt)

	xCopy := linearalgebra.CopyMatrix(designMatrix)

	XtX := linearalgebra.DotProduct(Xt, xCopy)
	XtXInv := linearalgebra.GetInverseMatrixByDeterminant(XtX)

	Xt2 := linearalgebra.CopyMatrix(designMatrix)
	Xt2 = linearalgebra.TransposeMatrix(Xt2)

	XtX_Inv_Xt := linearalgebra.DotProduct(XtXInv, Xt2)
	Y := make([][]float64, len(actualOutput))
	for i := range actualOutput {
		Y[i] = []float64{actualOutput[i]}
	}
	result := linearalgebra.DotProduct(XtX_Inv_Xt, Y)

	betas := make([]float64, len(result))
	for i := range result {
		betas[i] = result[i][0]
	}

	return MultiLinearModel{Betas: betas}, nil
}

func CreateLRModelWithRidge(observations [][]float64, actualOutput []float64, lambda float64) (MultiLinearModel, error) {
	for i := range observations {
		if len(observations[i]) != len(observations[0]) {
			return MultiLinearModel{}, fmt.Errorf("unexpected length in observation")
		}
	}

	// verify observations are normalized
	for i := range observations[0] {
		col := make([]float64, len(observations))
		for j := range observations {
			col[j] = observations[j][i]
		}
		meanCol := pkg.GetMean(col)
		stdDevCol := math.Sqrt(pkg.GetSampleVariance(col))
		if math.Abs(meanCol) > 1e-6 || math.Abs(stdDevCol-1) > 1e-6 {
			return MultiLinearModel{}, fmt.Errorf("observations must be normalized for ridge regression")
		}
	}

	Xt := linearalgebra.CopyMatrix(observations)
	Xt = linearalgebra.TransposeMatrix(Xt)

	xCopy := linearalgebra.CopyMatrix(observations)

	XtX := linearalgebra.DotProduct(Xt, xCopy)

	// Add lambda*I to XtX
	for i := range XtX {
		XtX[i][i] += lambda
	}
	XtXInv := linearalgebra.GetInverseMatrixByDeterminant(XtX)

	Xt2 := linearalgebra.CopyMatrix(observations)
	Xt2 = linearalgebra.TransposeMatrix(Xt2)

	XtX_Inv_Xt := linearalgebra.DotProduct(XtXInv, Xt2)
	Y := make([][]float64, len(actualOutput))
	for i := range actualOutput {
		Y[i] = []float64{actualOutput[i]}
	}
	result := linearalgebra.DotProduct(XtX_Inv_Xt, Y)

	betas := make([]float64, len(result))
	for i := range result {
		betas[i] = result[i][0]
	}

	return MultiLinearModel{Betas: betas}, nil
}

// Σ(Yi− Yi_hat)² + λ Σ|βj|
// Lasso regression loss function will add a penalty equal to the
// absolute value of the magnitude of coefficients. This function
// can be used to regularize a linear regression model and perform feature selection
// by shrinking some coefficients to zero.
func LassoLossFormula(observations [][]float64, actualOutput []float64, betas []float64, lambda float64) float64 {
	if len(observations) != len(actualOutput) {
		panic("Incompatible lengths between observations and actual output")
	}

	for i := range observations {
		if len(observations[i])+1 != len(betas) {
			panic("Incompatible lengths between observations and betas")
		}
	}

	sse := 0.0
	for i := range actualOutput {
		yiHat := betas[0] // intercept
		for j := 1; j < len(betas); j++ {
			yiHat += observations[i][j-1] * betas[j]
		}
		errI := yiHat - actualOutput[i]
		sse += errI * errI
	}

	// sum abs betas
	sumAbsBetas := 0.0
	for _, beta := range betas {
		sumAbsBetas += math.Abs(beta)
	}

	return sse + lambda*sumAbsBetas
}

// Σ(Yi− Yi_hat)² + λ Σ(βj)²
func RidgeLossFormula(observations [][]float64, actualOutput []float64, betas []float64, lambda float64) float64 {
	sse := 0.0
	for i := range actualOutput {
		yiHat := betas[0] // intercept
		for j := 1; j < len(betas); j++ {
			yiHat += observations[i][j-1] * betas[j]
		}
		errI := yiHat - actualOutput[i]
		sse += errI * errI
	}

	// sum squared betas
	sumSquaredBetas := 0.0
	for _, beta := range betas {
		sumSquaredBetas += beta * beta
	}

	return sse + lambda*sumSquaredBetas
}

// FitModelGradientDescent fits a multi linear regression model using gradient descent
func FitModelGradientDescent(observations [][]float64, actualOutput []float64, learningRate float64, maxIter int) MultiLinearModel {
	model := MultiLinearModel{
		Betas: make([]float64, len(observations[0])+1),
	}

	for iter := 0; iter < maxIter; iter++ {
		gradients := make([]float64, len(model.Betas))

		// Compute gradients
		for i, yi := range actualOutput {
			yiHat := model.Predict(observations[i])
			errI := yiHat - yi
			for currentBeta := 0; currentBeta < len(model.Betas); currentBeta++ {
				if currentBeta == 0 {
					// handle intercept differently
					gradients[currentBeta] += errI
					continue
				}
				gradients[currentBeta] += errI * observations[i][currentBeta-1]
			}
		}

		// Update betas
		for j := range model.Betas {
			model.Betas[j] -= (learningRate * gradients[j])
		}
	}
	return model
}
