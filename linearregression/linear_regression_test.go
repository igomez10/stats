package linearregression

import (
	"fmt"
	"log"
	"testing"
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

func (m Model) Predict(x float64) float64 {
	return m.B0 + m.B1*x
}

func TestExampleSimpleLinearRegression(t *testing.T) {
	// Tiny example
	x := []float64{1, 2, 3, 4, 5}
	y := []float64{2.1, 2.9, 3.7, 4.1, 5.2}

	model, err := FitSimpleLR(x, y)
	if err != nil {
		log.Fatal(err)
	}
	fmt.Printf("SimpleLR: y = %.4f + %.4f*x\n", model.B0, model.B1)
	fmt.Println("Pred at x=6:", model.Predict(6))
}
