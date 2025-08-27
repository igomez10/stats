package linearregression

import (
	"fmt"
	"log"
	"testing"
)

type SimpleLR struct {
	B0 float64 // intercept
	B1 float64 // slope
}

func (m SimpleLR) GetIntercept() float64 {
	return m.B0
}

func (m SimpleLR) GetSlope() float64 {
	return m.B1
}

func FitSimpleLR(x, y []float64) (SimpleLR, error) {
	if len(x) != len(y) || len(x) == 0 {
		return SimpleLR{}, fmt.Errorf("x and y must have same nonzero length")
	}
	n := float64(len(x))

	var sx, sy float64
	for i := range x {
		sx += x[i]
		sy += y[i]
	}
	xbar := sx / n
	ybar := sy / n

	var sxx, sxy float64
	for i := range x {
		dx := x[i] - xbar
		dy := y[i] - ybar
		sxx += dx * dx
		sxy += dx * dy
	}
	if sxx == 0 {
		return SimpleLR{}, fmt.Errorf("zero variance in x")
	}
	b1 := sxy / sxx
	b0 := ybar - b1*xbar
	return SimpleLR{B0: b0, B1: b1}, nil
}

func (m SimpleLR) Predict(x float64) float64 {
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
