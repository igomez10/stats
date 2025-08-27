package linearregression

import (
	"fmt"
	"log"
	"reflect"
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

func TestModel_Predict(t *testing.T) {
	type args struct {
		x float64
	}
	tests := []struct {
		name string
		m    Model
		args args
		want float64
	}{
		{"case 1", Model{B0: 1, B1: 2}, args{3}, 7},
		{"case 2", Model{B0: 0, B1: 1}, args{5}, 5},
		{"case 3", Model{B0: -1, B1: 1}, args{0}, -1},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := tt.m.Predict(tt.args.x); got != tt.want {
				t.Errorf("Model.Predict() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestFitSimpleLR(t *testing.T) {
	type args struct {
		x []float64
		y []float64
	}
	tests := []struct {
		name    string
		args    args
		want    Model
		wantErr bool
	}{
		{
			name: "valid case",
			args: args{
				x: []float64{1, 2, 3, 4, 5},
				y: []float64{1, 2, 3, 4, 5},
			},
			want:    Model{B0: 0, B1: 1},
			wantErr: false,
		},
		{
			name: "invalid case - different lengths",
			args: args{
				x: []float64{1, 2, 3},
				y: []float64{1, 2, 3, 4, 5},
			},
			want:    Model{},
			wantErr: true,
		},
		{
			name: "invalid case - zero variance",
			args: args{
				x: []float64{1, 1, 1, 1, 1},
				y: []float64{1, 2, 3, 4, 5},
			},
			want:    Model{},
			wantErr: true,
		},
		{
			name: "y = 2x",
			args: args{
				x: []float64{1, 2, 3, 4, 5},
				y: []float64{2, 4, 6, 8, 10},
			},
			want:    Model{B0: 0, B1: 2},
			wantErr: false,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := FitSimpleLR(tt.args.x, tt.args.y)
			if (err != nil) != tt.wantErr {
				t.Errorf("FitSimpleLR() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if !reflect.DeepEqual(got, tt.want) {
				t.Errorf("FitSimpleLR() = %v, want %v", got, tt.want)
			}
		})
	}
}
