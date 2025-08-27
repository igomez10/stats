package linearregression

import (
	"fmt"
	"log"
	"reflect"
	"testing"
)

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

func TestGetSSX(t *testing.T) {
	type args struct {
		x []float64
	}
	tests := []struct {
		name string
		args args
		want float64
	}{
		{"case1", args{[]float64{1, 2, 3}}, 2.0},
		{"case2", args{[]float64{4, 5, 6}}, 2.0},
		{"case3", args{[]float64{7, 8, 9}}, 2.0},
		{"case4", args{[]float64{10, 11, 12}}, 2.0},
		{"case5", args{[]float64{1, 100}}, 4900.5},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := GetSSX(tt.args.x); got != tt.want {
				t.Errorf("GetSSX() = %v, want %v", got, tt.want)
			}
		})
	}
}

func Test_mean(t *testing.T) {
	type args struct {
		a []float64
	}
	tests := []struct {
		name string
		args args
		want float64
	}{
		{"case1", args{[]float64{1, 2, 3}}, 2.0},
		{"case2", args{[]float64{4, 5, 6}}, 5.0},
		{"case3", args{[]float64{7, 8, 9}}, 8.0},
		{"case4", args{[]float64{10, 11, 12}}, 11.0},
		{"case5", args{[]float64{1, 100}}, 50.5},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := mean(tt.args.a); got != tt.want {
				t.Errorf("mean() = %v, want %v", got, tt.want)
			}
		})
	}
}

