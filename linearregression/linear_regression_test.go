package linearregression

import (
	"fmt"
	"log"
	"math"
	"reflect"
	"testing"
	"testing/quick"
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

func TestExplainSSTInTermsOfSSRandSSE(t *testing.T) {
	// we want to prove that SST = SSR + SSE
	// basically total_variability = explained_variability + unexplained_variability
	type testcase struct {
		x   []float64
		y   []float64
		sst float64
		ssr float64
		sse float64
	}
	tests := []testcase{
		{
			x:   []float64{1, 2, 3, 4, 5},
			y:   []float64{2, 3, 5, 4, 6},
			sst: 10.0,
			ssr: 8.10,
			sse: 1.90,
		},
	}
	for _, tt := range tests {
		t.Run(fmt.Sprintf("SST: %v, SSR: %v, SSE: %v", tt.sst, tt.ssr, tt.sse), func(t *testing.T) {
			if got := GetSSE(tt.x, tt.y); got-tt.sse > 1e-5 {
				t.Errorf("GetSSE() = %v, want %v", got, tt.sse)
			}

			if got := GetSSR(tt.x, tt.y); got-tt.ssr > 1e-5 {
				t.Errorf("GetSSR() = %v, want %v", got, tt.ssr)
			}

			if got := GetSST(tt.x, tt.y); got-tt.sst > 1e-5 {
				t.Errorf("GetSST() = %v, want %v", got, tt.sst)
			}

			if GetSST(tt.x, tt.y)-(GetSSR(tt.x, tt.y)+GetSSE(tt.x, tt.y)) > 1e-5 {
				t.Errorf("SST = %v, SSR + SSE = %v", GetSST(tt.x, tt.y), GetSSR(tt.x, tt.y)+GetSSE(tt.x, tt.y))
			}
		})
	}
}

func TestGetSumSquaresRegression(t *testing.T) {
	type args struct {
		x []float64
		y []float64
	}
	tests := []struct {
		name string
		args args
		want float64
	}{
		{
			name: "case1",
			args: args{
				x: []float64{1, 2, 3},
				y: []float64{2, 4, 6},
			},
			want: 8.0,
		},
		{
			name: "case2",
			args: args{
				x: []float64{1, 2, 3, 4, 5},
				y: []float64{2, 3, 5, 4, 6},
			},
			want: 8.10,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := GetSumSquaresRegression(tt.args.x, tt.args.y); got-tt.want > 1e-9 {
				t.Errorf("GetSumSquaresRegression() = %v, want %v", got, tt.want)
			}
		})
	}
}

// Property-based simulation: SSR(x, y) is non-negative for valid inputs
func TestGetSumSquaresRegression_Quick(t *testing.T) {
	// Property: SSR(x, y) is non-negative for valid inputs
	f := func(x []float64) bool {
		if len(x) < 2 {
			return true // vacuously true for too-small inputs
		}
		// sanitize x to avoid NaN/Inf
		xs := make([]float64, len(x))
		for i, v := range x {
			if math.IsNaN(v) || math.IsInf(v, 0) {
				v = float64(i)
			}
			if x[i] > 1e9 || x[i] < -1e9 {
				return true
			}
			xs[i] = v
		}
		// ensure variance in x
		allSame := true
		for _, v := range xs {
			if v != xs[0] {
				allSame = false
				break
			}
		}
		if allSame {
			xs[0] = xs[0] + 1
		}
		// create a linear y from x (valid case)
		y := make([]float64, len(xs))
		for i, xv := range xs {
			y[i] = 1.5*xv + 2.0
			if math.IsInf(y[i], 0) {
				return true
			}
			if y[i] > 1e9 || y[i] < -1e9 {
				return true
			}
		}
		got := GetSumSquaresRegression(xs, y)
		if !math.IsNaN(got) && !math.IsInf(got, 0) && got >= 1e-5 {
			return true
		}

		return false
	}
	if err := quick.Check(f, &quick.Config{}); err != nil {
		t.Error(err)
	}
}
