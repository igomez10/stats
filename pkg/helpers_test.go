package pkg

import (
	"testing"
)

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
			if got := GetMean(tt.args.a); got != tt.want {
				t.Errorf("GetMean() = %v, want %v", got, tt.want)
			}
		})
	}
}
