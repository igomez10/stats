package stats

import (
	"testing"
)

func TestPDF_GetExpectedValue(t *testing.T) {
	type fields struct {
		function func(float64) float64
		rangeMin float64
		rangeMax float64
	}
	tests := []struct {
		name   string
		fields fields
		want   float64
	}{
		{
			name: "Uniform Distribution",
			fields: fields{
				function: func(x float64) float64 {
					if x < 0 || x > 1 {
						return 0
					}
					return 1
				},
				rangeMin: 0,
				rangeMax: 1,
			},
			want: 0.5,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			pdf := &PDF{
				function: tt.fields.function,
				rangeMin: tt.fields.rangeMin,
				rangeMax: tt.fields.rangeMax,
			}
			if got := pdf.GetExpectedValue(); got-tt.want > 0.001 {
				t.Errorf("PDF.GetExpectedValue() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestPDF_GetVariance(t *testing.T) {
	type fields struct {
		function func(float64) float64
		rangeMin float64
		rangeMax float64
	}
	tests := []struct {
		name   string
		fields fields
		want   float64
	}{
		{
			name: "Uniform Distribution",
			fields: fields{
				function: func(x float64) float64 {
					if x < 0 || x > 1 {
						return 0
					}
					return 1
				},
				rangeMin: 0,
				rangeMax: 1,
			},
			want: 0.08333333333333333,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			pdf := &PDF{
				function: tt.fields.function,
				rangeMin: tt.fields.rangeMin,
				rangeMax: tt.fields.rangeMax,
			}
			if got := pdf.GetVariance(); got-tt.want > 0.001 {
				t.Errorf("PDF.GetVariance() = %v, want %v", got, tt.want)
			}
		})
	}
}
