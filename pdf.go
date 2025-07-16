package stats

import (
	"fmt"
	"math"
)

type PDF struct {
	function func(float64) float64
	rangeMin float64
	rangeMax float64
}

func ValidatePDF(function func(float64) float64, min, max float64) {
	// Check if the function is non-negative in the range
	for x := min; x <= max; x += 0.01 {
		if function(x) < 0 {
			panic("Function must be non-negative in the specified range but got a negative value: " + fmt.Sprintf("f(%f) = %f", x, function(x)))
		}
	}
	// Check if the function integrates to 1 over the range
	total := 0.0
	for x := min; x <= max; x += 0.01 {
		piece := function(x) * 0.01 // Approximate integral using Riemann sum
		total += piece
	}
	if math.Abs(total-1.0) > 0.01 {
		panic("Function must integrate to 1 over the specified range: " + fmt.Sprintf("integral from %f to %f = %f", min, max, total))
	}
}

func NewPDF(function func(float64) float64, min, max float64) *PDF {
	if min >= max {
		panic("Invalid range: min must be less than max")
	}
	if function == nil {
		panic("Function cannot be nil")
	}

	ValidatePDF(function, min, max)

	return &PDF{
		function: function,
		rangeMin: min,
		rangeMax: max,
	}
}

func (pdf *PDF) Print(step float64) {
	fmt.Printf("PDF from %.2f to %.2f with step %.2f:\n", pdf.rangeMin, pdf.rangeMax, step)
	for x := pdf.rangeMin; x <= pdf.rangeMax; x += step {
		fmt.Printf("f(%.2f) = %.4f\n", x, pdf.function(x))
	}
}
