package stats

import (
	"fmt"
	"math"
)

// PDF represents a Probability Density Function
// It contains a function that defines the PDF and the range over which it is defined.
// The function must be non-negative and integrate to 1 over the specified range.
// The PDF is defined by a function f(x) where x is in the range [rangeMin, rangeMax].
// The PDF can be used to calculate probabilities for **continuous** random variables.
// The PDF is defined for a continuous random variable X such that P(a <= X <= b) = ∫[a,b] f(x) dx
// where f(x) is the PDF function.
// The PDF is normalized so that the integral over the range equals 1.
type PDF struct {
	function func(float64) float64
	rangeMin float64
	rangeMax float64
}

// ValidatePDF checks if the PDF function is valid
// It ensures that the function is non-negative in the specified range
// and that it integrates to 1 over the range.
// It panics if the function is invalid.
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
	// since we are using a numerical approximation, we allow a small tolerance
	if math.Abs(total-1.0) > 0.01 {
		panic("Function must integrate to 1 over the specified range: " + fmt.Sprintf("integral from %f to %f = %f", min, max, total))
	}
}

// NewPDF creates a new PDF with the given function and range
// It validates the function and range before returning the PDF.
// It panics if the function is invalid or if the range is not valid.
// The function must be non-negative and integrate to 1 over the specified range.
// The range must be defined such that min < max.
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

// Print prints the PDF function values at regular intervals within the specified range
func (pdf *PDF) Print(step float64) {
	fmt.Printf("PDF from %.2f to %.2f with step %.2f:\n", pdf.rangeMin, pdf.rangeMax, step)
	for x := pdf.rangeMin; x <= pdf.rangeMax; x += step {
		fmt.Printf("f(%.2f) = %.4f\n", x, pdf.function(x))
	}
}
