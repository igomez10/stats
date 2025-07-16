package stats

import (
	"fmt"
	"math"
	"strconv"
	"testing"

	"github.com/NimbleMarkets/ntcharts/barchart"
	"github.com/charmbracelet/lipgloss"
)

func TestSomemain(t *testing.T) {
	// Example 1: Manual PMF creation
	fmt.Println("=== Manual PMF Example ===")
	pmf1 := NewPMF()
	// this means P(X=1) = 0.2, P(X=2) = 0.3, P(X=3) = 0.5
	pmf1.Set(1, 0.2)
	pmf1.Set(2, 0.3)
	pmf1.Set(3, 0.5)
	pmf1.Print()
}

func TestPMFRollingTwoDicesAndSum(t *testing.T) {
	fmt.Println("=== PMF of Rolling Two Dice and Summing ===")
	pmf := NewPMF()
	dice1Sides := 6
	dice2Sides := 6

	frequency := map[int]int{}
	for i := 1; i <= dice1Sides; i++ {
		for j := 1; j <= dice2Sides; j++ {
			sum := i + j
			frequency[sum]++
		}
	}
	possibleValues := len(frequency)

	fmt.Printf("Possible values when rolling two dice: %d\n", possibleValues)
	fmt.Println("Frequency of sums:")
	for sum, freq := range frequency {
		fmt.Printf("Sum %d: %d times\n", sum, freq)
	}

	// Set probabilities based on frequency
	for sum, freq := range frequency {
		pmf.Set(sum, float64(freq)/float64(36)) // 36 is the total number of outcomes (6*6)
	}

	fmt.Println("Bar chart:")
	// Assuming you have a barchart package to visualize the PMF
	datapoints := []barchart.BarData{}
	for sum, prob := range pmf.values {
		datapoints = append(datapoints, barchart.BarData{
			Label: strconv.Itoa(sum),
			Values: []barchart.BarValue{
				{
					Name:  fmt.Sprintf("Sum %d", sum),
					Value: prob * 100,
					Style: lipgloss.NewStyle().Foreground(lipgloss.Color("10"))}, // green
			},
		})
	}

	bc := barchart.New(100, 20)
	bc.PushAll(datapoints)
	bc.Draw()

	fmt.Println(bc.View())
}

func TestBinomialDistribution(t *testing.T) {
	fmt.Println("=== Binomial Distribution PMF ===")
	pmf := CreateBinomialPMF(10, 0.5)
	pmf.Print()
}

func TestIntegral(t *testing.T) {
	type TestCase struct {
		name     string
		from     float64
		to       float64
		step     float64
		function func(float64) float64
		expected float64
	}

	testCases := []TestCase{
		{
			name: "Test 1",
			from: 0,
			to:   1,
			step: 0.001,
			function: func(x float64) float64 {
				return math.Pow(x, 2) // Example function: f(x) = x^2
			},
			expected: 0.3333,
		},
		{
			name: "Test 2",
			from: -1,
			to:   1,
			step: 0.001,
			function: func(x float64) float64 {
				return math.Pow(x, 2) // Example function: f(x) = x^2
			},
			expected: 0.6667,
		},
	}

	for _, tc := range testCases {
		res := Integrate(tc.from, tc.to, tc.step, tc.function)
		if math.Abs(res-tc.expected) > 0.01 {
			t.Errorf("Expected %.4f but got %.4f", tc.expected, res)
		}
	}
}

func Integrate(from, to, step float64, fx func(float64) float64) float64 {
	var res float64 = 0
	for x := from; x < to; x += step {
		res += fx(x) * step
	}
	return res
}

// TestGetCumulativeDistribution tests the cumulative distribution function
func TestGetCumulativeDistribution(t *testing.T) {
	fmt.Println("=== Cumulative Distribution Function (CDF) ===")
	pmf := NewPMF()
	pmf.Set(1, 0.2)
	pmf.Set(2, 0.3)
	pmf.Set(3, 0.5)
	pmf.Normalize()

	cdf := make(map[int]float64)
	var cumulative float64
	for _, value := range pmf.GetSpace() {
		cumulative += pmf.Get(value)
		cdf[value] = cumulative
	}
	fmt.Println("CDF:")
	for value, prob := range cdf {
		fmt.Printf("P(X <= %d) = %.4f\n", value, prob)
	}

	// Check if the last value in CDF is 1
	lastValue := pmf.GetSpace()[len(pmf.GetSpace())-1]
	if math.Abs(cdf[lastValue]-1.0) > 0.01 {
		t.Errorf("Expected CDF at last value to be 1, got %.4f", cdf[lastValue])
	}
}

func TestPDF(t *testing.T) {
	fmt.Println("=== Probability Density Function (PDF) ===")
	pdf := NewPDF(func(x float64) float64 {
		if x < -3 || x > 10 {
			return 0
		}
		return 1 / (10 - (-3))
	}, -3, 10)

	// Example usage of PDF
	fmt.Println("PDF at x=0.5:", pdf.function(0.5))
	fmt.Println("PDF at x=1.5:", pdf.function(1.5)) // Should return 0 since it's outside the range
	pdf.function = func(x float64) float64 {
		return 1 / (pdf.rangeMax - pdf.rangeMin)
	}
	fmt.Println("PDF:", pdf.function(0.5))
}
