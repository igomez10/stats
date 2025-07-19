package stats

import (
	"fmt"
	"math"
	"os"
	"testing"

	"github.com/NimbleMarkets/ntcharts/barchart"
	"github.com/charmbracelet/lipgloss"
)

func TestSomemain(t *testing.T) {
	// Example 1: Manual PMF creation
	t.Log("=== Manual PMF Example ===")
	pmf1 := NewPMF()
	// this means P(X=1) = 0.2, P(X=2) = 0.3, P(X=3) = 0.5
	pmf1.Set(1, 0.2)
	pmf1.Set(2, 0.3)
	pmf1.Set(3, 0.5)
	pmf1.Print(os.Stdout)
}

func TestPMFRollingTwoDicesAndSum(t *testing.T) {
	t.Log("=== PMF of Rolling Two Dice and Summing ===")
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
	possibleValues := func() int {
		total := 0
		for _, freq := range frequency {
			total += freq
		}
		return total
	}()

	t.Logf("Possible values when rolling two dice: %d\n", possibleValues)
	t.Log("Frequency of sums:")
	for sum, freq := range frequency {
		t.Logf("Sum %d: %d times\n", sum, freq)
	}

	// Set probabilities based on frequency
	for sum, freq := range frequency {
		pmf.Set(float64(sum), float64(freq)/float64(possibleValues))
	}

	t.Log("Bar chart:\n")
	// Assuming you have a barchart package to visualize the PMF
	datapoints := []barchart.BarData{}
	for _, summedResult := range pmf.orderedValues {
		datapoints = append(datapoints, barchart.BarData{
			Label: fmt.Sprintf("  %d", int(summedResult)),
			Values: []barchart.BarValue{
				{
					Value: pmf.Get(summedResult) * 10,
					Style: lipgloss.NewStyle().Foreground(lipgloss.Color("10")),
				},
			},
		})
	}

	bc := barchart.New(75, 30)
	bc.PushAll(datapoints)
	bc.Draw()

	fmt.Println(bc.View())
}

func TestBinomialDistribution(t *testing.T) {
	t.Log("=== Binomial Distribution PMF ===")
	pmf := NewBinomialPMF(10, 0.5)
	pmf.Print(os.Stdout)
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

// TestGetCDFFromPMF tests the conversion from PMF to CDF
func TestGetCDFFromPMF(t *testing.T) {
	t.Log("=== Cumulative Distribution Function (CDF) ===")
	pmf := NewPMF()
	pmf.Set(1, 0.2)
	pmf.Set(2, 0.3)
	pmf.Set(3, 0.5)

	cdf := NewCDF()
	var cumulative float64
	for _, value := range pmf.orderedValues {
		cumulative += pmf.Get(value)
		cdf.Set(value, cumulative)
	}
	t.Log("CDF:")
	for _, value := range cdf.orderedValues {
		t.Logf("P(X <= %f) = %.4f\n", value, cdf.Get(value))
	}

	// Check if the last value in CDF is 1
	lastValue := pmf.orderedValues[len(pmf.orderedValues)-1]
	if math.Abs(cdf.Get(lastValue)-1.0) > 0.01 {
		t.Errorf("Expected CDF at last value to be 1, got %.4f", cdf.Get(lastValue))
	}
}

func TestPDF(t *testing.T) {
	t.Log("=== Probability Density Function (PDF) ===")
	pdf := NewPDF(func(x float64) float64 {
		if x < -3 || x > 10 {
			return 0
		}
		return float64(1) / float64((10 - (-3)))
	}, -3, 10)

	// Example usage of PDF
	t.Log("PDF at x=0.5:", pdf.function(0.5))
	t.Log("PDF at x=1.5:", pdf.function(1.5)) // Should return 0 since it's outside the range
	pdf.function = func(x float64) float64 {
		return 1 / (pdf.rangeMax - pdf.rangeMin)
	}
	t.Log("PDF:", pdf.function(0.5))
}

func TestNewPoissonPMF(t *testing.T) {
	t.Log("=== Poisson PMF ===")
	lambda := 3.0
	maxNumEvents := 10
	pmf := NewPoissonPMF(lambda, maxNumEvents)
	pmf.Print(os.Stdout)
	pmf.PrintAsBarChart(os.Stdout)
	// Check if the total sum of probabilities is approximately 1
	totalProb := pmf.TotalSumProbabilities()
	if math.Abs(totalProb-1.0) > 0.01 {
		t.Errorf("Expected total sum of probabilities to be 1, got %.4f", totalProb)
	}
}

func TestNewNormalPDF(t *testing.T) {
	t.Log("=== Normal PDF ===")

	type testCase struct {
		name     string
		mean     float64
		stdDev   float64
		expected float64
	}

	testCases := []testCase{
		{
			name:   "Standard Normal",
			mean:   0,
			stdDev: 1,

			expected: 0.398942,
		},
		{
			name:     "Normal with Mean 5 and StdDev 2",
			mean:     5,
			stdDev:   2,
			expected: 0.199471, // PDF at mean for N(5, 2)
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			pdf := NewNormalPDF(tc.mean, tc.stdDev, -10, 10)
			// Validate the PDF at the mean
			valueAtMean := pdf.function(tc.mean)
			if math.Abs(valueAtMean-tc.expected) > 0.01 {
				t.Errorf("Expected PDF at mean %.2f to be %.4f, got %.4f", tc.mean, tc.expected, valueAtMean)
			}

			// datapoints := []barchart.BarData{}
			// for i := pdf.rangeMin; i <= pdf.rangeMax; i += 1 {
			// 	currentValue := pdf.function(i)
			// 	datapoints = append(datapoints, barchart.BarData{
			// 		Label: fmt.Sprintf("%d", int(i)),
			// 		Values: []barchart.BarValue{
			// 			{
			// 				Name:  fmt.Sprintf("  %d", int(i)),
			// 				Value: currentValue * 10, // Scale for visualization
			// 				Style: lipgloss.NewStyle().Foreground(lipgloss.Color("10")),
			// 			},
			// 		},
			// 	})
			// }

			// bc := barchart.New(100, 30)
			// bc.PushAll(datapoints)
			// bc.Draw()
			// fmt.Println(bc.View())

		})
	}
}

func TestNormalize(t *testing.T) {
	t.Log("=== Normalize ===")
	type testCase struct {
		name     string
		x        float64
		mean     float64
		stdDev   float64
		expected float64
	}
	testCases := []testCase{
		{
			name:     "Standard Normal",
			x:        0,
			mean:     0,
			stdDev:   1,
			expected: 0,
		},
		{
			name:     "Normal with Mean 5 and StdDev 2",
			x:        7,
			mean:     5,
			stdDev:   2,
			expected: 1.0, // (7 - 5) / 2 = 1
		},
		{
			name:     "Negative Value",
			x:        -1,
			mean:     0,
			stdDev:   1,
			expected: -1.0, // (-1 - 0) / 1 = -1
		},
		{
			name:     "Zero Mean and StdDev",
			x:        0,
			mean:     0,
			stdDev:   0.0001, // Small stdDev to avoid division by zero
			expected: 0.0,    // Should still normalize to 0
		},
		{
			name:     "Exercise",
			x:        60,
			mean:     75,
			stdDev:   100,
			expected: -0.15, // (60 - 75) / 100 = -0.15
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			result := Normalize(tc.x, tc.mean, tc.stdDev)
			if math.Abs(result-tc.expected) > 1e-6 {
				t.Errorf("Expected %f, got %f", tc.expected, result)
			}
		})
	}
}

func TestGetNormalDistributionFunction(t *testing.T) {
	stdnormal := NewNormalPDF(0, 1, -10, 10)
	t.Logf("PDF value at x=0: %.4f", stdnormal.function(0))
	integral := Integrate(-20, 0, 0.0001, stdnormal.function)
	t.Logf("Integral from -20 to 0: %.4f", integral)

	// integrate from -20 to -0.15
	integral = Integrate(-20, -0.15, 0.0001, stdnormal.function)
	t.Logf("Integral from -20 to -0.15: %.4f", integral)
}

func TestIntegrate(t *testing.T) {
	type args struct {
		from float64
		to   float64
		step float64
		fx   func(float64) float64
	}
	tests := []struct {
		name string
		args args
		want float64
	}{
		{
			name: "Integrate 100 from -10 to 10",
			args: args{
				from: -10,
				to:   10,
				step: 0.01,
				fx: func(x float64) float64 {
					return 5
				},
			},
			want: 100, // Integral of 100 from -10 to 10 is 100
		},
		{
			name: "Integrate x^2 from 0 to 1",
			args: args{
				from: 0,
				to:   1,
				step: 0.001,
				fx: func(x float64) float64 {
					return x * x // f(x) = x^2
				},
			},
			want: 0.3333, // Integral of x^2 from 0 to 1 is 1/3
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := Integrate(tt.args.from, tt.args.to, tt.args.step, tt.args.fx); (got - tt.want) > 0.1 {
				t.Errorf("Integrate() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestNormalDistributionTable(t *testing.T) {
	t.Log("=== Normal Distribution Table ===")
	mean := 0.0
	stdDev := 1.0
	pdf := NewNormalPDF(mean, stdDev, -3, 3)

	type testCase struct {
		name     string
		zScore   float64
		expected float64
	}

	testCases := []testCase{
		{
			name:     "Z-score 0",
			zScore:   0,
			expected: 0,
		},
		{
			name:     "Z-score 1",
			zScore:   1,
			expected: 0.8413,
		},
		{
			name:     "Z-score -1",
			zScore:   -1,
			expected: 0.1587,
		},
		{
			name:     "Z-score 1.5",
			zScore:   1.5,
			expected: 0.9332,
		},
		{
			name:     "Z-score -1.5",
			zScore:   -1.5,
			expected: 0.0668,
		},
	}
	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			Integrate(-3, tc.zScore, 0.001, pdf.function)
		})
	}
}
func Test3_4_2(t *testing.T) {
	t.Log("=== 3.4.2 Exercise ===")
	mean := 75.0
	variance := 100.0
	x := 60.0
	stdDev := math.Sqrt(variance)
	// Calculate the z-score by normalizing the x value
	z := Normalize(x, mean, stdDev)
	t.Logf("Z-score for x=%.2f: %.4f", x, z)

	// Calculate the integral from -100 to z using mean=0, stdDev=1
	integral := Integrate(-100, z, 0.001, GetNormalDistributionFunction(0, 1))
	t.Logf("Integral from -100 to %.4f: %.3f", z, integral)
}

func Test3_4_3(t *testing.T) {
	t.Log("=== 3.4.3 Exercise ===")
	mean := 0.0
	stdDev := 1.0

	approximationStart := 5.0
	step := 0.001
	for i := 0.0; i <= approximationStart; i += step {
		from := -approximationStart + i
		to := approximationStart - i
		t.Logf("Integrating from %.4f to %.4f", from, to)
		if Integrate(from, to, step, GetNormalDistributionFunction(mean, stdDev))-0.90 < 0.0001 {
			t.Logf("Z-score for 90%%: between %.4f and %.4f", from, to)
			return
		}
	}
	t.Fatalf("Failed to find z-score for 90%% within the range")
}

func Test3_4_4(t *testing.T) {
	t.Log("=== 3.4.4 Exercise ===")
	// we want mean and stddevto so that find P(X < 89 ) = 0.90
	// we can normalize X
}

func TestStudentTDistribution(t *testing.T) {
	t.Log("=== Student's T Distribution ===")
	// Example parameters for Student's T distribution
	degreesOfFreedom := 10

	tDistributionFunction := GetStudentTDistributionFunction(float64(degreesOfFreedom))
	// Example usage of the Student's T distribution PDF
	t.Log("PDF at x=0:", tDistributionFunction(0))
	t.Log("PDF at x=1:", tDistributionFunction(1))
	t.Log("PDF at x=-1:", tDistributionFunction(-1))

	// Integrate the PDF from -3 to 3
	integral := Integrate(-10, 10, 0.001, tDistributionFunction)
	t.Logf("Integral from -10 to 10: %.4f", integral)

	// Check if the integral is approximately equal to 1
	if math.Abs(integral-1.0) > 0.01 {
		t.Errorf("Expected integral to be approximately 1, got %.4f", integral)
	}
}

func TestGetStudentTDistributionFunction(t *testing.T) {
	type args struct {
		degreesOfFreedom float64
		probability      float64
	}
	tests := []struct {
		name string
		args args
		want float64
	}{
		{
			name: "Degrees of Freedom 1 probability 0.9",
			args: args{
				degreesOfFreedom: 1,
				probability:      0.9,
			},
			want: 3.0787,
		},
		{
			name: "Degrees of Freedom 5 probability 0.9",
			args: args{
				degreesOfFreedom: 5,
				probability:      0.9,
			},
			want: 1.4759,
		},
		{
			name: "Degrees of Freedom 10 probability 0.9",
			args: args{
				degreesOfFreedom: 10,
				probability:      0.9,
			},
			want: 1.372,
		},
		{
			name: "Degrees of Freedom 7 probability 0.95",
			args: args{
				degreesOfFreedom: 7,
				probability:      0.95,
			},
			want: 1.895,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tscore := IntegrateUntilValue(
				-1000,
				500,
				tt.args.probability,
				0.0001,
				GetStudentTDistributionFunction(tt.args.degreesOfFreedom),
			)
			if math.Abs(tscore-tt.want) > 0.01 {
				t.Errorf("GetStudentTDistributionFunction() = %v, want %v", tscore, tt.want)
			}
		})
	}
}
