package stats

import (
	"fmt"
	"math"
	"math/rand/v2"
	"os"
	"reflect"
	"testing"

	"github.com/NimbleMarkets/ntcharts/barchart"
	"github.com/charmbracelet/lipgloss"
	"github.com/igomez10/stats/pkg"
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

func Test3_6_1(t *testing.T) {
	t.Log("=== 3.6.1 Exercise ===")
	// find t value for P(T < 2.228)  with degrees of freedom 10
	degreesOfFreedom := 10.0
	tDistributionFunction := GetStudentTDistributionFunction(degreesOfFreedom)

	t.Logf("Integrate(-1000, 2.228, 0.0001, tDistributionFunction): %v", Integrate(-10000, 2.227, 0.001, tDistributionFunction))
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

func TestGetExponentialDistributionFunction(t *testing.T) {
	type args struct {
		lambda float64
		value  float64
	}
	tests := []struct {
		name     string
		args     args
		expected float64
	}{
		{
			name: "Exponential PDF with lambda 1",
			args: args{
				lambda: 1,
				value:  0,
			},
			expected: 1.0,
		},
		{
			name: "Exponential PDF with lambda 2",
			args: args{
				lambda: 2,
				value:  0,
			},
			expected: 2.0,
		},
		{
			name: "Exponential PDF with lambda 0.5",
			args: args{
				lambda: 0.5,
				value:  0,
			},
			expected: 0.5,
		},
		{
			name: "Exponential PDF with lambda 1 and value 1",
			args: args{
				lambda: 1,
				value:  1,
			},
			expected: 0.3679,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := GetExponentialDistributionFunction(tt.args.lambda)(tt.args.value)
			if math.Abs(got-tt.expected) > 0.01 {
				t.Errorf("GetExponentialDistributionFunction() = %v, want %v", got, tt.expected)
			}
		})
	}
}

func TestGetDerivativeAtX(t *testing.T) {
	type args struct {
		step float64
		fx   func(float64) float64
	}
	tests := []struct {
		name string
		args args
		from float64
		to   float64
		want func(float64) float64
	}{
		{
			name: "Derivative of x^2",
			args: args{
				step: 0.001,
				fx: func(x float64) float64 {
					return x * x // f(x) = x^2
				},
			},
			from: -10,
			to:   10,
			want: func(x float64) float64 {
				return 2 * x // f'(x) = 2x
			},
		},
		{
			name: "Derivative of sin(x)",
			args: args{
				step: 0.001,
				fx: func(x float64) float64 {
					return math.Sin(x)
				},
			},
			from: -math.Pi,
			to:   math.Pi,
			want: func(x float64) float64 {
				return math.Cos(x)
			},
		},
		{
			name: "Derivative of exp(x)",
			args: args{
				step: 0.001,
				fx: func(x float64) float64 {
					return math.Exp(x) // f(x) = e^x
				},
			},
			from: -10,
			to:   10,
			want: func(x float64) float64 {
				return math.Exp(x) // f'(x) = e^x
			},
		},
		{
			name: "Derivative of log(x)",
			args: args{
				step: 0.1,
				fx: func(x float64) float64 {
					return math.Log(x)
				},
			},
			from: 1,
			to:   10,
			want: func(x float64) float64 {
				return 1 / x
			},
		},
		{
			name: "Derivative of a constant function",
			args: args{
				step: 0.001,
				fx: func(x float64) float64 {
					return 42 // f(x) = 42
				},
			},
			from: -10,
			to:   10,
			want: func(x float64) float64 {
				return 0 // f'(x) = 0 for a constant function
			},
		},
		{
			name: "Derivative of a linear function",
			args: args{
				step: 0.001,
				fx: func(x float64) float64 {
					return 2*x + 1 // f(x) = 2x + 1
				},
			},
			from: -10,
			to:   10,
			want: func(x float64) float64 {
				return 2 // f'(x) = 2 for a linear function
			},
		},
		{
			name: "Derivative of a quadratic function",
			args: args{
				step: 0.001,
				fx: func(x float64) float64 {
					return x*x + 3*x + 2 // f(x) = x^2 + 3x + 2
				},
			},
			from: -10,
			to:   10,
			want: func(x float64) float64 {
				return 2*x + 3 // f'(x) = 2x + 3 for a quadratic function
			},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			for i := tt.from; i <= tt.to; i += tt.args.step {
				x := math.Round(i*100) / 100 // Round to avoid floating point precision issues
				got := GetDerivativeAtX(x, tt.args.step, tt.args.fx)
				want := tt.want(x)
				if math.Abs(got-want) > 0.01 {
					t.Errorf("GetDerivativeAtX(%v) = %v, want %v, diff %v", x, got, want, math.Abs(got-want))
				}
			}
		})
	}
}

func TestFindCriticalPoint(t *testing.T) {
	type args struct {
		fx    func(float64) float64
		start float64
		end   float64
		step  float64
	}
	tests := []struct {
		name string
		args args
		want *float64
	}{
		{
			name: "Inflection point of x^2",
			args: args{
				fx:    func(x float64) float64 { return x * x }, // f(x) = x^2
				start: -0.5,
				end:   0.5,
				step:  0.001,
			},
			want: float64Ptr(0), // Inflection point at x = 0 for f(x)
		},
		{
			name: "Inflection point of x^3",
			args: args{
				fx:    func(x float64) float64 { return x * x * x }, // f(x) = x^3
				start: -1,
				end:   1,
				step:  0.00001,
			},
			want: float64Ptr(0), // Inflection point at x = 0 for f(x)
		},
		{
			name: "Inflection point of x^2 + 2x + 1",
			args: args{
				fx:    func(x float64) float64 { return x*x + 2*x + 1 }, // f(x) = x^2 + 2x + 1e
				start: -10,
				end:   10,
				step:  0.0000001,
			},
			want: float64Ptr(-1), // Inflection point at x = -1 for f(x)
		},
		{
			name: "Inflection point of sin(x)",
			args: args{
				fx:    func(x float64) float64 { return math.Sin(x) },
				start: -math.Pi,
				end:   math.Pi,
				step:  0.0001,
			},
			want: float64Ptr(-math.Pi / 2), // Inflection point at x = -π/2 for f(x) = sin(x)
		},
		{
			name: "no critical point for linear function",
			args: args{
				fx:    func(x float64) float64 { return x }, // f(x) = 42
				start: -10,
				end:   10,
				step:  0.001,
			},
			want: nil, // No critical point for a constant function
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := FindCriticalPoint(tt.args.fx, tt.args.start, tt.args.end, tt.args.step)
			if got == nil && tt.want == nil {
				return
			}
			if got == nil || tt.want == nil {
				t.Errorf("FindInflectionPoint() = %v, want %v", got, tt.want)
			}

			if math.Abs(*got-*tt.want) > 0.01 {
				t.Errorf("FindInflectionPoint() = %v, want %v", *got, *tt.want)
			}
			t.Logf("Inflection point found at: %.4f", *got)
		})
	}
}

func float64Ptr(i float64) *float64 {
	return &i
}

func TestGetMaximumLikelihoodEstimationGivenPoisson(t *testing.T) {
	data := []float64{9, 7, 9, 15, 10, 13, 11, 7, 2, 12}

	// We could use a likelihood function to find the MLE for Poisson distribution
	// For Poisson distribution, the likelihood function is:
	// L(lambda) = product((lambda^x * exp(-lambda)) / x!)
	// BUT... since the values are so small, we will use the log-likelihood function instead
	// This is because the product of many small numbers can lead to numerical underflow.
	// The log-likelihood function is:
	// likelihoodFunction := func(lambda float64) float64 {
	// 	likelihood := 1.0
	// 	for _, x := range data {
	// 		likelihood *= (math.Pow(lambda, x) * math.Exp(-lambda)) / Factorial(x)
	// 	}
	// 	return likelihood
	// }

	// define log-likelihood function for Poisson distribution
	logLikelihoodFn := GetLogLikelihoodFunctionPoisson(data)
	// find the critical point of the log-likelihood function (where derivative is 0)

	// start at smallest value in data and end at largest value in data
	// lambda should be in this range
	start := GetMin(data)
	end := GetMax(data)
	criticalPoint := FindCriticalPoint(logLikelihoodFn, start, end, 0.001)
	t.Logf("Maximum Likelihood Estimation (MLE) for Poisson distribution: %.4f", *criticalPoint)
}

func TestGetMax(t *testing.T) {
	type args struct {
		data []float64
	}
	tests := []struct {
		name string
		args args
		want float64
	}{
		{"Max of positive numbers", args{data: []float64{1, 2, 3, 4, 5}}, 5},
		{"Max of negative numbers", args{data: []float64{-1, -2, -3, -4, -5}}, -1},
		{"Max of mixed numbers", args{data: []float64{-1, 2, -3, 4, -5}}, 4},
		{"Max of single element", args{data: []float64{42}}, 42},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := GetMax(tt.args.data); got != tt.want {
				t.Errorf("GetMax() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestGetMin(t *testing.T) {
	type args struct {
		data []float64
	}
	tests := []struct {
		name string
		args args
		want float64
	}{
		{"Min of positive numbers", args{data: []float64{1, 2, 3, 4, 5}}, 1},
		{"Min of negative numbers", args{data: []float64{-1, -2, -3, -4, -5}}, -5},
		{"Min of mixed numbers", args{data: []float64{-1, 2, -3, 4, -5}}, -5},
		{"Min of single element", args{data: []float64{42}}, 42},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := GetMin(tt.args.data); got != tt.want {
				t.Errorf("GetMin() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestGetMaximumLikelihoodExponential(t *testing.T) {
	data := []float64{4.2, 3.1, 5.0, 2.8, 3.7, 6.4, 1.9, 2.6}
	// Define the log-likelihood function for Exponential distribution
	logLikelihoodFn := GetLogLikelihoodFunctionExponential(data)

	// Find the critical point of the log-likelihood function (where derivative is 0)
	start := 0.001
	end := 100.0
	criticalPoint := FindCriticalPoint(logLikelihoodFn, start, end, 0.00001)
	if criticalPoint == nil {
		t.Fatal("Critical point not found")
	}
	t.Logf("Maximum Likelihood Estimation (MLE) for Exponential distribution: %.4f", *criticalPoint)

	if math.Abs(*criticalPoint-0.269) > 0.001 {
		t.Errorf("Expected MLE to be 0.269, got %.3f", *criticalPoint)
	}
}

func TestGetJointDistribution(t *testing.T) {
	constantFn := func(x, y float64) float64 {
		return x - y
	}
	joint := NewJointPDF(constantFn, 0, 1, 0, 1)

	fmt.Println("Joint PDF at (0.5, 0.5):", joint.function(0.5, 0.5))

	// find marginal distribution for x
	marginalX := joint.GetMarginalX(0.01)
	fmt.Println("Marginal PDF for X at 0.5:", marginalX(0.5))
}

func TestJointPMF_GetMarginalX(t *testing.T) {
	type fields struct {
		values map[[2]float64]float64
	}
	tests := []struct {
		name   string
		fields fields
		want   [][]float64
	}{
		{
			name: "Single Value has 1 marginal",
			fields: fields{
				values: map[[2]float64]float64{
					{0, 0}: 0,
					{0, 1}: 1,
				},
			},
			want: [][]float64{
				{0, 1},
			},
		},
		{
			name: "Non-empty Joint PMF",
			fields: fields{
				values: map[[2]float64]float64{
					{0, 0}: 0.1,
					{0, 1}: 0.2,
					{1, 0}: 0.3,
					{1, 1}: 0.4,
				},
			},
			want: [][]float64{
				{0, 0.3},
				{1, 0.7},
			},
		},
		{
			name: "Multiple Values with Marginal X",
			fields: fields{
				values: map[[2]float64]float64{
					{0, 0}: 0.1,
					{0, 1}: 0.2,
					{1, 0}: 0.2,
					{1, 1}: 0.4,
					{2, 0}: 0.1,
				},
			},
			want: [][]float64{
				{0, 0.3},
				{1, 0.6},
				{2, 0.1},
			},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			j := &JointPMF{
				values: make(map[[2]float64]float64),
			}
			for k, v := range tt.fields.values {
				j.Set(k[0], k[1], v)
			}

			// Test marginal distribution for specific values
			if tt.name == "Empty Joint PMF" {
				// Check marginal distribution for all x
				if len(j.GetMarginalX().orderedValues) != len(tt.want) {
					t.Errorf("Expected %d marginal values for X, got %d", len(tt.want), len(j.GetMarginalX().orderedValues))
				}
				for i := range j.GetMarginalX().orderedValues {
					expected := tt.want[i][1]
					actual := j.GetMarginalX().Get(j.GetMarginalX().orderedValues[i])
					if math.Abs(expected-actual) > 0.01 {
						t.Errorf("Expected marginal value for X at %f to be %f, got %f", j.GetMarginalX().orderedValues[i], expected, actual)
					}
				}
			}
		})
	}
}

func TestJointPMF_GetMarginalY(t *testing.T) {
	type fields struct {
		values map[[2]float64]float64
	}
	tests := []struct {
		name   string
		fields fields
		want   [][]float64
	}{
		{
			name: "Single Value has 1 marginal",
			fields: fields{
				values: map[[2]float64]float64{
					{0, 0}: 0,
					{0, 1}: 1,
				},
			},
			want: [][]float64{
				{1, 1},
			},
		},
		{
			name: "Non-empty Joint PMF",
			fields: fields{
				values: map[[2]float64]float64{
					{0, 0}: 0.1,
					{0, 1}: 0.2,
					{1, 0}: 0.3,
					{1, 1}: 0.4,
				},
			},
			want: [][]float64{
				{0, 0.4},
				{1, 0.6},
			},
		},
		{
			name: "Multiple Values with Marginal X",
			fields: fields{
				values: map[[2]float64]float64{
					{0, 0}: 0.1,
					{0, 1}: 0.2,
					{1, 0}: 0.2,
					{1, 1}: 0.4,
					{2, 0}: 0.1,
				},
			},
			want: [][]float64{
				{0, 0.4},
				{1, 0.6},
			},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			j := &JointPMF{
				values: make(map[[2]float64]float64),
			}
			for k, v := range tt.fields.values {
				j.Set(k[0], k[1], v)
			}

			// Test marginal distribution for specific values
			if tt.name == "Empty Joint PMF" {
				// Check marginal distribution for all x
				if len(j.GetMarginalY().orderedValues) != len(tt.want) {
					t.Errorf("Expected %d marginal values for X, got %d", len(tt.want), len(j.GetMarginalY().orderedValues))
				}
				for i := range j.GetMarginalY().orderedValues {
					expected := tt.want[i][1]
					actual := j.GetMarginalY().Get(j.GetMarginalY().orderedValues[i])
					if math.Abs(expected-actual) > 0.01 {
						t.Errorf("Expected marginal value for X at %f to be %f, got %f", j.GetMarginalY().orderedValues[i], expected, actual)
					}
				}
			}
		})
	}
}

func TestFindLocalMinimum(t *testing.T) {
	type args struct {
		fn    func(float64) float64
		start float64
		step  float64
	}
	tests := []struct {
		name string
		args args
		want float64
	}{
		{
			name: "x2",
			args: args{
				fn: func(x float64) float64 {
					return x * x
				},
				start: -1000,
				step:  0.1,
			},
			want: 0,
		},
		{
			name: "x2 + 1",
			args: args{
				fn: func(x float64) float64 {
					return x*x + 1
				},
				start: 1000,
				step:  0.1,
			},
			want: 0,
		},
		{
			name: "x2 + x",
			args: args{
				fn: func(x float64) float64 {
					return x*x + x
				},
				start: 10000,
				step:  0.1,
			},
			want: -0.5,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := FindLocalMinimum(tt.args.fn, tt.args.start, tt.args.step); got != tt.want {
				t.Errorf("FindLocalMinimum() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestGetRightTailZScore(t *testing.T) {
	type args struct {
		confidenceLevel float64
	}
	tests := []struct {
		name string
		args args
		want float64
	}{
		{
			name: "90% confidence level",
			args: args{
				confidenceLevel: 0.90,
			},
			want: 1.2816, // Z-score for 90% confidence level
		},
		{
			name: "95% confidence level",
			args: args{
				confidenceLevel: 0.95,
			},
			want: 1.6449, // Z-score for 95% confidence level
		},
		{
			name: "99% confidence level",
			args: args{
				confidenceLevel: 0.99,
			},
			want: 2.3263, // Z-score for 99% confidence level
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := GetRightTailZScoreFromProbability(tt.args.confidenceLevel, -100, 100, 0.001); math.Abs(got-tt.want) > 0.01 {
				t.Errorf("GetRightTailZScore() = %v, want %v", got, tt.want)
			}
		})
	}
}

func Test421(t *testing.T) {
	// 4.2.1. Let the observed value of the mean X and of the sample variance of a random sample of size 20 from a distribution that is N(M, 02) be 81.2 and 26.5, respectively.
	// Find respectively 90%, 95% and 99% confidence intervals for M. Note how the lengths of the confidence intervals increase as the confidence increases.
	mean := 81.2
	variance := 26.5
	sampleSize := 20
	stdDev := math.Sqrt(variance)
	// Calculate the margin of error for each confidence level
	confidenceLevels := []float64{0.90, 0.95, 0.99}
	step := 0.001
	from := -10.0
	to := 10.0
	expected := [][]float64{
		{79.21, 83.19}, // Expected lower and upper bounds for 90% confidence level
		{78.79, 83.61}, // Expected lower and upper bounds for 95% confidence level
		{77.90, 84.50}, // Expected lower and upper bounds for 99% confidence level
	}
	for i, confLevel := range confidenceLevels {
		t.Run(fmt.Sprintf("Confidence level %.2f", confLevel), func(t *testing.T) {
			lower, upper := GetMeanConfidenceIntervalForNormalDistribution(
				mean,
				stdDev,
				sampleSize,
				confLevel,
				from,
				to,
				step,
			)

			if math.Abs(lower-expected[i][0]) > 0.01 || math.Abs(upper-expected[i][1]) > 0.01 {
				t.Errorf("Confidence level %.2f: got [%.2f, %.2f], want [%.2f, %.2f]",
					confLevel, lower, upper, expected[i][0], expected[i][1])
			}
		})
	}
}

func TestGetTScoreFromProbability(t *testing.T) {
	type args struct {
		confidenceLevel  float64
		degreesOfFreedom float64
	}
	tests := []struct {
		name string
		args args
		want float64
	}{
		{
			name: "Degrees of Freedom 1, Confidence Level 0.9",
			args: args{
				confidenceLevel:  0.9,
				degreesOfFreedom: 1,
			},
			want: 3.0787, // T-score for 90% confidence level with 1 degree of freedom
		},
		{
			name: "Degrees of Freedom 5, Confidence Level 0.9",
			args: args{
				confidenceLevel:  0.9,
				degreesOfFreedom: 5,
			},
			want: 1.4759, // T-score for 90% confidence level with 5 degrees of freedom
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := GetRightTailTScoreFromProbability(tt.args.confidenceLevel, tt.args.degreesOfFreedom, -1000, 1000, 0.001); math.Abs(got-tt.want) > 0.01 {
				t.Errorf("GetRightTailTScoreFromProbability() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestSolve427(t *testing.T) {
	// 4.2.7. Let a random sample of size 17 from the normal distribution N(u, o2) yield x = 4.7 and s2 = 5.76. Determine a 90% confidence interval for average.
	sample_mean := 4.7
	variance := 5.76
	sampleSize := 17
	stdDev := math.Sqrt(variance)
	confidenceLevel := 0.90

	lower, upper := GetMeanConfidenceIntervalForNormalDistribution(
		sample_mean,
		stdDev,
		sampleSize,
		confidenceLevel,
		-100.0,
		100.0,
		0.001,
	)

	if math.Abs(lower-3.68) > 0.01 || math.Abs(upper-5.72) > 0.01 {
		t.Errorf("Confidence level %.2f: got [%.2f, %.2f], want [%.2f, %.2f]",
			confidenceLevel, lower, upper, 3.68, 5.72)
	}
}

func TestSolve4220(t *testing.T) {
	// 4.2.20. When 100 tacks were thrown on a table, 60 of them landed
	// point up. Obtain a 95% confidence interval for the probability that a
	//  tack of this type lands point up. Assume independence.

	numTrials := 100.0                          // number of trials
	numSuccess := 60.0                          // number of successes
	proportionSuccess := numSuccess / numTrials // sample proportion
	confidenceLevel := 0.95
	lower, upper := GetProportionConfidenceInterval(proportionSuccess, numTrials, confidenceLevel, -100.0, 100.0, 0.001)
	if math.Abs(lower-0.504) > 0.01 || math.Abs(upper-0.696) > 0.01 {
		t.Errorf("Confidence level %f : got [ %f, %f ], want [ %f, %f ]",
			confidenceLevel, lower, upper, 0.504, 0.696)
	}
}
func TestSolve468(t *testing.T) {
	// a.

	// h0 p= 0.14
	// h1 p > 0.14
	p := 0.14
	// pSample := 0.18 //104.0 / 590.0
	pSample := 108 / 590.0 // Sample proportion from the problem statement
	t.Log("Sample proportion p:", pSample)
	// b. define critical region with alpha = 0.01
	alpha := 0.01
	zscore := GetRightTailZScoreFromProbability(1-alpha, -1000, 1000, 0.001)
	t.Log("Critical Z-score for alpha = 0.01:", zscore)

	// c. // Calculate the Z-score for the sample proportion
	z := (pSample - p) / (math.Sqrt(pSample * (1 - pSample) / 590.0))
	t.Log("Z-score for p = 0.18:", z)

	if z > zscore {
		t.Log("Reject H0: p > 0.14")
	} else {
		t.Log("Fail to reject H0: p <= 0.14")
	}

	// d. Calculate the p-value
	pValue := 1 - Integrate(-1000, z, 0.001, GetNormalDistributionFunction(0, 1))
	t.Log("P-value:", pValue)

	if pValue < alpha {
		t.Log("Reject H0 because p-value < alpha")
	} else {
		t.Log("Fail to reject H0: p <= 0.14")
	}
	t.Log("=== Solve 4.6.8 ===")
}

func TestSampleExercise(t *testing.T) {
	n := 10.0
	meanEstimated := 7.1
	stdDev := 0.12
	alpha := 0.1

	// h0 : mean = 7
	// h1 : mean != 7

	tStatistic := GetStudentTStatistic(meanEstimated, 7, stdDev, int(n))
	t.Logf("T-statistic: %.4f", tStatistic)

	halphAlpha := alpha / 2
	criticalT := GetTScore(n, 1-halphAlpha, -1000, 1000, 0.001)
	t.Log("Critical T-value for alpha/2 =", halphAlpha, ":", criticalT)

	if tStatistic > criticalT || tStatistic < -criticalT {
		t.Log("Reject H0: mean != 7")
	} else {
		t.Error("Failed to reject H0: mean = 7")
	}

	// now lets increase the sample size to 30
	n = 36.0
	zScore := GetZScore(meanEstimated, 7, stdDev, int(n))
	t.Logf("Z-score: %.4f", zScore)

	criticalZ := GetRightTailZScoreFromProbability(halphAlpha, -1000, 1000, 0.001)
	t.Log("Critical Z-value for alpha/2 =", halphAlpha, ":", criticalZ)

	if zScore > criticalZ {
		t.Log("Reject H0: mean != 7")
	} else {
		t.Error("Failed to reject H0: mean = 7")
	}
}

func TestSolve466(t *testing.T) {
	// 300 mg 284 279 289 292 287 295 285 279 306 298
	// 600 mg 298 307 297 279 291 335 299 300 306 291
	arr300 := []float64{284, 279, 289, 292, 287, 295, 285, 279, 306, 298}
	arr600 := []float64{298, 307, 297, 279, 291, 335, 299, 300, 306, 291}

	n := float64(len(arr300) + len(arr600))
	mean300 := average(arr300)
	mean600 := average(arr600)
	stdDev300 := stdev(arr300)
	stdDev600 := stdev(arr600)

	// h0 := "mean300 = mean600"
	// h1 := "mean300 != mean600"
	t.Logf("Sample mean for 300 mg: %.2f", mean300)
	t.Logf("Sample mean for 600 mg: %.2f", mean600)
	t.Logf("Sample standard deviation for 300 mg: %.2f", stdDev300)
	t.Logf("Sample standard deviation for 600 mg: %.2f", stdDev600)

	stdErr := math.Sqrt(stdDev600 * stdDev300)
	tStatistic := GetStudentTStatistic(mean300-mean600, 0, stdErr, int(n))
	t.Logf("T-statistic: %.4f", tStatistic)
	// Get the critical t-value for a two-tailed test with alpha = 0.05
	alpha := 0.05
	halphAlpha := alpha / 2
	criticalT := GetTScore(n-2, 1-halphAlpha, -1000, 1000, 0.001)
	t.Logf("Critical T-value for alpha/2 = %.2f: %.4f", halphAlpha, criticalT)

	if tStatistic > criticalT || tStatistic < -criticalT {
		t.Log("Reject H0: mean300 != mean600")
	} else {
		t.Error("Failed to reject H0: mean300 = mean600")
	}
}

func TestGetStudentTStatistic(t *testing.T) {
	t.Log("=== Get Student's T Statistic ===")
	// Example parameters for Student's T distribution
	sampleMean := 4.7
	populationMean := 5.0
	sampleStandardDeviation := 1.2
	sampleSize := 15

	// Get the t-statistic for the given degrees of freedom and confidence level
	tStatistic := GetStudentTStatistic(sampleMean, populationMean, sampleStandardDeviation, sampleSize)
	t.Logf("T-statistic: %.4f", tStatistic)
}

func sum(arr []float64) float64 {
	sum := 0.0
	for _, v := range arr {
		sum += v
	}
	return sum
}

func average(arr []float64) float64 {
	if len(arr) == 0 {
		return 0
	}
	return sum(arr) / float64(len(arr))
}

func stdev(arr []float64) float64 {
	if len(arr) == 0 {
		return 0
	}
	mean := average(arr)
	variance := 0.0
	for _, v := range arr {
		variance += (v - mean) * (v - mean)
	}
	variance /= float64(len(arr) - 1) // Sample variance
	return math.Sqrt(variance)
}

func TestGetTScore(t *testing.T) {
	type args struct {
		degreesOfFreedom float64
		confidenceLevel  float64
		from             float64
		to               float64
		step             float64
	}
	tests := []struct {
		name string
		args args
		want float64
	}{
		{
			name: "Degrees of Freedom 1, Confidence Level 0.9",
			args: args{
				degreesOfFreedom: 1,
				confidenceLevel:  0.9,
				from:             -1000,
				to:               1000,
				step:             0.001,
			},
			want: 3.0787, // T-score for 90% confidence level with 1 degree of freedom
		},
		{
			name: "Degrees of Freedom 5, Confidence Level 0.9",
			args: args{
				degreesOfFreedom: 5,
				confidenceLevel:  0.9,
				from:             -1000,
				to:               1000,
				step:             0.001,
			},
			want: 1.4759, // T-score for 90% confidence level with 5 degrees of freedom
		},
		{
			name: "Degrees of Freedom 9, Confidence Level 0.95",
			args: args{
				degreesOfFreedom: 9,
				confidenceLevel:  0.95,
				from:             -1000,
				to:               1000,
				step:             0.001,
			},
			want: 1.833, // T-score for 95% confidence level with 9 degrees of freedom
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := GetTScore(tt.args.degreesOfFreedom, tt.args.confidenceLevel, tt.args.from, tt.args.to, tt.args.step); math.Abs(got-tt.want) > 0.01 {
				t.Errorf("GetTScore() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestGetZScore(t *testing.T) {
	type args struct {
		sampleMean              float64
		populationMean          float64
		sampleStandardDeviation float64
		sampleSize              int
	}
	tests := []struct {
		name string
		args args
		want float64
	}{
		{
			name: "Z-score for sample mean greater than population mean",
			args: args{
				sampleMean:              105.0,
				populationMean:          100.0,
				sampleStandardDeviation: 15.0,
				sampleSize:              30,
			},
			want: 1.8257, // Z-score = (105 - 100)
		},
		{
			name: "Z-score for sample mean less than population mean",
			args: args{
				sampleMean:              95.0,
				populationMean:          100.0,
				sampleStandardDeviation: 15.0,
				sampleSize:              30,
			},
			want: -1.8257, // Z-score = (95 - 100)
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := GetZScore(tt.args.sampleMean, tt.args.populationMean, tt.args.sampleStandardDeviation, tt.args.sampleSize); math.Abs(got-tt.want) > 0.01 {
				t.Errorf("GetZScore() = %v, want %v", got, tt.want)
			}
		})
	}
}

func Test_stdev(t *testing.T) {
	type args struct {
		arr []float64
	}
	tests := []struct {
		name string
		args args
		want float64
	}{
		{
			name: "Standard deviation of positive numbers",
			args: args{arr: []float64{1, 2, 3, 4, 5}},
			want: 1.5811, // Standard deviation of {1, 2, 3, 4, 5} is approximately 1.581
		},
		{
			name: "Standard deviation of negative numbers",
			args: args{arr: []float64{-1, -2, -3, -4, -5}},
			want: 1.5811, // Standard deviation of {-1, -2, -3, -4, -5} is approximately 1.581
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := stdev(tt.args.arr); math.Abs(got-tt.want) > 0.01 {
				t.Errorf("stdev() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestSolve465(t *testing.T) {
	// 4.6.5. A random sample of size 100 from a normal distribution N(0, 1) yields a sample mean of 0.2 and a sample standard deviation of 1.5.
	// Find a 95% confidence interval for the population mean.
	self := []float64{16.20, 16.78, 17.38, 17.59, 17.37, 17.49, 18.18, 18.16, 18.36, 18.53, 15.92, 16.58, 17.57, 16.75, 17.28, 17.32, 17.51, 17.58, 18.26, 17.87}
	rival := []float64{15.95, 16.15, 17.05, 16.99, 17.34, 17.53, 17.34, 17.51, 18.10, 18.19, 16.04, 16.80, 17.24, 16.81, 17.11, 17.22, 17.33, 17.82, 18.19, 17.88}

	fmt.Printf("Self average: %.4f\n", average(self))
	fmt.Printf("Rival average: %.4f\n", average(rival))

	diff := make([]float64, len(self))
	for i := range self {
		diff[i] = rival[i] - self[i]
	}
	t.Logf("Differences: %v", diff)
	fmt.Printf("Average of differences: %.4f\n", average(diff))
	stdDevDiff := stdev(diff)
	fmt.Printf("Standard deviation of differences: %.4f\n", stdDevDiff)
	fmt.Println("Sample size:", len(diff))

	tStatistic := GetStudentTStatistic(average(diff), 0, stdev(diff), len(diff))
	fmt.Printf("T-statistic: %.4f\n", tStatistic)

	// get p-value
	pValue := Integrate(-1000, tStatistic, 0.001, GetStudentTDistributionFunction(float64(len(diff)-1)))
	fmt.Printf("P-value: %.4f\n", pValue)

	// obtain a point estimate with a 95% confidence interval
	confidenceLevel := 0.95
	lower, upper := GetMeanConfidenceIntervalForNormalDistribution(
		average(diff),
		stdDevDiff,
		len(diff),
		confidenceLevel,
		-100.0,
		100.0,
		0.001,
	)
	fmt.Printf("95%% Confidence Interval for the mean difference: [%.4f, %.4f]\n", lower, upper)
}

func TestExample1(t *testing.T) {
	//  A coin-operated soft-drink machine was designed to discharge on the average 7 ounces
	// of beverage per cup. In a test of the machine, ten cupfuls of beverage were drawn from
	// the machine and measured. The mean and standard deviation of the ten measurements
	// were 7.1 ounces and .12 ounce, respectively. Do these data present suﬀicient evidence to
	// indicate that the mean discharge differs from 7 ounces? Use 𝛼 = 0.1

	mean := 7.1
	populationMean := 7.0
	stdDev := 0.12
	sampleSize := 10
	alpha := 0.1
	tStatistic := GetStudentTStatistic(mean, populationMean, stdDev, sampleSize)
	t.Logf("T-statistic: %.4f", tStatistic)
	halphAlpha := alpha / 2
	criticalT := GetTScore(float64(sampleSize-1), 1-halphAlpha, -1000, 1000, 0.001)
	t.Logf("Critical T-value for alpha/2 = %.2f: %.4f", halphAlpha, criticalT)
	if tStatistic > criticalT || tStatistic < -criticalT {
		t.Log("Reject H0: mean != 7")
	} else {
		t.Error("Failed to reject H0: mean = 7")
	}
	// now lets increase the sample size to 30
	sampleSize = 36
	zScore := GetZScore(mean, populationMean, stdDev, sampleSize)
	t.Logf("Z-score: %.4f", zScore)
	criticalZ := GetRightTailZScoreFromProbability(halphAlpha, -1000, 1000, 0.001)
	t.Logf("Critical Z-value for alpha/2 = %.2f: %.4f", halphAlpha, criticalZ)
	if zScore > criticalZ {
		t.Log("Reject H0: mean != 7")
	} else {
		t.Error("Failed to reject H0: mean = 7")
	}
	t.Log("=== Example 1 ===")
}

func TestGetCovariance(t *testing.T) {
	type args struct {
		x []float64
		y []float64
	}
	tests := []struct {
		name string
		args args
		want float64
	}{
		{"case 1", args{[]float64{1, 2, 3}, []float64{4, 5, 6}}, 1.0},
		{"case 2", args{[]float64{1, 2, 3}, []float64{1, 2, 3}}, 1.0},
		{"case 3", args{[]float64{1, 2, 3}, []float64{7, 8, 9}}, 1.0},
		{"case 4", args{[]float64{1, 2, 3}, []float64{10, 11, 12}}, 1.0},
		{"case 5", args{[]float64{1, 2, 3}, []float64{2, 4, 10}}, 4.0},
		{"case 6", args{[]float64{1, 2, 3}, []float64{2, 7, 12}}, 5.0},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := GetCovariance(tt.args.x, tt.args.y); got != tt.want {
				t.Errorf("GetCovariance() = %f, want %f", got, tt.want)
			}
		})
	}
}

func TestGetVariance(t *testing.T) {
	type args struct {
		x []float64
	}
	tests := []struct {
		name string
		args args
		want float64
	}{
		{"case 1", args{[]float64{1, 2, 3}}, 1.0},
		{"case 2", args{[]float64{1, 2, 3, 4}}, 1.67},
		{"case 3", args{[]float64{1, 1, 1}}, 0.0},
		{"case 4", args{[]float64{1, 2, 3, 4, 5}}, 3.0},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := GetVariance(tt.args.x); got-tt.want > 1e-9 {
				t.Errorf("GetVariance() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestGetCorrelation(t *testing.T) {
	type args struct {
		x []float64
		y []float64
	}
	tests := []struct {
		name string
		args args
		want float64
	}{
		{"case 1", args{[]float64{1, 2, 3}, []float64{4, 5, 6}}, 1.0},
		{"case 2", args{[]float64{1, 2, 3}, []float64{1, 2, 3}}, 1.0},
		{"case 3", args{[]float64{1, 2, 3}, []float64{7, 8, 9}}, 1.0},
		{"case 4", args{[]float64{1, 2, 3}, []float64{10, 11, 12}}, 1.0},
		{"case 5", args{[]float64{1, 2, 3}, []float64{2, 4, 10}}, 0.9607689228},
		{"case 6", args{[]float64{1, 2, 3}, []float64{2, 7, 12}}, 5.0},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := GetCorrelation(tt.args.x, tt.args.y)
			diff := got - tt.want
			threshold := 1e-9
			if diff > threshold {
				t.Errorf("GetCorrelation() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestGetProbabilityFromZScore(t *testing.T) {
	tests := []struct {
		name string // description of this test case
		// Named input parameters for target function.
		zScore float64
		want   float64
	}{
		{name: "case 1", zScore: 0, want: 0.5},
		{name: "case 2", zScore: 1.96, want: 0.975},
		{name: "case 3", zScore: -1, want: 0.158},
		{name: "case 4", zScore: 2, want: 0.977},
		{name: "case 5", zScore: -2, want: 0.022},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := GetProbabilityFromZScore(tt.zScore)
			if math.Abs(got-tt.want) > 1e-3 {
				t.Errorf("GetProbabilityFromZScore() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestGetPValueFromZScore(t *testing.T) {
	tests := []struct {
		name string // description of this test case
		// Named input parameters for target function.
		zScore   float64
		testType HypothesisTest
		want     float64
	}{
		{
			name:     "case 1",
			zScore:   1.65,
			testType: RightTailed,
			want:     0.0495,
		},
		{
			name:     "case 2",
			zScore:   1.65,
			testType: LeftTailed,
			want:     0.9505,
		},
		{
			name:     "case 3",
			zScore:   1.65,
			testType: TwoTailed,
			want:     0.099,
		},
		{
			name:     "case 4",
			zScore:   0,
			testType: RightTailed,
			want:     0.5,
		},
		{
			name:     "case 5",
			zScore:   0,
			testType: TwoTailed,
			want:     1.0,
		},
		{
			name:     "case 6",
			zScore:   2.33,
			testType: RightTailed,
			want:     0.0099,
		},
		{
			name:     "case 7",
			zScore:   2.33,
			testType: TwoTailed,
			want:     0.0198,
		},
		{
			name:     "case 8",
			zScore:   -1.96,
			testType: LeftTailed,
			want:     0.025,
		},
		{
			name:     "case 9",
			zScore:   -1.96,
			testType: TwoTailed,
			want:     0.05,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := GetPValueFromZScore(tt.zScore, tt.testType)
			if math.Abs(got-tt.want) > 1e-3 {
				t.Errorf("GetPValueFromZScore() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestPredictionInterval(t *testing.T) {
	// 	•	Dataset (n = 8):
	// •	x: [10, 12, 15, 20, 25, 30, 35, 40]
	// •	y: [17.2, 19.1, 22.9, 30.5, 38.2, 43.0, 52.3, 58.7]
	// •	Target point: x_0 = 32
	x := []float64{10, 12, 15, 20, 25, 30, 35, 40}
	y := []float64{17.2, 19.1, 22.9, 30.5, 38.2, 43.0, 52.3, 58.7}
	// x0 := 32.0

	// Calculate confidence interval
	stdevY := math.Sqrt(GetVariance(y))
	t.Log("Standard Deviation Y: ", stdevY)
	meanY := pkg.GetMean(y)
	t.Log("Mean Y: ", meanY)
	lowerY, upperY := GetMeanConfidenceIntervalForNormalDistribution(
		meanY,
		stdevY,
		len(y),
		0.95,
		-100,
		100,
		0.01,
	)
	t.Log("Confidence Interval Y: [", lowerY, ", ", upperY, "]")

	meanX := pkg.GetMean(x)
	t.Log("Mean X: ", meanX)
	stdevX := math.Sqrt(GetVariance(x))
	t.Log("Standard Deviation X: ", stdevX)
	lowerX, upperX := GetMeanConfidenceIntervalForNormalDistribution(
		meanX,
		stdevX,
		len(x),
		0.95,
		-100,
		100,
		0.01,
	)
	t.Log("Confidence Interval X: [", lowerX, ", ", upperX, "]")
}

func TestGetKFold(t *testing.T) {
	tests := []struct {
		name string // description of this test case
		// Named input parameters for target function.
		observations [][]float64
		k            int
		want         [][][]float64
		wantErr      error
	}{
		{
			name: "case 1",
			observations: [][]float64{
				{1, 2},
				{3, 4},
				{5, 6},
				{7, 8},
				{9, 10},
			},
			k: 2,
			want: [][][]float64{
				{
					{1, 2},
					{5, 6},
					{9, 10},
				},
				{
					{3, 4},
					{7, 8},
				},
			},
		},
		{
			name: "case 2 - k greater than 1",
			observations: [][]float64{
				{1, 2},
			},
			k:       1,
			want:    nil,
			wantErr: fmt.Errorf("k must be greater than 1"),
		},
		{
			name: "case 3 - error k equal to 1",
			observations: [][]float64{
				{1, 2},
				{1, 2},
				{1, 2},
			},
			k:       4,
			want:    nil,
			wantErr: fmt.Errorf("Number of observations must be greater than k"),
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := GetKFold(tt.observations, tt.k)
			if err != nil && err.Error() != tt.wantErr.Error() {
				t.Errorf("GetKFold() error = %v", err)
				return
			}
			if !reflect.DeepEqual(got, tt.want) {
				t.Errorf("GetKFold() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestTTestExample(t *testing.T) {
	src := rand.NewPCG(0, 0)
	generator := Generator{
		Random: rand.New(src),
	}
	totalnums := generator.GenerateNormalSamples(10, 1, 1000000)
	t.Log("=== T-Test Example ===")
	g1 := Generator{
		Random: rand.New(rand.NewPCG(42, 99)),
	}
	g2 := Generator{
		Random: rand.New(rand.NewPCG(99, 1)),
	}
	sample1 := g1.GetRandomSample(totalnums, 1000)
	sample2 := g2.GetRandomSample(totalnums, 1000)
	// check sum is different
	if sum(sample1) == sum(sample2) || pkg.GetMean(sample1) == pkg.GetMean(sample2) {
		t.Error("Samples are equal, which is unlikely")
	}

	t.Logf("Sample 1 Mean: %.4f, StdDev: %.4f", pkg.GetMean(sample1), math.Sqrt(GetVariance(sample1)))
	t.Logf("Sample 2 Mean: %.4f, StdDev: %.4f", pkg.GetMean(sample2), math.Sqrt(GetVariance(sample2)))

	tStatistic := GetStudentTStatistic(
		pkg.GetMean(sample1)-pkg.GetMean(sample2),
		0,
		math.Sqrt(GetVariance(sample1)+GetVariance(sample2)),
		len(sample1)+len(sample2)-2,
	)
	t.Logf("T-statistic: %.4f", tStatistic)

	// get p-value
	degreesFreedom := len(sample1) + len(sample2) - 2
	tdistFunction := GetStudentTPDF(float64(degreesFreedom))
	pValue := Integrate(-1000, tStatistic+0.00001, 0.001, tdistFunction)

	t.Logf("P-value: %.4f", pValue)

	if pValue > 0.05 {
		t.Log("Do not reject H0: means are equal")
	} else {
		t.Error("Unexpected reject H0: means are not equal")
	}
}
