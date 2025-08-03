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

func TestDerivate(t *testing.T) {
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
				if math.Abs(got-want) > 0.01 { //&& (got/want)-1 > 0.01 {
					t.Errorf("Derivate(%v) = %v, want %v, diff %v", x, got, want, math.Abs(got-want))
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
	// 4.2.20. When 100 tacks were thrown on a table, 60 of them landed point up. Obtain a 95% confidence interval for the probability that a tack of this type lands point up. Assume independence.

	n := 100.0 // number of trials
	x := 60.0  // number of successes
	p := x / n // sample proportion
	confidenceLevel := 0.95
	lower, upper := GetProportionConfidenceInterval(p, n, confidenceLevel, -100.0, 100.0, 0.001)
	if math.Abs(lower-0.504) > 0.01 || math.Abs(upper-0.696) > 0.01 {
		t.Errorf("Confidence level %f : got [ %f, %f ], want [ %f, %f ]",
			confidenceLevel, lower, upper, 0.504, 0.696)
	}
}
