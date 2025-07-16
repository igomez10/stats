package stats

import (
	"fmt"
	"math"
	"strconv"
	"testing"

	"github.com/NimbleMarkets/ntcharts/barchart"
	"github.com/charmbracelet/lipgloss"
)

// PMF represents a Probability Mass Function
type PMF struct {
	values map[int]float64
}

// NewPMF creates a new PMF
func NewPMF() *PMF {
	return &PMF{
		values: make(map[int]float64),
	}
}

// GetSpace (also called range) returns the set of values with non-zero probability
func (p *PMF) GetSpace() []int {
	var space []int
	for value := range p.values {
		if p.values[value] > 0 {
			space = append(space, value)
		}
	}
	return space
}

// Set sets the probability for a given value
func (p *PMF) Set(value int, prob float64) {
	if prob < 0 || prob > 1 {
		panic("Probability must be between 0 and 1")
	}
	p.values[value] = prob
}

// Get returns the probability for a given value
func (p *PMF) Get(value int) float64 {
	return p.values[value]
}

// Normalize ensures all probabilities sum to 1
func (p *PMF) Normalize() {
	total := p.TotalSum()
	if total == 0 {
		return
	}
	for value := range p.values {
		p.values[value] /= total
	}
}

// TotalSum returns the sum of all probabilities
func (p *PMF) TotalSum() float64 {
	var total float64
	for _, prob := range p.values {
		total += prob
	}
	return total
}

// Mean calculates the expected value (mean) of the PMF
func (p *PMF) Mean() float64 {
	var mean float64
	for value, prob := range p.values {
		mean += float64(value) * prob
	}
	return mean
}

// Variance calculates the variance of the PMF
func (p *PMF) Variance() float64 {
	mean := p.Mean()
	var variance float64
	for value, prob := range p.values {
		diff := float64(value) - mean
		variance += diff * diff * prob
	}
	return variance
}

// StdDev calculates the standard deviation
func (p *PMF) StdDev() float64 {
	return math.Sqrt(p.Variance())
}

// Values returns all values with non-zero probability
func (p *PMF) Values() []int {
	var values []int
	for value := range p.values {
		values = append(values, value)
	}
	return values
}

// Print displays the PMF in a readable format
func (p *PMF) Print() {
	fmt.Println("PMF:")
	for value, prob := range p.values {
		fmt.Printf("  P(X=%d) = %.4f\n", value, prob)
	}
	fmt.Printf("Total: %.4f\n", p.TotalSum())
	fmt.Printf("Mean: %.4f\n", p.Mean())
	fmt.Printf("Std Dev: %.4f\n", p.StdDev())
}

// CreateBinomialPMF creates a binomial PMF
func CreateBinomialPMF(numberOfTrials int, probSuccess float64) *PMF {
	pmf := NewPMF()

	for i := 0; i <= numberOfTrials; i++ {
		// Calculate binomial coefficient C(n, k)
		coeff := binomialCoeff(numberOfTrials, i)
		// Calculate probability: C(n,k) * p^k * (1-p)^(n-k)
		prob := float64(coeff) * math.Pow(probSuccess, float64(i)) * math.Pow(1-probSuccess, float64(numberOfTrials-i))
		pmf.Set(i, prob)
	}

	return pmf
}

// binomialCoeff calculates binomial coefficient C(n, k)
// the binomial coefficient is the number of ways to choose k successes in n trials
func binomialCoeff(n, k int) int {
	if k > n || k < 0 {
		return 0
	}
	if k == 0 || k == n {
		return 1
	}

	// Use the property C(n,k) = C(n,n-k) to minimize calculations
	if k > n-k {
		k = n - k
	}

	result := 1
	for i := 0; i < k; i++ {
		result = result * (n - i) / (i + 1)
	}
	return result
}

// CreatePoissonPMF creates a Poisson PMF
func CreatePoissonPMF(lambda float64, maxK int) *PMF {
	pmf := NewPMF()

	for k := 0; k <= maxK; k++ {
		// Calculate Poisson probability: e^(-λ) * λ^k / k!
		prob := math.Exp(-lambda) * math.Pow(lambda, float64(k)) / float64(factorial(k))
		pmf.Set(k, prob)
	}

	return pmf
}

// factorial calculates factorial of n
func factorial(n int) int {
	if n <= 1 {
		return 1
	}
	result := 1
	for i := 2; i <= n; i++ {
		result *= i
	}
	return result
}

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

	// d1 := barchart.BarData{
	// 	Label: "A",
	// 	Values: []barchart.BarValue{
	// 		{"Item2", 21.2, lipgloss.NewStyle().Foreground(lipgloss.Color("10"))}}, // green
	// }
	// d2 := barchart.BarData{
	// 	Label: "B",
	// 	Values: []barchart.BarValue{
	// 		{"Item1", 15.2, lipgloss.NewStyle().Foreground(lipgloss.Color("9"))}}, // red
	// }
	// d3 := barchart.BarData{
	// 	Label: "B",
	// 	Values: []barchart.BarValue{
	// 		{"Item1", 15.2, lipgloss.NewStyle().Foreground(lipgloss.Color("9"))}}, // red
	// }

	// bc := barchart.New(10, 10)
	// bc.PushAll([]barchart.BarData{d1, d2, d3})
	// bc.Draw()

	fmt.Println(bc.View())
}

func TestBinomialDistribution(t *testing.T) {
	fmt.Println("=== Binomial Distribution PMF ===")
	pmf := CreateBinomialPMF(10, 0.5)
	pmf.Print()
}
