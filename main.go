package stats

import (
	"fmt"
	"math"
)

// Combination calculates the number of combinations of n items taken r at a time
func Combination(totalItems, takenItems int) int {
	if takenItems > totalItems || takenItems < 0 {
		return 0
	}
	if takenItems == 0 || takenItems == totalItems {
		return 1
	}

	// Use the property C(n, r) = C(n, n-r) to minimize calculations
	if takenItems > totalItems-takenItems {
		takenItems = totalItems - takenItems
	}

	numerator := 1
	denominator := 1
	for i := 0; i < takenItems; i++ {
		numerator *= (totalItems - i)
		denominator *= (i + 1)
	}
	return numerator / denominator
}

// Factorial calculates Factorial of n
func Factorial(n float64) float64 {
	if n <= 1 {
		return 1
	}
	result := 1.0
	for i := 2.0; i <= n; i++ {
		result *= i
	}
	return result
}

// NewPoissonPMF creates a Poisson PMF
// Poisson is always PMF
// lambda is the average rate of occurrence
// numEvents is the maximum number of events to consider
// The PMF is defined for k = 0, 1, ..., numEvents
// The total number of outcomes is numEvents + 1 (from 0 to numEvents)
// The PMF is normalized so that the sum of probabilities equals 1
func NewPoissonPMF(lambda float64, numEvents int) *PMF {
	pmf := NewPMF()
	for currentNumEvents := 0; currentNumEvents <= numEvents; currentNumEvents++ {
		// Calculate Poisson probability: e^(-λ) * λ^k / k!
		prob := math.Exp(-lambda) * math.Pow(lambda, float64(currentNumEvents)) / float64(Factorial(float64(currentNumEvents)))
		pmf.Set(float64(currentNumEvents), prob)
	}

	return pmf
}

// NewBinomialPMF creates a binomial PMF
// A binomial PMF is defined by the number of trials (n) and the probability of success (p)
// It calculates the probability of getting k successes in n trials
// using the formula: P(X=k) = C(n, k) * p^k * (1-p)^(n-k)
// where C(n, k) is the binomial coefficient "n choose k"
// The PMF is defined for k = 0, 1, ..., n
// The total number of outcomes is n+1 (from 0 to n)
// The PMF is normalized so that the sum of probabilities equals 1
func NewBinomialPMF(numberOfTrials int, probSuccess float64) *PMF {
	pmf := NewPMF()
	for i := 0; i <= numberOfTrials; i++ {
		// Calculate binomial numCombinations C(n, k)
		numCombinations := Combination(numberOfTrials, i)
		// Calculate probability: C(n,k) * p^k * (1-p)^(n-k)
		prob := float64(numCombinations) * math.Pow(probSuccess, float64(i)) * math.Pow(1-probSuccess, float64(numberOfTrials-i))
		pmf.Set(float64(i), prob)
	}

	return pmf
}

// GetNormalDistributionFunction returns a function that represents the normal distribution
// The function is defined as:
// fx = (1 / (stdDev * math.Sqrt(2*math.Pi))) * exp(-0.5 * ((x - mean) / stdDev) ^ 2)
// where mean is the mean of the distribution and stdDev is the standard deviation
// This function can be used to create a PDF for a normal distribution
func GetNormalDistributionFunction(mean, stdDev float64) func(float64) float64 {
	return func(x float64) float64 {
		// fx = (1 / (stdDev * math.Sqrt(2*math.Pi))) * exp(-0.5 * ((x - mean) / stdDev) ^ 2)
		return (1 / (stdDev * math.Sqrt(2*math.Pi))) * math.Exp(-0.5*math.Pow((x-mean)/stdDev, 2))
	}
}

func GetStudentTDistributionFunction(degreesOfFreedom float64) func(float64) float64 {
	return func(x float64) float64 {
		// fx = (Gamma((degreesOfFreedom+1)/2) / (sqrt(degreesOfFreedom*math.Pi) * Gamma(degreesOfFreedom/2))) * (1 + ((x*x)/degreesOfFreedom))^(-(degreesOfFreedom+1)/2)
		return (math.Gamma((degreesOfFreedom+1)/2) / (math.Sqrt(degreesOfFreedom*math.Pi) * math.Gamma(degreesOfFreedom/2))) * math.Pow(1+((x*x)/degreesOfFreedom), -(degreesOfFreedom+1)/2)
	}
}

func NewNormalPDF(mean, stdDev, rangeMin, rangeMax float64) *PDF {
	if stdDev <= 0 {
		panic("Standard deviation must be positive")
	}
	if rangeMin >= rangeMax {
		panic("Invalid range: rangeMin must be less than rangeMax")
	}

	normalPDF := NewPDF(
		GetNormalDistributionFunction(mean, stdDev),
		rangeMin,
		rangeMax,
	)

	return normalPDF
}

func NewStudentTDistributionPDF(degreesOfFreedom, rangeMin, rangeMax float64) *PDF {
	if degreesOfFreedom <= 0 {
		panic("Degrees of freedom must be positive")
	}
	if rangeMin >= rangeMax {
		panic("Invalid range: rangeMin must be less than rangeMax")
	}

	tDistributionPDF := NewPDF(
		GetStudentTDistributionFunction(degreesOfFreedom),
		rangeMin,
		rangeMax,
	)

	return tDistributionPDF
}

// GetExponentialDistributionFunction returns a function that represents the exponential distribution
// The function is defined as:
// fx = lambda * exp(-lambda * x)
// where lambda is the rate parameter of the distribution
// This function can be used to create a PDF for an exponential distribution
// This is used to model the time between events in a Poisson process
// The exponential distribution is defined for x >= 0 only
func GetExponentialDistributionFunction(lambda float64) func(float64) float64 {
	return func(x float64) float64 {
		if x < 0 {
			return 0
		}
		return lambda * math.Exp(-lambda*x)
	}
}

// NewExponentialPDF creates an exponential PDF
// The exponential distribution is defined for x >= 0
// The PDF is defined as:
// fx = lambda * exp(-lambda * x)
// where lambda is the rate parameter of the distribution
// The PDF is normalized so that the integral from 0 to infinity equals 1
// The rangeMin and rangeMax define the limits of integration
// The PDF is defined for x in [rangeMin, rangeMax]
func NewExponentialPDF(lambda, rangeMin, rangeMax float64) *PDF {
	if lambda <= 0 {
		panic("Lambda must be positive")
	}
	if rangeMin >= rangeMax {
		panic("Invalid range: rangeMin must be less than rangeMax")
	}
	if rangeMin < 0 {
		panic("Exponential distribution is defined for x >= 0")
	}

	exponentialPDF := NewPDF(
		GetExponentialDistributionFunction(lambda),
		rangeMin,
		rangeMax,
	)

	return exponentialPDF
}

// Normalize normalizes a value x based on the mean and standard deviation
// It returns the z-score, which is the number of standard deviations away from the mean
// z = (x - mean) / stdDev
// This is useful for standardizing values in a normal distribution
// It transforms the value into a standard normal variable (mean = 0, stdDev = 1)
func Normalize(x float64, mean, stdDev float64) float64 {
	return (x - mean) / stdDev
}

func Integrate(from, to, step float64, fx func(float64) float64) float64 {
	var res float64 = 0
	for x := from; x < to; x += step {
		res += fx(x) * step
	}
	return res
}

// IntegrateUntilValue returns the limit of integration until the accumulated value reaches or exceeds targetValue
// THIS FUNCTION IS NOT A STANDARD INTEGRATION FUNCTION, IT DOES NOT RETURN THE AREA UNDER THE CURVE
// INSTEAD, IT RETURNS THE ACCUMULATED VALUE OF THE INTEGRAL UNTIL IT REACHES OR EXCEEDS targetValue
// To avoid infinite loops, it stops at maxValue
// It uses a step size to approximate the integral
// The function fx is the integrand function
// It returns the accumulated value of the integral until it reaches or exceeds targetValue
// If the integral does not reach targetValue before maxValue, it returns the accumulated value
// This is useful for numerical integration where you want to find the area under the curve until a certain value is reached
func IntegrateUntilValue(from, toMaxValue, targetValue float64, step float64, fx func(float64) float64) float64 {
	var res float64 = 0
	for x := from; x < toMaxValue; x += step {
		res += fx(x) * step
		if res >= targetValue {
			return x
		}
	}
	panic("Integral did not reach target value before max value")
}

// GetDerivativeAtX calculates the derivative of a function fx at a point x using the finite difference method
// It returns a function that takes a float64 x and returns the derivative at that point
// The derivative is calculated as:
// f'(x) = (f(x + h) - f(x - h)) / (2 * h)
// where h is the step size
// This is a numerical approximation of the derivative
func GetDerivativeAtX(x, step float64, fx func(float64) float64) float64 {
	return (fx(x+step) - fx(x-step)) / (2 * step)
}

// FindCriticalPoint finds the inflection point by finding the point where
func FindCriticalPoint(fx func(float64) float64, start, end, step float64) *float64 {
	firstDerivative := func(x float64) float64 {
		return GetDerivativeAtX(x, step, fx)
	}

	var lastValue float64
	for x := start; x <= end; x += step {
		firstDerivativeValue := firstDerivative(x)

		if x == start {
			lastValue = firstDerivativeValue
			continue
		}

		// Check if the sign of the first derivative has changed
		if lastValue < 0 && firstDerivativeValue > 0 {
			return &x // Found a critical point (minimum)
		}
		if lastValue > 0 && firstDerivativeValue < 0 {
			return &x // Found a critical point (maximum)
		}

		if math.Abs(firstDerivativeValue) < step {
			// If the first derivative is close to zero, we have found a critical point
			return &x
		}

	}

	return nil // No inflection point found in the given range
}

// GetMaximumLikelihoodEstimation calculates the maximum likelihood estimation (MLE) for a given dataset
// For a normal distribution, the MLE is the sample mean
// This function assumes the data is normally distributed
// It returns the mean of the data as the MLE
// If the data is empty, it panics
func GetMaximumLikelihoodEstimation(data []float64) float64 {
	if len(data) == 0 {
		panic("Data cannot be empty")
	}

	var sum float64
	for _, value := range data {
		sum += value
	}
	return sum / float64(len(data)) // Return the mean as the MLE for a normal distribution
}

func GetMin(data []float64) float64 {
	if len(data) == 0 {
		panic("Data cannot be empty")
	}

	min := data[0]
	for _, value := range data {
		if value < min {
			min = value
		}
	}
	return min
}

func GetMax(data []float64) float64 {
	if len(data) == 0 {
		panic("Data cannot be empty")
	}

	max := data[0]
	for _, value := range data {
		if value > max {
			max = value
		}
	}
	return max
}

// FindMaximumLikelihoodPoisson find the lambda parameter for a Poisson distribution
// given a dataset, it finds the value of lambda that maximizes the log-likelihood function
func FindMaximumLikelihoodPoisson(data []float64) float64 {
	start := GetMin(data)
	end := GetMax(data)
	logLikelihoodFn := GetLogLikelihoodFunctionPoisson(data)
	// Find the critical point of the log-likelihood function (where derivative is 0)
	criticalPoint := FindCriticalPoint(logLikelihoodFn, start, end, 0.001)
	if criticalPoint == nil {
		fmt.Println("No critical point found")
		return 0.0
	}
	return *criticalPoint
}

func GetLogLikelihoodFunctionPoisson(data []float64) func(float64) float64 {
	return func(lambda float64) float64 {
		logLikelihood := 0.0
		for _, x := range data {
			logLikelihood += x*math.Log(lambda) - lambda - math.Log(Factorial(x))
		}
		return logLikelihood
	}
}
