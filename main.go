package stats

import (
	"math"
	"stats/pkg"
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

// Exp returns e**x, the base-e exponential of x.
// Special cases are:
// - If x is NaN, the result is NaN.
// - If x is +Inf, the result is +Inf.
// - If x is -Inf, the result is 0.
func GetPoissonDistributionFunction(lambda float64) func(float64) float64 {
	return func(k float64) float64 {
		if k < 0 {
			return 0
		}
		// Poisson PMF: P(X=k) = (e^(-λ) * λ^k) / k!
		return math.Exp(-lambda) * math.Pow(lambda, k) / Factorial(k)
	}
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
		// prob := math.Exp(-lambda) * math.Pow(lambda, float64(currentNumEvents)) / float64(Factorial(float64(currentNumEvents)))
		prob := GetPoissonDistributionFunction(lambda)(float64(currentNumEvents))
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

type singleValueFunction func(float64) float64

func (s singleValueFunction) Integrate(from, to, step float64) float64 {
	return Integrate(from, to, step, s)
}

type multiValueFunction func(float64, float64) float64

func (m multiValueFunction) IntegrateX(from, to, step float64, y float64) float64 {
	var res float64 = 0
	for x := from; x < to; x += step {
		res += m(x, y) * step
	}
	return res
}

func (m multiValueFunction) IntegrateY(from, to, step float64, x float64) float64 {
	var res float64 = 0
	for y := from; y < to; y += step {
		res += m(x, y) * step
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
	panic("Integral did not reach target value before maximum integration limit")
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

		lastValue = firstDerivativeValue
	}

	return nil // No inflection point found in the given range
}

// GetMaximumLikelihoodNormal calculates the maximum likelihood estimation (MLE) for a given dataset
// For a normal distribution, the MLE is the sample mean
// This function assumes the data is normally distributed
// It returns the mean of the data as the MLE
// If the data is empty, it panics
func GetMaximumLikelihoodNormal(data []float64) float64 {
	if len(data) == 0 {
		panic("Data cannot be empty")
	}

	return pkg.GetMean(data)
}

// GetMin returns the minimum value in a slice of float64
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

// GetMax returns the maximum value in a slice of float64
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

// GetMaximumLikelihoodPoisson find the lambda parameter for a Poisson distribution
// given a dataset, it finds the value of lambda that maximizes the log-likelihood function
func GetMaximumLikelihoodPoisson(data []float64, start, end, step float64) float64 {
	start = math.Max(GetMax(data), start) // Ensure start is positive to avoid division by zero
	end = math.Min(GetMin(data), end)     // Ensure end is not less than the minimum data value
	logLikelihoodFn := GetLogLikelihoodFunctionPoisson(data)
	criticalPoint := FindCriticalPoint(logLikelihoodFn, start, end, step)
	if criticalPoint == nil {
		panic("No critical point found for Poisson distribution")
	}
	return *criticalPoint
}

// GetLogLikelihoodFunctionPoisson is a wrapper function that returns a log-likelihood function for a Poisson distribution
func GetLogLikelihoodFunctionPoisson(data []float64) func(float64) float64 {
	return func(lambda float64) float64 {
		res := 1.0
		for _, x := range data {
			res *= GetPoissonDistributionFunction(lambda)(x)
		}
		return math.Log(res)
	}
}

// GetLogLikelihoodFunctionNormal is a wrapper function that returns a log-likelihood function for a normal distribution
func GetLogLikelihoodFunctionNormal(data []float64) func(float64, float64) float64 {
	return func(mean, stdDev float64) float64 {
		if stdDev <= 0 {
			panic("Standard deviation must be positive")
		}
		res := 1.0
		for _, x := range data {
			res *= GetNormalDistributionFunction(mean, stdDev)(x)
		}
		return math.Log(res)
	}
}

// GetLogLikelihoodFunctionExponential is a wrapper function that returns a log-likelihood function for an exponential distribution
func GetLogLikelihoodFunctionExponential(data []float64) func(float64) float64 {
	return func(lambda float64) float64 {
		res := 0.0
		for _, x := range data {
			res += math.Log(GetExponentialDistributionFunction(lambda)(x))
		}
		return res
	}
}

// GetMaximumLikelihoodExponentialDistribution finds the MLE, the best estimate for the lambda parameter of an exponential distribution
// since we use numerrical approximations, we need to find the critical point of the log-likelihood function setting a minimum and maximum range
// for the lambda parameter and the step size for the search.
// the data is a slice of float64 representing the observed values
func GetMaximumLikelihoodExponentialDistribution(data []float64, start, end, step float64) float64 {
	if len(data) == 0 {
		panic("Data cannot be empty")
	}

	criticalPoint := FindCriticalPoint(GetLogLikelihoodFunctionExponential(data), start, end, step)
	if criticalPoint == nil {
		panic("No critical point found for exponential distribution")
	}
	return *criticalPoint
}

type JointPDF struct {
	function  func(x float64, y float64) float64
	rangeMinX float64
	rangeMaxX float64
	rangeMinY float64
	rangeMaxY float64
}

func (j *JointPDF) GetMarginalX(step float64) func(float64) float64 {
	return func(x float64) float64 {
		wrapper := func(y float64) float64 {
			return j.function(x, y)
		}
		integral := Integrate(j.rangeMinY, j.rangeMaxY, step, wrapper)
		return integral
	}
}

func (j *JointPDF) GetMarginalY(step float64) func(float64) float64 {
	return func(y float64) float64 {
		wrapper := func(x float64) float64 {
			return j.function(x, y)
		}
		integral := Integrate(j.rangeMinX, j.rangeMaxX, step, wrapper)
		return integral
	}
}

func NewJointPDF(function func(float64, float64) float64, minX, maxX, minY, maxY float64) *JointPDF {
	if minX >= maxX || minY >= maxY {
		panic("Invalid range: min must be less than max")
	}
	if function == nil {
		panic("Function cannot be nil")
	}

	return &JointPDF{
		function:  function,
		rangeMinX: minX,
		rangeMaxX: maxX,
		rangeMinY: minY,
		rangeMaxY: maxY,
	}
}

type JointPMF struct {
	values map[[2]float64]float64
}

func NewJointPMF() *JointPMF {
	return &JointPMF{
		values: make(map[[2]float64]float64),
	}
}

func (j *JointPMF) TotalSumProbabilities() float64 {
	var total float64
	for _, prob := range j.values {
		total += prob
	}
	return total
}

func (j *JointPMF) Set(x, y, prob float64) {
	if prob < 0 || prob > 1 {
		panic("Probability must be between 0 and 1")
	}

	// validate that the sum of all probabilities does not exceed 1
	if prob+j.TotalSumProbabilities()-1 > 0.01 {
		panic("Total probability cannot exceed 1")
	}

	j.values[[2]float64{x, y}] = prob
}

func (j *JointPMF) Get(x, y float64) float64 {
	return j.values[[2]float64{x, y}]
}

func (j *JointPMF) GetMarginalX() *PMF {
	pmf := NewPMF()
	for key, prob := range j.values {
		valueSoFar := pmf.Get(key[0])
		pmf.Set(key[0], valueSoFar+prob)
	}
	return pmf
}

func (j *JointPMF) GetMarginalY() *PMF {
	pmf := NewPMF()
	for key, prob := range j.values {
		valueSoFar := pmf.Get(key[1])
		pmf.Set(key[1], valueSoFar+prob)
	}
	return pmf
}

func FindLocalMinimum(fn func(float64) float64, start, step float64) float64 {
	// start at start
	// compare with x+step and x-step
	// go to smaller, evaluate again
	// store visited values in map
	// visit next until encounters values already visited
	visited := map[float64]bool{}
	cursor := start
	for !visited[cursor] {
		visited[cursor] = true
		currentValue := fn(cursor)
		nextValue := fn(cursor + step)
		previousValue := fn(cursor - step)

		if nextValue < currentValue {
			cursor += step
		}
		if previousValue < currentValue {
			cursor -= step
		}
		cursor = math.Round(cursor*1000) / 1000
	}
	return cursor
}

func GetConfidenceIntervalForNormalDistributionWithZscore(mean, stdDev, confidenceLevel, from, to, step float64) (float64, float64) {
	if confidenceLevel <= 0 || confidenceLevel >= 1 {
		panic("Confidence level must be between 0 and 1")
	}

	z := GetRightTailZScoreFromProbability(confidenceLevel, from, to, step)
	if z == 0 {
		panic("Z-score cannot be zero")
	}
	marginOfError := z * stdDev
	lower := mean - marginOfError
	upper := mean + marginOfError

	return lower, upper
}

// GetRightTailZScoreFromProbability returns the z-score for a left tail of a normal distribution
// to get this value we integrate the normal distribution function from -Inf until the cdf reaches the confidence level
// since we do numerical integration, we need to specify a range for the integration
func GetRightTailZScoreFromProbability(confidenceLevel, from, to, step float64) float64 {
	standardNormal := GetNormalDistributionFunction(0, 1)
	return IntegrateUntilValue(from, to, confidenceLevel, step, standardNormal)
}

// GetStandardErrorEmpirical calculates the standard error of a statistic this can be
// the mean, proportion, or regression coefficient
// In theory we could use the notation Var(estimator) / n but since we typically
// only have one sample, we cannot get the variance of the estimator, we only have
// one value per estimator
func GetStandardErrorEmpirical(estimator float64, sampleSize int) float64 {
	if sampleSize <= 1 {
		panic("Sample size must be greater than 1")
	}
	return estimator / math.Sqrt(float64(sampleSize))
}

func GetRightTailTScoreFromProbability(confidenceLevel, degreesOfFreedom, from, to, step float64) float64 {
	tDistribution := GetStudentTDistributionFunction(degreesOfFreedom)
	return IntegrateUntilValue(from, to, confidenceLevel, step, tDistribution)
}

// GetMeanConfidenceIntervalForNormalDistribution calculates the confidence interval for a normal distribution
// given a sample mean, sample standard deviation, sample size, confidence level, and range for integration
// Internally, it will use the z-score for large samples (n > 30) or t-score for small samples (n <= 30)
func GetMeanConfidenceIntervalForNormalDistribution(sampleMean, sampleStdDev float64, sampleSize int, confidenceLevel, from, to, step float64) (float64, float64) {
	alpha := (1 - confidenceLevel) / 2

	// score could be a z-score or t-score depending on the sample size
	var score float64
	if sampleSize > 30 {
		// if sample size is greater than 30, we can use the z-score
		score = -GetRightTailZScoreFromProbability(alpha, from, to, step)
	} else {
		// if sample size is less than or equal to 30, we use the t-score
		degreesOfFreedom := float64(sampleSize - 1)
		score = -GetRightTailTScoreFromProbability(alpha, degreesOfFreedom, from, to, step)
	}

	marginOfError := score * GetStandardErrorEmpirical(sampleStdDev, sampleSize)
	lowerBound := sampleMean - marginOfError
	upperBound := sampleMean + marginOfError

	return lowerBound, upperBound
}

// GetProportionConfidenceInterval calculates the confidence interval for a proportion
// given the success probability, sample size, confidence level, and range for integration
func GetProportionConfidenceInterval(successProbability, sampleSize, confidenceLevel, from, to, step float64) (float64, float64) {
	alpha := (1 - confidenceLevel) / 2
	zScore := -GetRightTailZScoreFromProbability(alpha, from, to, step)

	// Calculate the standard error
	// SE = sqrt((p * (1 - p)) / n)
	x := successProbability * (1 - successProbability)
	standardError := math.Sqrt(x / sampleSize)

	// Calculate the margin of error
	marginOfError := zScore * standardError

	// Return the confidence interval
	upper := successProbability + marginOfError
	lower := successProbability - marginOfError
	return lower, upper
}

func GetTScore(degreesOfFreedom, confidenceLevel, from, to, step float64) float64 {
	if degreesOfFreedom <= 0 {
		panic("Degrees of freedom must be greater than 0")
	}
	if confidenceLevel <= 0 || confidenceLevel >= 1 {
		panic("Confidence level must be between 0 and 1")
	}

	return GetRightTailTScoreFromProbability(confidenceLevel, degreesOfFreedom, from, to, step)
}

func GetStudentTStatistic(sampleMean, populationMean, sampleStandardDeviation float64, sampleSize int) float64 {
	tStatistic := (sampleMean - populationMean) / (sampleStandardDeviation / math.Sqrt(float64(sampleSize)))
	return tStatistic
}

func GetZScore(sampleMean, populationMean, sampleStandardDeviation float64, sampleSize int) float64 {
	zScore := (sampleMean - populationMean) / (sampleStandardDeviation / math.Sqrt(float64(sampleSize)))
	return zScore
}

// GetCovariance calculates the covariance between two variables
// This is also just SSXY divided by n-1
func GetCovariance(x, y []float64) float64 {
	if len(x) != len(y) {
		panic("Vectors must be of the same length")
	}

	meanX := pkg.GetMean(x)
	meanY := pkg.GetMean(y)

	c := 0.0
	for i := range x {
		diffX := x[i] - meanX
		diffY := y[i] - meanY
		c += diffX * diffY
	}
	covariance := c / float64(len(x)-1) // sample covariance
	return covariance
}

// GetVariance calculates the variance of a sample
// This is also just the SSX divided by n
func GetVariance(x []float64) float64 {
	if len(x) == 0 {
		panic("Vector must not be empty")
	}

	mean := pkg.GetMean(x)
	c := 0.0
	for i := range x {
		c += (x[i] - mean) * (x[i] - mean)
	}
	variance := c / float64(len(x)-1) // sample variance
	return variance
}

// GetCorrelation calculates the correlation between two variables
// This is also just the covariance divided by the product of the standard deviations
func GetCorrelation(x, y []float64) float64 {
	if len(x) != len(y) {
		panic("Vectors must be of the same length")
	}

	covariance := GetCovariance(x, y)
	stdDevX := math.Sqrt(GetVariance(x))
	stdDevY := math.Sqrt(GetVariance(y))

	if stdDevX == 0 || stdDevY == 0 {
		return 0
	}

	return covariance / (stdDevX * stdDevY)
}

// GetProbabilityFromZScore calculates the probability associated with a z-score
// This can be used as a z score table to retrieve values by zscore.
// If you want the zscore use GetZScoreFromProbability
func GetProbabilityFromZScore(zScore float64) float64 {
	// Use the cumulative distribution function (CDF) of the standard normal distribution
	// to find the probability associated with the z-score
	return Integrate(-100, zScore, 0.001, GetNormalDistributionFunction(0, 1))
}

type HypothesisTest int

const (
	TwoTailed   HypothesisTest = 0
	LeftTailed  HypothesisTest = 1
	RightTailed HypothesisTest = 2
)

// GetPValueFromZScore returns the p-value corresponding to a z-score in a given test.
// The p-value (probability value) is the probability of observing a test statistic
// at least as extreme as the one obtained, assuming the null hypothesis (H0) is true.
//
// Interpretation:
//   - A small p-value means such an extreme result would be unlikely under H0,
//     providing evidence against H0.
//   - A large p-value means such a result is quite likely under H0,
//     so the data provide little or no evidence against H0.
func GetPValueFromZScore(zScore float64, testType HypothesisTest) float64 {
	// Use the cumulative distribution function (CDF) of the standard normal distribution
	// to find the p-value associated with the z-score
	switch testType {
	case LeftTailed:
		return GetProbabilityFromZScore(zScore)
	case TwoTailed:
		return 2 * (1 - GetProbabilityFromZScore(math.Abs(zScore)))
	case RightTailed:
		return 1 - GetProbabilityFromZScore(zScore)
	default:
		return 0
	}
}
