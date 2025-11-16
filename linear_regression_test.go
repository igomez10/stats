package stats

import (
	"encoding/csv"
	"fmt"
	"log"
	"math"
	"os"
	"reflect"
	"strconv"
	"testing"
	"testing/quick"

	"github.com/igomez10/stats/pkg"
)

func TestExampleSimpleLinearRegression(t *testing.T) {
	// Tiny example
	x := []float64{1, 2, 3, 4, 5}
	y := []float64{2.1, 2.9, 3.7, 4.1, 5.2}

	model, err := CreateSLRModelWithOLS(x, y)
	if err != nil {
		log.Fatal(err)
	}
	fmt.Printf("SimpleLR: y = %.4f + %.4f*x\n", model.B0, model.B1)
	fmt.Println("Pred at x=6:", model.Predict(6))
}

func TestModel_Predict(t *testing.T) {
	type args struct {
		x float64
	}
	tests := []struct {
		name string
		m    SimpleModel
		args args
		want float64
	}{
		{"case 1", SimpleModel{B0: 1, B1: 2}, args{3}, 7},
		{"case 2", SimpleModel{B0: 0, B1: 1}, args{5}, 5},
		{"case 3", SimpleModel{B0: -1, B1: 1}, args{0}, -1},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := tt.m.Predict(tt.args.x); got != tt.want {
				t.Errorf("Model.Predict() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestFitSimpleLR(t *testing.T) {
	type args struct {
		x []float64
		y []float64
	}
	tests := []struct {
		name    string
		args    args
		want    SimpleModel
		wantErr bool
	}{
		{
			name: "valid case",
			args: args{
				x: []float64{1, 2, 3, 4, 5},
				y: []float64{1, 2, 3, 4, 5},
			},
			want:    SimpleModel{B0: 0, B1: 1},
			wantErr: false,
		},
		{
			name: "invalid case - different lengths",
			args: args{
				x: []float64{1, 2, 3},
				y: []float64{1, 2, 3, 4, 5},
			},
			want:    SimpleModel{},
			wantErr: true,
		},
		{
			name: "invalid case - zero variance",
			args: args{
				x: []float64{1, 1, 1, 1, 1},
				y: []float64{1, 2, 3, 4, 5},
			},
			want:    SimpleModel{},
			wantErr: true,
		},
		{
			name: "y = 2x",
			args: args{
				x: []float64{1, 2, 3, 4, 5},
				y: []float64{2, 4, 6, 8, 10},
			},
			want:    SimpleModel{B0: 0, B1: 2},
			wantErr: false,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := CreateSLRModelWithOLS(tt.args.x, tt.args.y)
			if (err != nil) != tt.wantErr {
				t.Errorf("FitSimpleLR() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if !reflect.DeepEqual(got, tt.want) {
				t.Errorf("FitSimpleLR() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestGetSSX(t *testing.T) {
	type args struct {
		x []float64
	}
	tests := []struct {
		name string
		args args
		want float64
	}{
		{"case1", args{[]float64{1, 2, 3}}, 2.0},
		{"case2", args{[]float64{4, 5, 6}}, 2.0},
		{"case3", args{[]float64{7, 8, 9}}, 2.0},
		{"case4", args{[]float64{10, 11, 12}}, 2.0},
		{"case5", args{[]float64{1, 100}}, 4900.5},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := GetSSXSimple(tt.args.x); got != tt.want {
				t.Errorf("GetSSX() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestExplainSSTInTermsOfSSRandSSE(t *testing.T) {
	// we want to prove that SST = SSR + SSE
	// basically total_variability = explained_variability + unexplained_variability
	type testcase struct {
		x   []float64
		y   []float64
		sst float64
		ssr float64
		sse float64
	}
	tests := []testcase{
		{
			x:   []float64{1, 2, 3, 4, 5},
			y:   []float64{2, 3, 5, 4, 6},
			sst: 10.0,
			ssr: 8.10,
			sse: 1.90,
		},
	}
	for _, tt := range tests {
		t.Run(fmt.Sprintf("SST: %v, SSR: %v, SSE: %v", tt.sst, tt.ssr, tt.sse), func(t *testing.T) {
			if got := GetSSE(tt.x, tt.y); math.Abs(got-tt.sse) > 1e-5 {
				t.Errorf("GetSSE() = %v, want %v", got, tt.sse)
			}

			if got := GetSSR(tt.x, tt.y); math.Abs(got-tt.ssr) > 1e-5 {
				t.Errorf("GetSSR() = %v, want %v", got, tt.ssr)
			}

			if got := GetSSTSimple(tt.x, tt.y); math.Abs(got-tt.sst) > 1e-5 {
				t.Errorf("GetSST() = %v, want %v", got, tt.sst)
			}

			if math.Abs(GetSSTSimple(tt.x, tt.y)-GetSSR(tt.x, tt.y)-GetSSE(tt.x, tt.y)) > 1 {
				t.Errorf("SST = %v, SSR + SSE = %v", GetSSTSimple(tt.x, tt.y), GetSSR(tt.x, tt.y)-GetSSE(tt.x, tt.y))
			}
		})
	}
}

func TestGetSumSquaresRegression(t *testing.T) {
	type args struct {
		x []float64
		y []float64
	}
	tests := []struct {
		name string
		args args
		want float64
	}{
		{
			name: "case1",
			args: args{
				x: []float64{1, 2, 3},
				y: []float64{2, 4, 6},
			},
			want: 8.0,
		},
		{
			name: "case2",
			args: args{
				x: []float64{1, 2, 3, 4, 5},
				y: []float64{2, 3, 5, 4, 6},
			},
			want: 8.10,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := GetSumSquaresRegressionSimple(tt.args.x, tt.args.y); got-tt.want > 1e-9 {
				t.Errorf("GetSumSquaresRegression() = %v, want %v", got, tt.want)
			}
		})
	}
}

// Property-based simulation: SSR(x, y) is non-negative for valid inputs
func TestGetSumSquaresRegression_Quick(t *testing.T) {
	// Property: SSR(x, y) is non-negative for valid inputs
	f := func(x []float64) bool {
		if len(x) < 2 {
			return true // vacuously true for too-small inputs
		}
		// sanitize x to avoid NaN/Inf
		xs := make([]float64, len(x))
		for i, v := range x {
			if math.IsNaN(v) || math.IsInf(v, 0) {
				v = float64(i)
			}
			if x[i] > 1e9 || x[i] < -1e9 {
				return true
			}
			xs[i] = v
		}

		// create a linear y from x (valid case)
		y := make([]float64, len(xs))
		for i, xv := range xs {
			y[i] = 1.5*xv + 2.0
			if math.IsInf(y[i], 0) {
				return true
			}
			if y[i] > 1e9 || y[i] < -1e9 {
				return true
			}
		}
		got := GetSumSquaresRegressionSimple(xs, y)
		if !math.IsNaN(got) && !math.IsInf(got, 0) && got >= 1e-5 {
			return true
		}

		return false
	}
	if err := quick.Check(f, &quick.Config{}); err != nil {
		t.Error(err)
	}
}

func TestExplainSSTInTermsOfSSRandSSE_Quick(t *testing.T) {
	// Check that SST = SSR + SSE
	f := func(x, y []int) bool {
		if len(x) != len(y) {
			return true
		}
		if len(x) == 0 {
			return true
		}

		for i := range x {
			if x[i] > math.MaxInt32 || x[i] < -math.MaxInt32 {
				x[i] = x[i] % 100

			}
		}
		// check if all x are same
		allSame := true
		for i := 1; i < len(x); i++ {
			if x[i] != x[0] {
				allSame = false
				break
			}
		}
		if allSame {
			return true // avoid zero variance
		}

		for i := range y {
			if y[i] > math.MaxInt32 || y[i] < -math.MaxInt32 {
				y[i] = y[i] % 100
			}
		}

		// check if all y are same
		allSame = true
		for i := 1; i < len(y); i++ {
			if y[i] != y[0] {
				allSame = false
				break
			}
		}
		if allSame {
			return true
		}

		xfloats := make([]float64, len(x))
		for i := range x {
			xfloats[i] = float64(x[i])
		}
		yfloats := make([]float64, len(y))
		for i := range y {
			yfloats[i] = float64(y[i])
		}

		sse := GetSSE(xfloats, yfloats)
		ssr := GetSSR(xfloats, yfloats)
		sst := GetSSTSimple(xfloats, yfloats)

		if math.IsNaN(sse) || math.IsNaN(ssr) {
			return false
		}

		result := sst - (ssr + sse)
		if result > 1e-9 {
			return false
		}

		return true
	}

	if err := quick.Check(f, &quick.Config{
		MaxCount: 1000000,
	}); err != nil {
		t.Error(err)
	}
}

func TestGetSlopeFromSSXYAndSSX(t *testing.T) {
	type args struct {
		x []float64
		y []float64
	}
	tests := []struct {
		name string
		args args
		want float64
	}{
		{
			name: "case1",
			args: args{
				x: []float64{1, 2, 3},
				y: []float64{2, 4, 6},
			},
			want: 2.0,
		},
		{
			name: "case2",
			args: args{
				x: []float64{1, 2, 3, 4, 5},
				y: []float64{2, 3, 5, 4, 6},
			},
			want: 0.9,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := GetSlopeFromSSXYAndSSX(tt.args.x, tt.args.y); got-tt.want > 1e-9 {
				t.Errorf("GetSlopeFromSSXYAndSSX() = %v, want %v", got, tt.want)
			}

			model, err := CreateSLRModelWithOLS(tt.args.x, tt.args.y)
			if err != nil {
				t.Errorf("CreateSLRModel() error = %v", err)
				return
			}
			if got := model.GetSlope(); got-tt.want > 1e-9 {
				t.Errorf("GetSlope() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestGetSlopeFromSSXYAndSSX_Quick(t *testing.T) {
	// Property: Slope is well-defined for valid inputs
	f := func(x, y []float64) bool {
		if len(x) > 1000 {
			x = x[:1000]
		}
		if len(x) > len(y) {
			x = x[:len(y)]
		} else {
			y = y[:len(x)]
		}
		if len(x) < 2 {
			return true
		}

		// Check for NaN/Inf values
		for i := range x {
			if math.IsNaN(x[i]) || math.IsInf(x[i], 0) {
				return false
			}
			x[i] = math.Mod(x[i], math.MaxInt16)
		}
		for i := range y {
			if math.IsNaN(y[i]) || math.IsInf(y[i], 0) {
				return false
			}
			y[i] = math.Mod(y[i], math.MaxInt16)
		}

		// Calculate slope
		slope := GetSlopeFromSSXYAndSSX(x, y)
		model, err := CreateSLRModelWithOLS(x, y)
		if err != nil {
			return false
		}

		if math.IsNaN(model.GetSlope()) || math.IsInf(model.GetSlope(), 0) {
			return false
		}

		if model.GetSlope() != slope {
			return false
		}

		return true
	}

	if err := quick.Check(f, &quick.Config{
		MaxCount: 1000,
	}); err != nil {
		t.Error(err)
	}
}

func TestGetInterceptFromSlopeAndMeans(t *testing.T) {
	type args struct {
		slope float64
		meanX float64
		meanY float64
	}
	tests := []struct {
		name string
		args args
		want float64
	}{
		{
			name: "case1",
			args: args{
				slope: 2.0,
				meanX: 2.0,
				meanY: 4.0,
			},
			want: 0.0,
		},
		{
			name: "case2",
			args: args{
				slope: 0.9,
				meanX: 3.0,
				meanY: 4.0,
			},
			want: 1.3,
		},
		{
			name: "case3",
			args: args{
				slope: 1.0,
				meanX: 1.0,
				meanY: 1.0,
			},
			want: 0.0,
		},
		{
			name: "case4",
			args: args{
				slope: 1.5,
				meanX: 2.0,
				meanY: 3.0,
			},
			want: 2.0,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := GetInterceptFromSlopeAndMeans(tt.args.slope, tt.args.meanX, tt.args.meanY); got-tt.want > 1e-9 {
				t.Errorf("GetInterceptFromSlopeAndMeans() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestGetInterceptFromSlopeAndMeans_Quick(t *testing.T) {
	// Property: Intercept is well-defined for valid inputs
	f := func(x, y []float64) bool {
		if len(x) > 1000 {
			x = x[:1000]
		}
		if len(x) > len(y) {
			x = x[:len(y)]
		} else {
			y = y[:len(x)]
		}
		if len(x) < 2 {
			return true
		}

		// Check for NaN/Inf values
		for i := range x {
			if math.IsNaN(x[i]) || math.IsInf(x[i], 0) {
				return false
			}
			x[i] = math.Mod(x[i], math.MaxInt16)
		}
		for i := range y {
			if math.IsNaN(y[i]) || math.IsInf(y[i], 0) {
				return false
			}
			y[i] = math.Mod(y[i], math.MaxInt16)
		}

		// Calculate slope
		model, err := CreateSLRModelWithOLS(x, y)
		if err != nil {
			return false
		}

		if math.IsNaN(model.GetSlope()) || math.IsInf(model.GetSlope(), 0) {
			return false
		}

		intercept := GetInterceptFromSlopeAndMeans(model.GetSlope(), pkg.GetMean(x), pkg.GetMean(y))
		if math.IsNaN(intercept) || math.IsInf(intercept, 0) {
			return false
		}

		if model.GetIntercept() != intercept {
			return false
		}

		return true
	}

	if err := quick.Check(f, &quick.Config{
		MaxCount: 1000,
	}); err != nil {
		t.Error(err)
	}
}

// Reimplementing SLR in python but in go
func TestTomatoMeterCorrelation(t *testing.T) {
	// load csv
	f, err := os.Open("test_fixtures/movie_ratings.csv")
	if err != nil {
		t.Fatalf("failed to open csv file: %v", err)
	}
	defer f.Close()

	reader := csv.NewReader(f)
	// Read the CSV data
	records, err := reader.ReadAll()
	if err != nil {
		t.Fatalf("failed to read csv file: %v", err)
	}

	type MovieRecord struct {
		ID                string `json:"id,omitempty" csv:"id"`
		Title             string `json:"title,omitempty" csv:"title"`
		TomatoMeterRating *int   `json:"tomatometer_rating,omitempty" csv:"tomatometer_rating,omitempty"`
		AudienceRating    *int   `json:"audience_rating,omitempty" csv:"audience_rating,omitempty"`
	}

	res := []MovieRecord{}

	// Process the records
	seen := map[string]bool{}
	for i := 1; i < len(records); i++ { // skip header
		record := records[i]
		if seen[record[1]] {
			continue
		} else {
			seen[record[1]] = true
		}
		var tom, aud *int
		if record[2] == "NA" {
			continue
		}

		tom_rating, err := strconv.Atoi(record[2])
		if err != nil {
			t.Fatalf("failed to convert tomatometer rating: %v", err)
		}
		tom = &tom_rating
		if record[3] == "NA" {
			continue
		}

		aud_rating, err := strconv.Atoi(record[3])
		if err != nil {
			t.Fatalf("failed to convert audience rating: %v", err)
		}
		aud = &aud_rating

		movie := MovieRecord{
			ID:                record[0],
			Title:             record[1],
			TomatoMeterRating: tom,
			AudienceRating:    aud,
		}

		res = append(res, movie)
		// Do something with the movie record
	}

	expectedFilteredMovieLength := 197
	if len(res) != expectedFilteredMovieLength {
		t.Error("unexpected length", len(res))
	}

	xTomatoRating := []float64{}
	for i := range res {
		xTomatoRating = append(xTomatoRating, float64(*res[i].TomatoMeterRating))
	}

	yAudienceRating := []float64{}
	for i := range res {
		yAudienceRating = append(yAudienceRating, float64(*res[i].AudienceRating))
	}

	model, err := CreateSLRModelWithOLS(xTomatoRating, yAudienceRating)
	if err != nil {
		t.Errorf("unexpected error")
	}

	expectedB0 := 34.5089
	if model.B0-expectedB0 > 0.01 {
		t.Error("unexpected b0", model.B0)
	}

	expectedB1 := 0.4461
	if model.B1-expectedB1 > 0.01 {
		t.Error("unexpected b1", model.B1)
	}

	// lets predict with a tomato rating of 88
	expectedTomatoRating := 73.762
	givenTomatoRating := 88.0
	if model.Predict(givenTomatoRating)-expectedTomatoRating > 0.01 {
		t.Error("unexpected prediction")
	}

	cov := GetCovariance(xTomatoRating, yAudienceRating)
	t.Log("covariance", cov)
	corr := GetCorrelation(xTomatoRating, yAudienceRating)
	t.Log("correlation", corr)

	varB_hat_1 := GetVariance(yAudienceRating) / sumSquares(xTomatoRating)
	t.Log("varB_hat_1", varB_hat_1)
}

func GetError(B0, B1, x, yGot float64) float64 {
	yExpected := B0 + B1*x
	return math.Abs(yExpected-yGot) * math.Abs(yExpected-yGot)
}

func GetTotalError(B0, B1 float64, xs, ys []float64) float64 {
	cumulativeError := 0.0
	for i := range xs {
		cumulativeError += GetError(B0, B1, xs[i], ys[i])
	}
	return cumulativeError
}

func TestBuildLinearRegressionModelManually(t *testing.T) {
	x := []float64{0, 1, 2, 3, 4, 5, 6, 7, 8, 10}
	y := []float64{5, 4, 6, 5, 8, 7, 7, 8, 8, 9}
	model, err := CreateSLRModelWithOLS(x, y)
	if err != nil {
		t.Error("unexpected error", err)
	}

	// now do the regression by hand
	var cPointB1 *float64
	b0DependentFunction := func(b0 float64) float64 {
		b1DependentFunction := func(b1 float64) float64 {
			return GetTotalError(b0, b1, x, y)
		}
		cPointB1 = FindCriticalPoint(b1DependentFunction, -1.01, 1.01, 0.001)
		return GetTotalError(b0, *cPointB1, x, y)
	}
	cPointB0 := FindCriticalPoint(b0DependentFunction, GetMin(y), GetMax(y), 0.001)

	if math.Abs(*cPointB0-model.B0) > 0.001 {
		t.Log("Critical point B0 Manual", *cPointB0)
		t.Log("Critical point B0 Model", model.B0)
	}

	if math.Abs(*cPointB1-model.B1) > 0.001 {
		t.Log("Critical point B1 Manual", *cPointB1)
		t.Log("Critical point B1 Model", model.B1)
	}

}

func TestQuiz1Manually(t *testing.T) {
	t.Run("1.b", func(t *testing.T) {
		// For the simple linear regression model, the larger the quantity
		// ∑ni=1(xi−x_mean)^2 is, the smaller the standard error of the least squares
		// slope estimator tends to be.
		x := []float64{0, 1, 2, 3, 4, 5, 6, 7, 8, 10}
		y := []float64{5, 4, 6, 5, 8, 7, 7, 8, 8, 9}
		standardErrorB1 := GetStandardErrorB1(x, y)

		x[0] = 999999999
		standardErrorB1New := GetStandardErrorB1(x, y)

		// We expect the standard error to be smaller because SSX is now bigger
		if standardErrorB1New >= standardErrorB1 {
			t.Error("unexpected standard error", standardErrorB1, standardErrorB1New)
		}
	})
}

func TestModel_GetCoefficientDetermination(t *testing.T) {
	tests := []struct {
		name string // description of this test case
		// Named input parameters for receiver constructor.
		x    []float64
		y    []float64
		want float64
	}{
		{
			name: "perfect model, should be good r2",
			x:    []float64{1, 2, 3, 4},
			y:    []float64{1, 2, 3, 4},
			want: 1,
		},
		{
			name: "one outlier, still not bad",
			x:    []float64{1, 2, 3, 4},
			y:    []float64{1, 2, 3, 10},
			want: 0.783,
		},
		{
			name: "random, very bad",
			x:    []float64{1, 2, 3, 4},
			y:    []float64{1e10, -1e10, -1e10, 1e10},
			want: 0,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := GetCoefficientDetermination(tt.x, tt.y)
			if math.Abs(got-tt.want) > 1e-3 {
				t.Errorf("GetCoefficientDetermination() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestMultipleLinearRegression(t *testing.T) {
	type args struct {
		X [][]float64
		Y []float64
	}

	tests := []struct {
		name      string
		args      args
		want      MultiLinearModel
		tolerance float64
	}{
		{
			name: "house-prices-3-features",
			args: args{
				X: [][]float64{
					{2100, 4, 5},
					{1600, 3, 15},
					{2400, 4, 2},
					{1410, 2, 30},
					{3000, 5, 8},
					{1850, 3, 12},
				},
				Y: []float64{420, 310, 460, 210, 560, 340},
			},
			want: MultiLinearModel{Betas: []float64{
				27.3114924,
				0.10495301,
				46.31520567,
				-1.85704885,
			}},
			tolerance: 1e-4,
		},
		{
			name: "simple-1-feature",
			args: args{
				X: [][]float64{
					{1},
					{2},
					{3},
					{4},
					{5},
					{6},
					{7},
					{8},
					{9},
					{10},
				},
				Y: []float64{2.3, 2.5, 2.7, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0},
			},
			want: MultiLinearModel{Betas: []float64{
				1.54,
				0.4291,
			}},
			tolerance: 1e-4,
		},
		{
			name: "two-features",
			args: args{
				X: [][]float64{
					{0, 5},
					{3, 4},
					{3, 4},
					{3, 2},
					{5, 1},
				},
				Y: []float64{2, 3, 4, 5, 6},
			},
			want: MultiLinearModel{Betas: []float64{
				5.6114,
				0.2370,
				-0.7109,
			}},
			tolerance: 1e-4,
		},
		{
			name: "three-features",
			args: args{
				X: [][]float64{
					{0, 5, 1},
					{3, 4, 2},
					{3, 4, 3},
					{3, 2, 4},
					{5, 3, 5},
					{7, 3, 6},
					{8, 2, 7},
					{9, 1, 8},
					{10, 0, 9},
				},
				Y: []float64{2, 3, 4, 5, 6, 7, 8, 9, 10},
			},
			want: MultiLinearModel{Betas: []float64{
				1.0,
				0,
				0,
				1.0,
			}},
			tolerance: 1e-4,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := CreateLRModelWithOLS(tt.args.X, tt.args.Y)
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}

			if len(got.Betas) != len(tt.want.Betas) {
				t.Fatalf("Betas length mismatch: got %d, want %d", len(got.Betas), len(tt.want.Betas))
			}

			for i := range tt.want.Betas {
				if got.Betas[i]-tt.want.Betas[i] > tt.tolerance {
					t.Errorf("beta[%d] = %v, want %v (tol=%v)", i, got.Betas[i], tt.want.Betas[i], tt.tolerance)
				}
			}
		})
	}
}

func TestMultiLinearModel_GetMSE(t *testing.T) {
	tests := []struct {
		name         string // description of this test case
		betas        []float64
		observations [][]float64
		actualOutput []float64
		want         float64
	}{
		{
			name:  "simple-1-feature",
			betas: []float64{0, 1},
			observations: [][]float64{
				{1},
				{2},
				{3},
				{4},
			},
			actualOutput: []float64{1, 2, 3, 4},
			want:         0,
		},
		{
			name:  "one outlier",
			betas: []float64{0, 1},
			observations: [][]float64{
				{1},
				{2},
				{3},
				{4},
			},
			actualOutput: []float64{1, 2, 3, 10},
			want:         18,
		},
		{
			name:  "beta non 0",
			betas: []float64{6, 15},
			observations: [][]float64{
				{1},
				{2},
				{3},
				{4},
			},
			actualOutput: []float64{21, 36, 51, 66},
			want:         0,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			m := MultiLinearModel{Betas: tt.betas}
			got := m.GetMSE(tt.observations, tt.actualOutput)
			if got-tt.want > 1e-9 {
				t.Errorf("GetMSE() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestMultiLinearModel_Predict(t *testing.T) {
	tests := []struct {
		name   string // description of this test case
		xInput []float64
		betas  []float64
		want   float64
	}{
		{
			name:   "simple-1-feature",
			xInput: []float64{4},
			betas:  []float64{0, 1},
			want:   4,
		},
		{
			name:   "two-features",
			xInput: []float64{3, 4},
			betas:  []float64{0, 1, 2},
			want:   11,
		},
		{
			name:   "three-features",
			xInput: []float64{10, 0, 9},
			betas:  []float64{0, 1, 2, 3},
			want:   37,
		},
		{
			name:   "four-features",
			xInput: []float64{1, 2, 3, 4},
			betas:  []float64{0, 1, 2, 3, 4},
			want:   40,
		},
		{
			name:   "with-intercept",
			xInput: []float64{5, 10},
			betas:  []float64{2, 3, 4},
			want:   2 + 3*5 + 4*10,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			m := MultiLinearModel{Betas: tt.betas}
			got := m.Predict(tt.xInput)
			if got-tt.want > 1e-9 {
				t.Errorf("Predict() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestCreateLRModelWithRidge(t *testing.T) {
	tests := []struct {
		name string // description of this test case
		// Named input parameters for target function.
		observations [][]float64
		actualOutput []float64
		lambda       float64
		want         MultiLinearModel
		wantErr      bool
	}{
		{
			name: "simple-1-feature",
			observations: [][]float64{
				{1, 4},
				{3, 8},
				{5, 6},
			},
			actualOutput: []float64{2, 3, 4},
			lambda:       1,
			want:         MultiLinearModel{Betas: []float64{0.625, 0.125}},
			wantErr:      false,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {

			// normalize observations
			normalizedObservations := make([][]float64, len(tt.observations))
			for i := range tt.observations[0] {
				col := make([]float64, len(tt.observations))
				for j := range tt.observations {
					col[j] = tt.observations[j][i]
				}
				mean := pkg.GetMean(col)
				std := math.Sqrt(pkg.GetSampleVariance(col))
				for j := range tt.observations {
					if normalizedObservations[j] == nil {
						normalizedObservations[j] = make([]float64, len(tt.observations[0]))
					}
					normalizedObservations[j][i] = (tt.observations[j][i] - mean) / std
				}
			}

			got, gotErr := CreateLRModelWithRidge(normalizedObservations, tt.actualOutput, tt.lambda)
			if gotErr != nil {
				if !tt.wantErr {
					t.Errorf("CreateLRModelWithRidge() failed: %v", gotErr)
				}
				return
			}
			if tt.wantErr {
				t.Fatal("CreateLRModelWithRidge() succeeded unexpectedly")
			}
			if len(got.Betas) != len(tt.want.Betas) {
				t.Fatalf("Betas length mismatch: got %d, want %d", len(got.Betas), len(tt.want.Betas))
			}
			for i := range tt.want.Betas {
				if math.Abs(got.Betas[i]-tt.want.Betas[i]) > 1e-3 {
					t.Errorf("beta[%d] = %v, want %v", i, got.Betas[i], tt.want.Betas[i])
				}
			}
		})
	}
}
func TestLassoLossFormula(t *testing.T) {
	tests := []struct {
		name string // description of this test case
		// Named input parameters for target function.
		observations [][]float64
		actualOutput []float64
		betas        []float64
		lambda       float64
		want         float64
	}{
		{
			name: "no penalty, perfect fit",
			observations: [][]float64{
				{1, 0},
				{0, 1},
			},
			actualOutput: []float64{1, 0},
			betas:        []float64{0, 1, 0},
			lambda:       0,
			want:         0,
		},
		{
			name: "no penalty, imperfect fit",
			observations: [][]float64{
				{1, 1},
				{0, 1},
			},
			actualOutput: []float64{1, 0},
			betas:        []float64{0, 1, 1},
			lambda:       0,
			want:         2,
		},
		{
			name: "1 penalty, perfect fit",
			observations: [][]float64{
				{1, 1},
				{0, 1},
			},
			actualOutput: []float64{1, 0},
			betas:        []float64{0, 1, 0},
			lambda:       1,
			want:         1,
		},
		{
			name: "1 penalty, other perfect fit",
			observations: [][]float64{
				{1, 1},
				{0, 1},
				{0, 1},
			},
			actualOutput: []float64{1, 0, 0},
			betas:        []float64{0, 1, 0},
			lambda:       1,
			want:         1,
		},
		{
			name: "1 penalty, imperfect fit",
			observations: [][]float64{
				{1, 1},
				{0, 1},
			},
			actualOutput: []float64{1, 0},
			betas:        []float64{0, 1, 1},
			lambda:       1,
			want:         4,
		},
		{
			name: "2 penalty, imperfect fit",
			observations: [][]float64{
				{1, 1},
				{0, 1},
			},
			actualOutput: []float64{1, 0},
			betas:        []float64{0, 1, 1},
			lambda:       2,
			want:         6,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := LassoLossFormula(tt.observations, tt.actualOutput, tt.betas, tt.lambda)
			if math.Abs(got-tt.want) > 1e-6 {
				t.Errorf("LassoLossFormula() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestRidgeLossFormula(t *testing.T) {
	tests := []struct {
		name string // description of this test case
		// Named input parameters for target function.
		observations [][]float64
		actualOutput []float64
		betas        []float64
		lambda       float64
		want         float64
	}{
		{
			name: "no penalty, perfect fit",
			observations: [][]float64{
				{1, 0},
				{0, 1},
			},
			actualOutput: []float64{1, 0},
			betas:        []float64{0, 1, 0},
			lambda:       0,
			want:         0,
		},
		{
			name: "no penalty, imperfect fit",
			observations: [][]float64{
				{1, 1},
				{0, 1},
			},
			actualOutput: []float64{1, 0},
			betas:        []float64{0, 1, 1},
			lambda:       0,
			want:         2,
		},
		{
			name: "1 penalty, perfect fit",
			observations: [][]float64{
				{1, 1},
				{0, 1},
			},
			actualOutput: []float64{1, 0},
			betas:        []float64{0, 1, 0},
			lambda:       1,
			want:         1,
		},
		{
			name: "1 penalty, other perfect fit",
			observations: [][]float64{
				{1, 1},
				{0, 1},
				{0, 1},
			},
			actualOutput: []float64{1, 0, 0},
			betas:        []float64{0, 1, 0},
			lambda:       1,
			want:         1,
		},
		{
			name: "1 penalty, imperfect fit",
			observations: [][]float64{
				{1, 1},
				{0, 1},
			},
			actualOutput: []float64{1, 0},
			betas:        []float64{0, 1, 1},
			lambda:       1,
			want:         4,
		},
		{
			name: "2 penalty, imperfect fit",
			observations: [][]float64{
				{1, 1},
				{0, 1},
			},
			actualOutput: []float64{1, 0},
			betas:        []float64{0, 1, 1},
			lambda:       2,
			want:         6,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := RidgeLossFormula(tt.observations, tt.actualOutput, tt.betas, tt.lambda)
			if math.Abs(got-tt.want) > 1e-6 {
				t.Errorf("RidgeLossFormula() = %v, want %v", got, tt.want)
			}
		})
	}
}
