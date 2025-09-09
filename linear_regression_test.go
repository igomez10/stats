package stats

import (
	"encoding/csv"
	"fmt"
	"log"
	"math"
	"os"
	"reflect"
	"stats/pkg"
	"strconv"
	"testing"
	"testing/quick"
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
		m    Model
		args args
		want float64
	}{
		{"case 1", Model{B0: 1, B1: 2}, args{3}, 7},
		{"case 2", Model{B0: 0, B1: 1}, args{5}, 5},
		{"case 3", Model{B0: -1, B1: 1}, args{0}, -1},
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
		want    Model
		wantErr bool
	}{
		{
			name: "valid case",
			args: args{
				x: []float64{1, 2, 3, 4, 5},
				y: []float64{1, 2, 3, 4, 5},
			},
			want:    Model{B0: 0, B1: 1},
			wantErr: false,
		},
		{
			name: "invalid case - different lengths",
			args: args{
				x: []float64{1, 2, 3},
				y: []float64{1, 2, 3, 4, 5},
			},
			want:    Model{},
			wantErr: true,
		},
		{
			name: "invalid case - zero variance",
			args: args{
				x: []float64{1, 1, 1, 1, 1},
				y: []float64{1, 2, 3, 4, 5},
			},
			want:    Model{},
			wantErr: true,
		},
		{
			name: "y = 2x",
			args: args{
				x: []float64{1, 2, 3, 4, 5},
				y: []float64{2, 4, 6, 8, 10},
			},
			want:    Model{B0: 0, B1: 2},
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
			if got := GetSSX(tt.args.x); got != tt.want {
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

			if got := GetSST(tt.x, tt.y); math.Abs(got-tt.sst) > 1e-5 {
				t.Errorf("GetSST() = %v, want %v", got, tt.sst)
			}

			if math.Abs(GetSST(tt.x, tt.y)-GetSSR(tt.x, tt.y)-GetSSE(tt.x, tt.y)) > 1 {
				t.Errorf("SST = %v, SSR + SSE = %v", GetSST(tt.x, tt.y), GetSSR(tt.x, tt.y)-GetSSE(tt.x, tt.y))
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
			if got := GetSumSquaresRegression(tt.args.x, tt.args.y); got-tt.want > 1e-9 {
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
		got := GetSumSquaresRegression(xs, y)
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
		sst := GetSST(xfloats, yfloats)

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
