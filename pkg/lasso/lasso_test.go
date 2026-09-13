// Copyright (c) 2026 Ted Dunning
// SPDX-License-Identifier: MIT

package lasso

import (
	"math"
	"testing"
)

func TestSoftThreshold(t *testing.T) {
	tests := []struct {
		rho    float64
		lambda float64
		want   float64
	}{
		{rho: 5.0, lambda: 2.0, want: 3.0},
		{rho: -5.0, lambda: 2.0, want: -3.0},
		{rho: 1.5, lambda: 2.0, want: 0.0},
		{rho: -1.5, lambda: 2.0, want: 0.0},
		{rho: 0.0, lambda: 2.0, want: 0.0},
	}

	for _, tt := range tests {
		got := SoftThreshold(tt.rho, tt.lambda)
		if math.Abs(got-tt.want) > 1e-9 {
			t.Errorf("SoftThreshold(%f, %f) = %f, want %f", tt.rho, tt.lambda, got, tt.want)
		}
	}
}

func TestLassoFit(t *testing.T) {
	// Synthetic linear model:
	// y = 2.0 + 3.0*x1 + 0.0*x2 + noise
	// Feature 1 is informative, Feature 2 is noise/zero.
	N := 100
	X := make([][]float64, N)
	y := make([]float64, N)

	for i := 0; i < N; i++ {
		x1 := float64(i) * 0.1
		x2 := float64((i%5)-2) * 0.05
		X[i] = []float64{x1, x2}
		y[i] = 2.0 + 3.0*x1
	}

	cfg := Config{
		Lambda:        0.001,
		MaxIterations: 2000,
		Tolerance:     1e-7,
		FitIntercept:  true,
	}

	res, err := Fit(X, y, cfg, nil)
	if err != nil {
		t.Fatalf("Fit failed: %v", err)
	}

	if !res.Converged {
		t.Errorf("expected model to converge")
	}

	// Check intercept ~ 2.0
	if math.Abs(res.Model.Intercept-2.0) > 0.05 {
		t.Errorf("expected intercept ~ 2.0, got %f", res.Model.Intercept)
	}

	// Check weight 0 ~ 3.0
	if math.Abs(res.Model.Weights[0]-3.0) > 0.05 {
		t.Errorf("expected weight[0] ~ 3.0, got %f", res.Model.Weights[0])
	}

	// Check weight 1 is 0.0 (sparse)
	if math.Abs(res.Model.Weights[1]) > 1e-3 {
		t.Errorf("expected weight[1] ~ 0.0, got %f", res.Model.Weights[1])
	}
}
