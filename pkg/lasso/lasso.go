package lasso

import (
	"fmt"
	"math"
)

// Config defines the hyperparameters and settings for L1 regularized linear regression.
type Config struct {
	Lambda        float64 `json:"regularization_penalty"` // Regularization penalty size (L1 lambda)
	MaxIterations int     `json:"max_iterations"`          // Maximum number of coordinate descent iterations
	Tolerance     float64 `json:"tolerance"`               // Convergence tolerance (max parameter change)
	FitIntercept  bool    `json:"fit_intercept"`           // Whether to fit an unregularized intercept
}

// DefaultConfig returns default configuration for Lasso regression.
func DefaultConfig() Config {
	return Config{
		Lambda:        0.01,
		MaxIterations: 2000,
		Tolerance:     1e-6,
		FitIntercept:  true,
	}
}

// Model represents the learned sparse linear regression model parameters.
type Model struct {
	Intercept float64   `json:"intercept"`
	Weights   []float64 `json:"weights"`
}

// Predict calculates the linear prediction for a single feature vector x.
func (m *Model) Predict(x []float64) float64 {
	pred := m.Intercept
	for i := 0; i < len(m.Weights) && i < len(x); i++ {
		pred += m.Weights[i] * x[i]
	}
	return pred
}

// ProgressCallback is called after each iteration with current training statistics.
type ProgressCallback func(iteration int, maxDelta float64, mse float64, nonZero int)

// FitResult holds the fitted model and metadata about the training run.
type FitResult struct {
	Model          Model
	Iterations     int
	Converged      bool
	MSE            float64
	NonZeroWeights int
}

// SoftThreshold performs the soft-thresholding operator S(rho, lambda).
func SoftThreshold(rho, lambda float64) float64 {
	if rho > lambda {
		return rho - lambda
	}
	if rho < -lambda {
		return rho + lambda
	}
	return 0.0
}

// Fit trains an L1 regularized linear regression (Lasso) model using coordinate descent.
// X is N samples x p features.
// y is N target values.
// progress callback is optional.
func Fit(X [][]float64, y []float64, cfg Config, onProgress ProgressCallback) (*FitResult, error) {
	n := len(y)
	if n == 0 {
		return nil, fmt.Errorf("cannot fit with 0 observations")
	}
	if len(X) != n {
		return nil, fmt.Errorf("X has %d rows but y has %d elements", len(X), n)
	}
	p := len(X[0])
	if p == 0 {
		return nil, fmt.Errorf("feature dimension cannot be 0")
	}

	if cfg.MaxIterations <= 0 {
		cfg.MaxIterations = 1000
	}
	if cfg.Tolerance <= 0 {
		cfg.Tolerance = 1e-6
	}

	// Precompute column sum of squares: z_j = (1/N) * sum_i (X_ij^2)
	colNormSq := make([]float64, p)
	for j := 0; j < p; j++ {
		sumSq := 0.0
		for i := 0; i < n; i++ {
			sumSq += X[i][j] * X[i][j]
		}
		colNormSq[j] = sumSq / float64(n)
	}

	// Initialize weights to 0
	weights := make([]float64, p)
	intercept := 0.0

	// If fitting intercept, initialize to mean of y
	if cfg.FitIntercept {
		sumY := 0.0
		for i := 0; i < n; i++ {
			sumY += y[i]
		}
		intercept = sumY / float64(n)
	}

	// Initialize residuals: r_i = y_i - (intercept + X_i * weights) = y_i - intercept
	residuals := make([]float64, n)
	for i := 0; i < n; i++ {
		residuals[i] = y[i] - intercept
	}

	converged := false
	var finalIter int
	var finalMSE float64
	var nonZero int

	invN := 1.0 / float64(n)

	for iter := 1; iter <= cfg.MaxIterations; iter++ {
		maxChange := 0.0

		// 1. Update intercept if configured
		if cfg.FitIntercept {
			sumRes := 0.0
			for i := 0; i < n; i++ {
				sumRes += residuals[i]
			}
			dIntercept := sumRes * invN
			intercept += dIntercept
			for i := 0; i < n; i++ {
				residuals[i] -= dIntercept
			}
			maxChange = math.Max(maxChange, math.Abs(dIntercept))
		}

		// 2. Coordinate descent over all features j = 0..p-1
		for j := 0; j < p; j++ {
			z_j := colNormSq[j]
			if z_j == 0 {
				// Constant zero feature has no effect
				weights[j] = 0.0
				continue
			}

			// rho_j = (1/N) * sum_i (X_ij * (residuals_i + X_ij * weights_j))
			//       = (1/N) * sum_i (X_ij * residuals_i) + z_j * weights_j
			dotProduct := 0.0
			for i := 0; i < n; i++ {
				dotProduct += X[i][j] * residuals[i]
			}
			rho_j := (dotProduct * invN) + z_j*weights[j]

			newWeight := SoftThreshold(rho_j, cfg.Lambda) / z_j
			dWeight := newWeight - weights[j]

			if dWeight != 0 {
				// Update residuals: r_i = r_i - X_ij * dWeight
				for i := 0; i < n; i++ {
					residuals[i] -= X[i][j] * dWeight
				}
				change := math.Abs(dWeight)
				if change > maxChange {
					maxChange = change
				}
				weights[j] = newWeight
			}
		}

		// Compute MSE and non-zero count
		sumSqRes := 0.0
		for i := 0; i < n; i++ {
			sumSqRes += residuals[i] * residuals[i]
		}
		finalMSE = sumSqRes * invN
		nonZero = 0
		for _, w := range weights {
			if w != 0 {
				nonZero++
			}
		}
		finalIter = iter

		if onProgress != nil && (iter == 1 || iter%20 == 0 || maxChange < cfg.Tolerance || iter == cfg.MaxIterations) {
			onProgress(iter, maxChange, finalMSE, nonZero)
		}

		if maxChange < cfg.Tolerance {
			converged = true
			break
		}
	}

	return &FitResult{
		Model: Model{
			Intercept: intercept,
			Weights:   weights,
		},
		Iterations:     finalIter,
		Converged:      converged,
		MSE:            finalMSE,
		NonZeroWeights: nonZero,
	}, nil
}
