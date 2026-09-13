// Copyright (c) 2026 Ted Dunning
// SPDX-License-Identifier: MIT

package model

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"strconv"
	"time"
)

// TrainConfig holds the configuration loaded from the JSON config file.
type TrainConfig struct {
	BucketIntervalRaw     any     `json:"bucket_interval"`          // Can be float64 seconds or string like "60s"
	Horizon               int     `json:"horizon"`                  // Number of lag buckets
	RegularizationPenalty float64 `json:"regularization_penalty"`   // Regularization penalty size (L1 lambda)
	Epsilon               float64 `json:"epsilon"`                  // Small constant added inside log: log(count + epsilon)
	MaxIterations         int     `json:"max_iterations,omitempty"` // Optional max iterations for Lasso
	Tolerance             float64 `json:"tolerance,omitempty"`      // Optional convergence tolerance
	FitIntercept          *bool   `json:"fit_intercept,omitempty"`  // Optional whether to fit intercept (default true)
}

// ResolvedConfig represents normalized TrainConfig with parsed values.
type ResolvedConfig struct {
	BucketInterval        float64 `json:"bucket_interval"`        // In seconds
	Horizon               int     `json:"horizon"`                // Lag count
	RegularizationPenalty float64 `json:"regularization_penalty"` // L1 penalty
	Epsilon               float64 `json:"epsilon"`                // Small constant
	MaxIterations         int     `json:"max_iterations"`
	Tolerance             float64 `json:"tolerance"`
	FitIntercept          bool    `json:"fit_intercept"`
}

// LoadConfig reads and validates a JSON config file.
func LoadConfig(filePath string) (*ResolvedConfig, error) {
	data, err := os.ReadFile(filePath)
	if err != nil {
		return nil, fmt.Errorf("failed to read config file %s: %w", filePath, err)
	}

	return ParseConfig(data)
}

// ParseConfig parses and validates JSON config data.
func ParseConfig(data []byte) (*ResolvedConfig, error) {
	var raw struct {
		BucketInterval           any      `json:"bucket_interval"`
		BucketIntervalSecs       *float64 `json:"bucket_interval_seconds"`
		Horizon                  int      `json:"horizon"`
		RegularizationPenalty    *float64 `json:"regularization_penalty"`
		RegularizationPenaltyAlt *float64 `json:"regularization_penalty_size"`
		Lambda                   *float64 `json:"lambda"`
		Penalty                  *float64 `json:"penalty"`
		Epsilon                  *float64 `json:"epsilon"`
		MaxIterations            int      `json:"max_iterations"`
		Tolerance                float64  `json:"tolerance"`
		FitIntercept             *bool    `json:"fit_intercept"`
	}

	if err := json.Unmarshal(data, &raw); err != nil {
		return nil, fmt.Errorf("failed to parse JSON config: %w", err)
	}

	res := &ResolvedConfig{
		Horizon:       raw.Horizon,
		MaxIterations: raw.MaxIterations,
		Tolerance:     raw.Tolerance,
		FitIntercept:  true,
	}

	if raw.FitIntercept != nil {
		res.FitIntercept = *raw.FitIntercept
	}

	// Parse bucket interval
	if raw.BucketIntervalSecs != nil && *raw.BucketIntervalSecs > 0 {
		res.BucketInterval = *raw.BucketIntervalSecs
	} else if raw.BucketInterval != nil {
		switch v := raw.BucketInterval.(type) {
		case float64:
			res.BucketInterval = v
		case int:
			res.BucketInterval = float64(v)
		case string:
			if d, err := time.ParseDuration(v); err == nil {
				res.BucketInterval = d.Seconds()
			} else if sec, err := strconv.ParseFloat(v, 64); err == nil {
				res.BucketInterval = sec
			} else {
				return nil, fmt.Errorf("invalid bucket_interval format: %q", v)
			}
		default:
			return nil, fmt.Errorf("unsupported bucket_interval type")
		}
	}

	// Parse regularization penalty
	if raw.RegularizationPenalty != nil {
		res.RegularizationPenalty = *raw.RegularizationPenalty
	} else if raw.RegularizationPenaltyAlt != nil {
		res.RegularizationPenalty = *raw.RegularizationPenaltyAlt
	} else if raw.Lambda != nil {
		res.RegularizationPenalty = *raw.Lambda
	} else if raw.Penalty != nil {
		res.RegularizationPenalty = *raw.Penalty
	}

	// Parse epsilon
	if raw.Epsilon != nil {
		res.Epsilon = *raw.Epsilon
	} else {
		res.Epsilon = 1e-4
	}

	// Validation
	if res.BucketInterval <= 0 {
		return nil, fmt.Errorf("bucket_interval must be positive, got %f", res.BucketInterval)
	}
	if res.Horizon <= 0 {
		return nil, fmt.Errorf("horizon must be positive, got %d", res.Horizon)
	}
	if res.RegularizationPenalty < 0 {
		return nil, fmt.Errorf("regularization_penalty must be non-negative, got %f", res.RegularizationPenalty)
	}
	if res.Epsilon <= 0 {
		return nil, fmt.Errorf("epsilon must be positive, got %f", res.Epsilon)
	}
	if res.MaxIterations <= 0 {
		res.MaxIterations = 2000
	}
	if res.Tolerance <= 0 {
		res.Tolerance = 1e-6
	}

	return res, nil
}

// TrainingStats holds training diagnostic statistics.
type TrainingStats struct {
	NumEvents      int     `json:"num_events"`
	NumBuckets     int     `json:"num_buckets"`
	FirstEventTime float64 `json:"first_event_time"`
	LastEventTime  float64 `json:"last_event_time"`
	Iterations     int     `json:"iterations"`
	Converged      bool    `json:"converged"`
	MSE            float64 `json:"mse"`
	NonZeroWeights int     `json:"non_zero_weights"`
}

// ModelOutput is the JSON structure containing the regression parameters and configuration.
type ModelOutput struct {
	BucketInterval        float64        `json:"bucket_interval"`
	Horizon               int            `json:"horizon"`
	Epsilon               float64        `json:"epsilon"`
	RegularizationPenalty float64        `json:"regularization_penalty"`
	Intercept             float64        `json:"intercept"`
	Weights               []float64      `json:"weights"`
	TrainingStats         *TrainingStats `json:"training_stats,omitempty"`
}

// PredictLogCount calculates y_hat = intercept + sum(weights_j * lags_j).
func (m *ModelOutput) PredictLogCount(lags []float64) float64 {
	pred := m.Intercept
	for i := 0; i < len(m.Weights) && i < len(lags); i++ {
		pred += m.Weights[i] * math.Log(lags[i]+m.Epsilon)
	}
	return pred
}

// PredictExpectedCount calculates the predicted event count per bucket:
// expected_count = max(exp(y_hat) - epsilon, 0).
func (m *ModelOutput) PredictExpectedCount(lags []float64) float64 {
	yHat := m.PredictLogCount(lags)
	count := math.Exp(yHat)
	if count < 0 {
		return 0
	}
	return count
}

// Save writes the model JSON to a file or stdout.
func (m *ModelOutput) Save(filePath string) error {
	data, err := json.MarshalIndent(m, "", "  ")
	if err != nil {
		return fmt.Errorf("failed to serialize model: %w", err)
	}
	return os.WriteFile(filePath, data, 0644)
}

// LoadModel reads a model JSON from a file.
func LoadModel(filePath string) (*ModelOutput, error) {
	data, err := os.ReadFile(filePath)
	if err != nil {
		return nil, fmt.Errorf("failed to read model file %s: %w", filePath, err)
	}
	var model ModelOutput
	if err := json.Unmarshal(data, &model); err != nil {
		return nil, fmt.Errorf("failed to parse model JSON: %w", err)
	}
	if model.BucketInterval <= 0 {
		return nil, fmt.Errorf("invalid bucket_interval in model: %f", model.BucketInterval)
	}
	if model.Horizon <= 0 || len(model.Weights) != model.Horizon {
		return nil, fmt.Errorf("model weights length (%d) does not match horizon (%d)", len(model.Weights), model.Horizon)
	}
	return &model, nil
}
