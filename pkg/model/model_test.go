// Copyright (c) 2026 Ted Dunning
// SPDX-License-Identifier: MIT

package model

import (
	"math"
	"os"
	"path/filepath"
	"testing"
)

func TestParseConfig(t *testing.T) {
	jsonConfig := `{
		"bucket_interval": "60s",
		"horizon": 5,
		"regularization_penalty": 0.05,
		"epsilon": 0.01
	}`

	cfg, err := ParseConfig([]byte(jsonConfig))
	if err != nil {
		t.Fatalf("ParseConfig failed: %v", err)
	}

	if cfg.BucketInterval != 60.0 {
		t.Errorf("expected BucketInterval 60.0, got %f", cfg.BucketInterval)
	}
	if cfg.Horizon != 5 {
		t.Errorf("expected Horizon 5, got %d", cfg.Horizon)
	}
	if cfg.RegularizationPenalty != 0.05 {
		t.Errorf("expected RegularizationPenalty 0.05, got %f", cfg.RegularizationPenalty)
	}
	if cfg.Epsilon != 0.01 {
		t.Errorf("expected Epsilon 0.01, got %f", cfg.Epsilon)
	}
}

func TestModelPredict(t *testing.T) {
	m := &ModelOutput{
		BucketInterval: 10.0,
		Horizon:        2,
		Epsilon:        0.5,
		Intercept:      math.Log(10.5), // so baseline expected count is 10.0
		Weights:        []float64{0.0, 0.0},
	}

	expectedCount := m.PredictExpectedCount([]float64{5.0, 5.0})
	if math.Abs(expectedCount-10.0) > 1e-4 {
		t.Errorf("expected count 10.0, got %f", expectedCount)
	}
}

func TestModelSaveLoad(t *testing.T) {
	tmpDir := t.TempDir()
	modelFile := filepath.Join(tmpDir, "test_model.json")

	orig := &ModelOutput{
		BucketInterval:        30.0,
		Horizon:               3,
		Epsilon:               0.1,
		RegularizationPenalty: 0.02,
		Intercept:             1.23,
		Weights:               []float64{0.5, 0.0, -0.2},
	}

	if err := orig.Save(modelFile); err != nil {
		t.Fatalf("Save failed: %v", err)
	}

	loaded, err := LoadModel(modelFile)
	if err != nil {
		t.Fatalf("LoadModel failed: %v", err)
	}

	if loaded.BucketInterval != orig.BucketInterval || loaded.Horizon != orig.Horizon || loaded.Intercept != orig.Intercept {
		t.Errorf("loaded model mismatch: got %+v, want %+v", loaded, orig)
	}

	_ = os.Remove(modelFile)
}
