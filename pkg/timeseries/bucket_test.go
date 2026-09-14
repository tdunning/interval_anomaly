// Copyright (c) 2026 Ted Dunning
// SPDX-License-Identifier: MIT

package timeseries

import (
	"math"
	"testing"
)

func TestBucketEvents(t *testing.T) {
	events := []Event{
		{Seconds: 10.0},
		{Seconds: 12.0},
		{Seconds: 25.0},
		{Seconds: 40.0},
		{Seconds: 15.0}, // out-of-order
	}

	interval := 10.0
	bucketed, err := BucketEvents(events, interval)
	if err != nil {
		t.Fatalf("BucketEvents failed: %v", err)
	}

	if bucketed.StartTime != 10.0 {
		t.Errorf("expected startTime 10.0, got %f", bucketed.StartTime)
	}

	// Range is 10.0 to 40.0:
	// Bucket 0: [10, 20) -> 10.0, 12.0, 15.0 => count 3
	// Bucket 1: [20, 30) -> 25.0 => count 1
	// Bucket 2: [30, 40) -> (empty) => count 0
	// Bucket 3: [40, 50) -> 40.0 => count 1
	expectedCounts := []float64{3, 1, 0, 1}
	if len(bucketed.Counts) != len(expectedCounts) {
		t.Fatalf("expected %d buckets, got %d", len(expectedCounts), len(bucketed.Counts))
	}

	for i, c := range expectedCounts {
		if bucketed.Counts[i] != c {
			t.Errorf("bucket %d: expected count %f, got %f", i, c, bucketed.Counts[i])
		}
	}
}

func TestBuildARDataset(t *testing.T) {
	bucketed := &BucketedSeries{
		StartTime:      0.0,
		BucketInterval: 1.0,
		Counts:         []float64{1, 2, 3, 4, 5, 6},
		TotalEvents:    21,
	}

	horizon := 2
	epsilon := 0.5
	dataset, err := bucketed.BuildARDataset(horizon, epsilon)
	if err != nil {
		t.Fatalf("BuildARDataset failed: %v", err)
	}

	// Total buckets = 6, horizon = 2 => 4 samples (indices 2, 3, 4, 5)
	if len(dataset.Y) != 4 {
		t.Fatalf("expected 4 samples, got %d", len(dataset.Y))
	}

	// Sample 0 (target index 2):
	// target count = 3 => Y[0] = log(3 + 0.5) = log(3.5)
	// features: lag 1 (index 1 => count 2), lag 2 (index 0 => count 1)
	expectedY0 := math.Log(3.5)
	if math.Abs(dataset.Y[0]-expectedY0) > 1e-6 {
		t.Errorf("sample 0: expected Y=%f, got %f", expectedY0, dataset.Y[0])
	}
	if math.Abs(dataset.X[0][0]-math.Log(2.5)) > 1e-6 || math.Abs(dataset.X[0][1]-math.Log(1.5)) > 1e-6 {
		t.Errorf("sample 0: expected X=[log(2.5), log(1.5)], got %v", dataset.X[0])
	}

	// Sample 3 (target index 5):
	// target count = 6 => Y[3] = log(6.5)
	// features: lag 1 (index 4 => 5), lag 2 (index 3 => 4)
	expectedY3 := math.Log(6.5)
	if math.Abs(dataset.Y[3]-expectedY3) > 1e-6 {
		t.Errorf("sample 3: expected Y=%f, got %f", expectedY3, dataset.Y[3])
	}
	if math.Abs(dataset.X[3][0]-math.Log(5.5)) > 1e-6 || math.Abs(dataset.X[3][1]-math.Log(4.5)) > 1e-6 {
		t.Errorf("sample 3: expected X=[log(5.5), log(4.5)], got %v", dataset.X[3])
	}
}
