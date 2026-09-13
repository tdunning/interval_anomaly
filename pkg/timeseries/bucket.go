// Copyright (c) 2026 Ted Dunning
// SPDX-License-Identifier: MIT

package timeseries

import (
	"fmt"
	"math"
	"sort"
)

// BucketedSeries contains the bucketed event counts and time information.
type BucketedSeries struct {
	StartTime      float64
	BucketInterval float64
	Counts         []float64
	TotalEvents    int
}

// BucketEvents takes a slice of events, sorts them by timestamp, and counts
// the number of events falling into each constant time interval of bucketInterval seconds.
func BucketEvents(events []Event, bucketInterval float64) (*BucketedSeries, error) {
	if len(events) == 0 {
		return nil, fmt.Errorf("no events to bucket")
	}
	if bucketInterval <= 0 {
		return nil, fmt.Errorf("bucket interval must be positive, got %f", bucketInterval)
	}

	// Make a copy and sort by timestamp
	sorted := make([]Event, len(events))
	copy(sorted, events)
	sort.Slice(sorted, func(i, j int) bool {
		return sorted[i].Seconds < sorted[j].Seconds
	})

	startTime := sorted[0].Seconds
	endTime := sorted[len(sorted)-1].Seconds

	numBuckets := int(math.Floor((endTime-startTime)/bucketInterval)) + 1
	counts := make([]float64, numBuckets)

	for _, ev := range sorted {
		idx := int(math.Floor((ev.Seconds - startTime) / bucketInterval))
		if idx < 0 {
			idx = 0
		}
		if idx >= numBuckets {
			idx = numBuckets - 1
		}
		counts[idx]++
	}

	return &BucketedSeries{
		StartTime:      startTime,
		BucketInterval: bucketInterval,
		Counts:         counts,
		TotalEvents:    len(events),
	}, nil
}

// ARDataset holds design matrix X and target vector y for autoregressive linear regression.
// Each row i in X has features [c_{t-1}, c_{t-2}, ..., c_{t-p}], and y[i] = log(c_t + epsilon).
type ARDataset struct {
	X       [][]float64 // K samples x p features
	Y       []float64   // K target values
	Horizon int
	Epsilon float64
}

// BuildARDataset constructs the AR(p) feature matrix X and target y from the bucketed series.
// p is the horizon (number of lag buckets).
// epsilon is added to the count before taking the natural logarithm: log(count + epsilon).
func (b *BucketedSeries) BuildARDataset(horizon int, epsilon float64) (*ARDataset, error) {
	if horizon <= 0 {
		return nil, fmt.Errorf("horizon must be positive, got %d", horizon)
	}
	if epsilon <= 0 {
		return nil, fmt.Errorf("epsilon must be positive, got %f", epsilon)
	}
	if len(b.Counts) <= horizon {
		return nil, fmt.Errorf("number of buckets (%d) must be greater than horizon (%d)", len(b.Counts), horizon)
	}

	nSamples := len(b.Counts) - horizon
	X := make([][]float64, nSamples)
	Y := make([]float64, nSamples)

	for i := 0; i < nSamples; i++ {
		targetIdx := horizon + i
		X[i] = make([]float64, horizon)
		// Lags: lag 1 (targetIdx - 1), lag 2 (targetIdx - 2), ..., lag p (targetIdx - horizon)
		for j := 0; j < horizon; j++ {
			lagIdx := targetIdx - 1 - j
			X[i][j] = math.Log(b.Counts[lagIdx] + epsilon)
		}
		Y[i] = math.Log(b.Counts[targetIdx] + epsilon)
	}

	return &ARDataset{
		X:       X,
		Y:       Y,
		Horizon: horizon,
		Epsilon: epsilon,
	}, nil
}
