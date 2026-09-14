// Copyright (c) 2026 Ted Dunning
// SPDX-License-Identifier: MIT

package detector

import (
	"fmt"
	"math"

	"interval_anomaly/pkg/model"
	"interval_anomaly/pkg/timeseries"
)

// AnomalyRecord represents the output record for an event anomaly evaluation.
type AnomalyRecord struct {
	RawTime          string  `json:"time"`
	TimeSeconds      float64 `json:"time_seconds"`
	NthOrderDiff     float64 `json:"nth_order_diff"`
	ExpectedRate     float64 `json:"expected_rate"`
	AnomalyStatistic float64 `json:"anomaly_statistic"`
}

// StreamDetector processes a stream of events online or in batch, maintaining
// bucket counts and rolling event history to compute n-th order differences and anomaly statistics.
type StreamDetector struct {
	model         *model.ModelOutput
	n             int                // order of time difference
	recentEvents  []timeseries.Event // ring buffer / window for n-th order diff
	bucketHistory []float64          // past p completed bucket counts (most recent lag first)
	bucketRates   map[int64]float64  // estimated rate for each retained bucket

	initialized bool
	baseBucket  int64
	currBucket  int64
	currCount   float64
	minRate     float64
}

// NewStreamDetector creates a new StreamDetector for a given model and difference order n.
func NewStreamDetector(m *model.ModelOutput, n int) (*StreamDetector, error) {
	if m == nil {
		return nil, fmt.Errorf("model cannot be nil")
	}
	if n <= 0 {
		return nil, fmt.Errorf("difference order n must be positive, got %d", n)
	}

	return &StreamDetector{
		model:         m,
		n:             n,
		recentEvents:  make([]timeseries.Event, 0, n),
		bucketHistory: make([]float64, m.Horizon), // initialized with zeros
		bucketRates:   make(map[int64]float64),
		minRate:       1e-9,
	}, nil
}

// getBucketIndex returns the bucket index for a timestamp in seconds.
func (d *StreamDetector) getBucketIndex(sec float64) int64 {
	return int64(math.Floor(sec / d.model.BucketInterval))
}

// advanceBuckets shifts the bucket window when the event timestamp enters a new bucket.
func (d *StreamDetector) advanceBuckets(newBucket int64) {
	if !d.initialized {
		d.currBucket = newBucket
		d.currCount = 0
		d.initialized = true
		return
	}

	if newBucket <= d.currBucket {
		return
	}

	// For all buckets from currBucket up to newBucket-1, record their counts into history
	// Bucket currBucket gets currCount, and any skipped buckets get 0 count.
	d.pushBucketToHistory(d.currCount)
	d.bucketRates[d.currBucket] = d.model.PredictRate(d.bucketHistory)
	for b := d.currBucket + 1; b < newBucket; b++ {
		d.pushBucketToHistory(0.0)
		d.bucketRates[b] = d.model.PredictRate(d.bucketHistory)
	}

	d.currBucket = newBucket
	d.currCount = 0
}

// pushBucketToHistory pushes a completed bucket count into bucketHistory (lags).
// bucketHistory[0] is lag 1 (most recent), bucketHistory[p-1] is lag p (oldest).
func (d *StreamDetector) pushBucketToHistory(count float64) {
	// Shift elements right: [c_{t-1}, c_{t-2}, ..., c_{t-p}]
	for i := len(d.bucketHistory) - 1; i > 0; i-- {
		d.bucketHistory[i] = d.bucketHistory[i-1]
	}
	if len(d.bucketHistory) > 0 {
		d.bucketHistory[0] = count
	}
}

// expectedRateForInterval returns the duration-weighted average bucket estimate
// over the interval from startSeconds through endSeconds.
func (d *StreamDetector) expectedRateForInterval(startSeconds, endSeconds float64) float64 {
	if endSeconds <= startSeconds {
		return d.bucketRates[d.currBucket]
	}

	startBucket := d.getBucketIndex(startSeconds)
	endBucket := d.getBucketIndex(endSeconds)
	weightedRate := 0.0
	for bucket := startBucket; bucket <= endBucket; bucket++ {
		bucketStart := float64(bucket) * d.model.BucketInterval
		bucketEnd := bucketStart + d.model.BucketInterval
		overlapStart := math.Max(startSeconds, bucketStart)
		overlapEnd := math.Min(endSeconds, bucketEnd)
		if overlapEnd > overlapStart {
			rate, ok := d.bucketRates[bucket]
			if !ok {
				rate = d.model.PredictRate(d.bucketHistory)
			}
			weightedRate += rate * (overlapEnd - overlapStart)
		}
	}

	return weightedRate / (endSeconds - startSeconds)
}

func (d *StreamDetector) discardExpiredBucketRates() {
	if len(d.recentEvents) == 0 {
		return
	}
	firstRetainedBucket := d.getBucketIndex(d.recentEvents[0].Seconds)
	for bucket := range d.bucketRates {
		if bucket < firstRetainedBucket {
			delete(d.bucketRates, bucket)
		}
	}
}

// ProcessEvent processes an incoming event. If at least n previous events have been seen,
// it returns an AnomalyRecord; otherwise it returns nil (warming up the n-th order buffer).
func (d *StreamDetector) ProcessEvent(ev timeseries.Event) (*AnomalyRecord, error) {
	targetBucket := d.getBucketIndex(ev.Seconds)
	d.advanceBuckets(targetBucket)

	var record *AnomalyRecord

	if len(d.recentEvents) == d.n {
		prevEv := d.recentEvents[0]
		diff := ev.Seconds - prevEv.Seconds
		if diff < 0 {
			diff = 0 // Safeguard if timestamps arrive slightly out-of-order
		}
		expectedRate := d.expectedRateForInterval(prevEv.Seconds, ev.Seconds)

		// Calculate anomaly statistic: actual time difference divided by expected rate
		rateForDiv := expectedRate
		if rateForDiv < d.minRate {
			// Avoid division by zero when expected count is zero
			if d.model.Epsilon > 0 {
				rateForDiv = d.model.Epsilon
			} else {
				rateForDiv = d.minRate
			}
		}

		stat := diff * rateForDiv / float64(d.n)

		timeStr := ev.RawText
		if timeStr == "" {
			if ev.IsEpoch {
				timeStr = timeseries.FormatEpochWithMillis(ev.Seconds)
			} else {
				timeStr = timeseries.FormatISOWithMillis(ev.Timestamp)
			}
		}

		record = &AnomalyRecord{
			RawTime:          timeStr,
			TimeSeconds:      ev.Seconds,
			NthOrderDiff:     diff,
			ExpectedRate:     expectedRate,
			AnomalyStatistic: stat,
		}

		// Shift recent events window
		d.recentEvents = append(d.recentEvents[1:], ev)
		d.discardExpiredBucketRates()
	} else {
		// Window not full yet
		d.recentEvents = append(d.recentEvents, ev)
	}

	// Increment event count in current bucket
	d.currCount++

	return record, nil
}
