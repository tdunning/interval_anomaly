// Copyright (c) 2026 Ted Dunning
// SPDX-License-Identifier: MIT

package detector

import (
	"math"
	"testing"

	"interval_anomaly/pkg/model"
	"interval_anomaly/pkg/timeseries"
)

func TestStreamDetector(t *testing.T) {
	// Constant expected count of 4.5 per 10s bucket, i.e. a rate of 0.45/s.
	m := &model.ModelOutput{
		BucketInterval: 10.0,
		Horizon:        2,
		Epsilon:        0.5,
		Intercept:      math.Log(4.5),
		Weights:        []float64{0.0, 0.0},
	}

	// n = 2 differences (t[i] - t[i-2])
	det, err := NewStreamDetector(m, 2)
	if err != nil {
		t.Fatalf("NewStreamDetector failed: %v", err)
	}

	// Event 0 at t = 100.0
	r0, err := det.ProcessEvent(timeseries.Event{Seconds: 100.0, RawText: "100.0"})
	if err != nil || r0 != nil {
		t.Fatalf("expected r0 to be nil during warmup, got %v", r0)
	}

	// Event 1 at t = 102.0
	r1, err := det.ProcessEvent(timeseries.Event{Seconds: 102.0, RawText: "102.0"})
	if err != nil || r1 != nil {
		t.Fatalf("expected r1 to be nil during warmup, got %v", r1)
	}

	// Event 2 at t = 105.0 -> 2nd order diff = 105.0 - 100.0 = 5.0
	// Expected rate = 4.5 / 10 = 0.45 events per second
	// Anomaly statistic = diff * rate / n = 5.0 * 0.45 / 2 = 1.125
	r2, err := det.ProcessEvent(timeseries.Event{Seconds: 105.0, RawText: "105.0"})
	if err != nil || r2 == nil {
		t.Fatalf("expected valid record for r2, got err=%v, r2=%v", err, r2)
	}

	if math.Abs(r2.NthOrderDiff-5.0) > 1e-4 {
		t.Errorf("expected nth_order_diff 5.0, got %f", r2.NthOrderDiff)
	}
	if math.Abs(r2.ExpectedRate-0.45) > 1e-4 {
		t.Errorf("expected expected_rate 0.45, got %f", r2.ExpectedRate)
	}
	if math.Abs(r2.AnomalyStatistic-1.125) > 1e-4 {
		t.Errorf("expected anomaly_statistic 1.125, got %f", r2.AnomalyStatistic)
	}

	// Event 3 at t = 107.0 -> diff = 107.0 - 102.0 = 5.0
	r3, err := det.ProcessEvent(timeseries.Event{Seconds: 107.0, RawText: "107.0"})
	if err != nil || r3 == nil {
		t.Fatalf("expected valid record for r3, got err=%v, r3=%v", err, r3)
	}
	if math.Abs(r3.NthOrderDiff-5.0) > 1e-4 {
		t.Errorf("expected nth_order_diff 5.0, got %f", r3.NthOrderDiff)
	}
}

func TestStreamDetectorAveragesOverlappingBucketRates(t *testing.T) {
	// The regression is evaluated at the end of the first bucket, after its
	// single event enters the lag window, giving a count of 2 per 10s bucket.
	// Both halves of the interval [8, 12] therefore use a rate of 0.2/s.
	m := &model.ModelOutput{
		BucketInterval: 10.0,
		Horizon:        1,
		Epsilon:        1.0,
		Weights:        []float64{1.0},
	}
	det, err := NewStreamDetector(m, 1)
	if err != nil {
		t.Fatalf("NewStreamDetector failed: %v", err)
	}

	if _, err := det.ProcessEvent(timeseries.Event{Seconds: 8.0}); err != nil {
		t.Fatalf("failed to process first event: %v", err)
	}
	record, err := det.ProcessEvent(timeseries.Event{Seconds: 12.0})
	if err != nil || record == nil {
		t.Fatalf("expected record at t=12, got err=%v record=%v", err, record)
	}
	if math.Abs(record.ExpectedRate-0.2) > 1e-4 {
		t.Errorf("expected duration-weighted rate 0.2, got %f", record.ExpectedRate)
	}
}
