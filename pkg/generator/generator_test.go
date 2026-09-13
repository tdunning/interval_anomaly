// Copyright (c) 2026 Ted Dunning
// SPDX-License-Identifier: MIT

package generator

import (
	"math"
	"testing"
	"time"
)

func TestKFunction(t *testing.T) {
	kScale := 500.0
	kOffset := 100.0

	// At t = 0, sin(0) = 0 => exp(0) = 1
	// k(0) = 500 * (exp(2) - 1) + 100
	expectedK0 := 500.0*(math.Exp(2.0)-1.0) + 100.0
	k0 := K(0, kScale, kOffset)
	if math.Abs(k0-expectedK0) > 1e-6 {
		t.Errorf("K(0) = %f, want %f", k0, expectedK0)
	}

	// Over a week period, check that K is always positive
	for tSec := 0.0; tSec <= WeekInSeconds; tSec += 3600.0 {
		val := K(tSec, kScale, kOffset)
		if val <= 0 {
			t.Errorf("K(%f) = %f, expected positive", tSec, val)
		}
	}
}

func TestRatePerHour(t *testing.T) {
	kScale := 500.0
	kOffset := 100.0
	offset := 0.0

	r0 := RatePerHour(0, kScale, kOffset, offset)
	if r0 <= 0 {
		t.Fatalf("RatePerHour(0) = %f, expected positive", r0)
	}

	// Check that rate per second is rate per hour / 3600
	rSec := RatePerSecond(0, kScale, kOffset, offset)
	if math.Abs(rSec-(r0/HourInSeconds)) > 1e-9 {
		t.Errorf("RatePerSecond mismatch: %f vs %f", rSec, r0/HourInSeconds)
	}
}

func TestGenerateEventsAndSplit(t *testing.T) {
	cfg := GeneratorConfig{
		KScale:        500.0,
		KOffset:       100.0,
		Offset:        0.0,
		TotalDuration: 2 * 24 * time.Hour, // 2 days for quick test
		BaseTime:      time.Date(2025, 1, 1, 0, 0, 0, 0, time.UTC),
		TimeFormat:    "iso",
		Seed:          42,
	}

	events, err := GenerateEvents(cfg)
	if err != nil {
		t.Fatalf("GenerateEvents failed: %v", err)
	}

	if len(events) == 0 {
		t.Fatalf("expected non-zero events")
	}

	// Verify events are in non-decreasing chronological order
	for i := 1; i < len(events); i++ {
		if events[i].Seconds < events[i-1].Seconds {
			t.Errorf("events not in chronological order at index %d: %f < %f",
				i, events[i].Seconds, events[i-1].Seconds)
		}
	}

	// Test splitting
	events90, events10 := SplitEvents(events, 0.90)
	if len(events90)+len(events10) != len(events) {
		t.Errorf("split lengths sum %d != total %d", len(events90)+len(events10), len(events))
	}

	expected90Len := int(math.Round(float64(len(events)) * 0.90))
	if len(events90) != expected90Len {
		t.Errorf("expected 90%% count %d, got %d", expected90Len, len(events90))
	}

	// Test Epoch format
	cfgEpoch := cfg
	cfgEpoch.TimeFormat = "epoch"
	eventsEpoch, err := GenerateEvents(cfgEpoch)
	if err != nil {
		t.Fatalf("GenerateEvents with epoch format failed: %v", err)
	}
	if len(eventsEpoch) == 0 {
		t.Fatalf("expected non-zero events for epoch format")
	}
}
