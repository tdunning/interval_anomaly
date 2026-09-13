// Copyright (c) 2026 Ted Dunning
// SPDX-License-Identifier: MIT

package main

import (
	"flag"
	"fmt"
	"os"
	"strconv"
	"strings"
	"time"

	"interval_anomaly/pkg/generator"
	"interval_anomaly/pkg/timeseries"
)

// parseDuration parses duration strings such as "30d", "720h", "3600s", or bare numbers as seconds.
func parseDuration(s string) (time.Duration, error) {
	s = strings.TrimSpace(s)
	if s == "" {
		return 0, fmt.Errorf("empty duration string")
	}

	// Support day suffix like "30d" or "7d"
	if strings.HasSuffix(strings.ToLower(s), "d") {
		daysStr := strings.TrimSuffix(strings.ToLower(s), "d")
		days, err := strconv.ParseFloat(daysStr, 64)
		if err != nil {
			return 0, fmt.Errorf("invalid day duration %q: %w", s, err)
		}
		return time.Duration(days * 24 * float64(time.Hour)), nil
	}

	// Try standard Go duration (e.g. "720h", "30m", "100s")
	if d, err := time.ParseDuration(s); err == nil {
		return d, nil
	}

	// Try bare float/int seconds
	if sec, err := strconv.ParseFloat(s, 64); err == nil {
		return time.Duration(sec * float64(time.Second)), nil
	}

	return 0, fmt.Errorf("unable to parse total duration %q (use e.g. '30d', '720h', or seconds)", s)
}

func writeEventFile(filePath string, events []timeseries.Event) error {
	file, err := os.Create(filePath)
	if err != nil {
		return fmt.Errorf("failed to create file %s: %w", filePath, err)
	}
	defer file.Close()

	for _, ev := range events {
		if _, err := fmt.Fprintln(file, ev.RawText); err != nil {
			return fmt.Errorf("failed to write to %s: %w", filePath, err)
		}
	}
	return nil
}

func main() {
	kScale := flag.Float64("k_scale", 500.0, "Scaling parameter k_scale in events per hour (default 500)")
	kOffset := flag.Float64("k_offset", 100.0, "Offset parameter k_offset in events per hour (default 100)")
	offset := flag.Float64("offset", 0.0, "Diurnal phase offset in days (default 0.0)")
	totalTimeStr := flag.String("total_time", "30d", "Total simulation time (e.g., '30d', '720h', '2592000s') (required)")
	timeFormat := flag.String("format", "iso", "Timestamp output format: 'iso' (RFC3339) or 'epoch' (seconds)")
	out90 := flag.String("out_90", "train_events.csv", "Output file for first 90% of event data (training)")
	out10 := flag.String("out_10", "test_events.csv", "Output file for last 10% of event data (test/anomaly)")
	splitRatio := flag.Float64("split", 0.90, "Split ratio for first file (default 0.90 for 90%/10%)")
	seed := flag.Int64("seed", 0, "Random seed (0 for non-deterministic)")
	flag.Parse()

	totalDuration, err := parseDuration(*totalTimeStr)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error parsing total_time: %v\n", err)
		flag.Usage()
		os.Exit(1)
	}

	cfg := generator.GeneratorConfig{
		KScale:        *kScale,
		KOffset:       *kOffset,
		Offset:        *offset,
		TotalDuration: totalDuration,
		BaseTime:      time.Date(2025, 1, 1, 0, 0, 0, 0, time.UTC),
		TimeFormat:    *timeFormat,
		Seed:          *seed,
	}

	fmt.Fprintf(os.Stderr, "=== Synthetic Event Generator ===\n")
	fmt.Fprintf(os.Stderr, "Parameters: k_scale=%.2f events/hr, k_offset=%.2f events/hr, offset=%.4f days\n",
		cfg.KScale, cfg.KOffset, cfg.Offset)
	fmt.Fprintf(os.Stderr, "Total simulation duration: %v (%.1f hours / %.1f days)\n",
		totalDuration, totalDuration.Hours(), totalDuration.Hours()/24.0)
	fmt.Fprintf(os.Stderr, "Format: %s, Split ratio: %.1f%% / %.1f%%\n",
		*timeFormat, *splitRatio*100.0, (1.0-*splitRatio)*100.0)

	events, err := generator.GenerateEvents(cfg)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error generating events: %v\n", err)
		os.Exit(1)
	}

	fmt.Fprintf(os.Stderr, "Generated %d synthetic events in total\n", len(events))

	events90, events10 := generator.SplitEvents(events, *splitRatio)
	fmt.Fprintf(os.Stderr, "Split: %d events in training set, %d events in test set\n",
		len(events90), len(events10))

	// Write 90% file
	if err := writeEventFile(*out90, events90); err != nil {
		fmt.Fprintf(os.Stderr, "Error writing 90%% output file: %v\n", err)
		os.Exit(1)
	}
	fmt.Fprintf(os.Stderr, "Wrote 90%% of events (%d) to: %s\n", len(events90), *out90)

	// Write 10% file
	if err := writeEventFile(*out10, events10); err != nil {
		fmt.Fprintf(os.Stderr, "Error writing 10%% output file: %v\n", err)
		os.Exit(1)
	}
	fmt.Fprintf(os.Stderr, "Wrote 10%% of events (%d) to: %s\n", len(events10), *out10)

	fmt.Fprintf(os.Stderr, "Done.\n")
}
