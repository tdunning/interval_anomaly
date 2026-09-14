// Copyright (c) 2026 Ted Dunning
// SPDX-License-Identifier: MIT

package main

import (
	"bufio"
	"encoding/csv"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"math"
	"os"
	"strconv"
	"strings"
	"time"

	"interval_anomaly/pkg/lasso"
	"interval_anomaly/pkg/model"
	"interval_anomaly/pkg/timeseries"
)

func main() {
	configPath := flag.String("config", "config.json", "Path to JSON configuration file (required)")
	inputPath := flag.String("input", "", "Path to input events file (defaults to stdin)")
	outputPath := flag.String("output", "", "Path to output model JSON file (defaults to stdout)")
	diagnosticPath := flag.String("diagnostic", "", "Path to diagnostic CSV with actual and predicted bucket rates")
	wikiFormat := flag.Bool("wiki", false, "Read hourly pagecounts rows (date,time,term,views,...) instead of event times")
	colIndex := flag.Int("col", 0, "Column index (0-based) for timestamp in CSV data")
	flag.Parse()

	if *configPath == "" {
		_, _ = fmt.Fprintf(os.Stderr, "Error: -config parameter is required\n")
		flag.Usage()
		os.Exit(1)
	}

	_, _ = fmt.Fprintf(os.Stderr, "=== Historical Event Analysis & Autoregressive Fit ===\n")
	_, _ = fmt.Fprintf(os.Stderr, "Loading configuration from: %s\n", *configPath)

	cfg, err := model.LoadConfig(*configPath)
	if err != nil {
		_, _ = fmt.Fprintf(os.Stderr, "Error loading configuration: %v\n", err)
		os.Exit(1)
	}

	if *wikiFormat {
		cfg.BucketInterval = 3600.0
		_, _ = fmt.Fprintf(os.Stderr, "Wiki format selected: bucket_interval forced to 3600s (1 hour)\n")
	}

	_, _ = fmt.Fprintf(os.Stderr, "Config loaded: bucket_interval=%.3fs, horizon=%d, lambda=%.4e, epsilon=%.4e\n",
		cfg.BucketInterval, cfg.Horizon, cfg.RegularizationPenalty, cfg.Epsilon)

	// Open input
	var reader io.Reader
	if *inputPath == "" || *inputPath == "-" {
		_, _ = fmt.Fprintf(os.Stderr, "Reading event data from stdin...\n")
		reader = os.Stdin
	} else {
		_, _ = fmt.Fprintf(os.Stderr, "Reading event data from file: %s...\n", *inputPath)
		file, err := os.Open(*inputPath)
		if err != nil {
			_, _ = fmt.Fprintf(os.Stderr, "Error opening input file: %v\n", err)
			os.Exit(1)
		}
		defer file.Close()
		reader = file
	}

	// Parse timestamps
	var bucketed *timeseries.BucketedSeries
	var numEvents int

	if *wikiFormat {
		_, _ = fmt.Fprintf(os.Stderr, "Reading hourly wiki counts...\n")
		bucketed, err = readWikiCounts(reader)
		if err != nil {
			_, _ = fmt.Fprintf(os.Stderr, "Error reading wiki counts: %v\n", err)
			os.Exit(1)
		}
		numEvents = bucketed.TotalEvents
		_, _ = fmt.Fprintf(os.Stderr, "Parsed %d views into %d hourly buckets\n", numEvents, len(bucketed.Counts))
	} else {
		scanner := bufio.NewScanner(reader)
		var events []timeseries.Event
		lineNum := 0
		skippedHeader := 0

		for scanner.Scan() {
			lineNum++
			line := scanner.Text()
			ev, err := timeseries.ExtractTimestampFromLine(line, *colIndex)
			if err != nil {
				// If line 1 fails, it might be a CSV header
				if lineNum == 1 {
					skippedHeader++
					continue
				}
				// Skip empty or unparseable lines with warning if small amount
				continue
			}
			events = append(events, ev)
		}

		if err := scanner.Err(); err != nil {
			_, _ = fmt.Fprintf(os.Stderr, "Error reading input: %v\n", err)
			os.Exit(1)
		}

		if len(events) == 0 {
			_, _ = fmt.Fprintf(os.Stderr, "Error: no valid events parsed from input\n")
			os.Exit(1)
		}

		numEvents = len(events)
		_, _ = fmt.Fprintf(os.Stderr, "Parsed %d events across %d lines (skipped %d header/invalid lines)\n",
			len(events), lineNum, skippedHeader)

		// Bucket events
		_, _ = fmt.Fprintf(os.Stderr, "Bucketing events with interval %.3f seconds...\n", cfg.BucketInterval)
		bucketed, err = timeseries.BucketEvents(events, cfg.BucketInterval)
		if err != nil {
			_, _ = fmt.Fprintf(os.Stderr, "Error bucketing events: %v\n", err)
			os.Exit(1)
		}
	}

	_, _ = fmt.Fprintf(os.Stderr, "Bucketing complete: %d buckets spanned from t=%.3f to t=%.3f\n",
		len(bucketed.Counts), bucketed.StartTime, bucketed.StartTime+float64(len(bucketed.Counts))*cfg.BucketInterval)
	if len(bucketed.Counts) > 0 {
		bucketed.Counts = bucketed.Counts[:len(bucketed.Counts)-1]
		_, _ = fmt.Fprintf(os.Stderr, "Omitting final bucket; %d buckets remain for training\n", len(bucketed.Counts))
	}

	// Build AR dataset
	_, _ = fmt.Fprintf(os.Stderr, "Constructing autoregressive dataset (horizon=%d, epsilon=%.4e)...\n",
		cfg.Horizon, cfg.Epsilon)
	dataset, err := bucketed.BuildARDataset(cfg.Horizon, cfg.Epsilon)
	if err != nil {
		_, _ = fmt.Fprintf(os.Stderr, "Error constructing AR dataset: %v\n", err)
		os.Exit(1)
	}

	nSamples := len(dataset.Y)
	_, _ = fmt.Fprintf(os.Stderr, "Dataset constructed: %d training samples with %d lag features each\n",
		nSamples, cfg.Horizon)

	// Lasso regression
	lassoCfg := lasso.Config{
		Lambda:        cfg.RegularizationPenalty,
		MaxIterations: cfg.MaxIterations,
		Tolerance:     cfg.Tolerance,
		FitIntercept:  cfg.FitIntercept,
	}

	_, _ = fmt.Fprintf(os.Stderr, "Starting L1 regularized coordinate descent (max_iter=%d, tol=%.1e, lambda=%.4e)...\n",
		lassoCfg.MaxIterations, lassoCfg.Tolerance, lassoCfg.Lambda)

	progressCb := func(iter int, maxDelta float64, mse float64, nonZero int) {
		_, _ = fmt.Fprintf(os.Stderr, "  [Iter %4d] max_delta=%.6e  MSE=%.6f  active_features=%d/%d\n",
			iter, maxDelta, mse, nonZero, cfg.Horizon)
	}

	fitRes, err := lasso.Fit(dataset.X, dataset.Y, lassoCfg, progressCb)
	if err != nil {
		_, _ = fmt.Fprintf(os.Stderr, "Error fitting L1 regression: %v\n", err)
		os.Exit(1)
	}

	if fitRes.Converged {
		_, _ = fmt.Fprintf(os.Stderr, "Optimization CONVERGED after %d iterations! Final MSE=%.6f, Non-zero weights=%d/%d\n",
			fitRes.Iterations, fitRes.MSE, fitRes.NonZeroWeights, cfg.Horizon)
	} else {
		_, _ = fmt.Fprintf(os.Stderr, "Optimization reached max iterations (%d). Final MSE=%.6f, Non-zero weights=%d/%d\n",
			fitRes.Iterations, fitRes.MSE, fitRes.NonZeroWeights, cfg.Horizon)
	}

	if *diagnosticPath != "" && *diagnosticPath != "-" {
		if err := writeDiagnosticCSV(*diagnosticPath, bucketed, dataset, fitRes.Model); err != nil {
			_, _ = fmt.Fprintf(os.Stderr, "Error writing diagnostic CSV to %s: %v\n", *diagnosticPath, err)
			os.Exit(1)
		}
		_, _ = fmt.Fprintf(os.Stderr, "Diagnostic bucket rates successfully written to %s\n", *diagnosticPath)
	}

	// Prepare Model Output JSON
	modelOutput := model.ModelOutput{
		BucketInterval:        timeseries.RoundToMillis(cfg.BucketInterval),
		Horizon:               cfg.Horizon,
		Epsilon:               cfg.Epsilon,
		RegularizationPenalty: cfg.RegularizationPenalty,
		Intercept:             fitRes.Model.Intercept,
		Weights:               fitRes.Model.Weights,
		TrainingStats: &model.TrainingStats{
			NumEvents:      numEvents,
			NumBuckets:     len(bucketed.Counts),
			FirstEventTime: bucketed.StartTime,
			LastEventTime:  bucketed.StartTime + float64(len(bucketed.Counts))*cfg.BucketInterval,
			Iterations:     fitRes.Iterations,
			Converged:      fitRes.Converged,
			MSE:            fitRes.MSE,
			NonZeroWeights: fitRes.NonZeroWeights,
		},
	}

	jsonBytes, err := json.MarshalIndent(modelOutput, "", "  ")
	if err != nil {
		_, _ = fmt.Fprintf(os.Stderr, "Error serializing model to JSON: %v\n", err)
		os.Exit(1)
	}

	if *outputPath != "" && *outputPath != "-" {
		err = os.WriteFile(*outputPath, jsonBytes, 0644)
		if err != nil {
			_, _ = fmt.Fprintf(os.Stderr, "Error writing model to %s: %v\n", *outputPath, err)
			os.Exit(1)
		}
		_, _ = fmt.Fprintf(os.Stderr, "Model parameters successfully written to %s\n", *outputPath)
	} else {
		fmt.Println(string(jsonBytes))
	}
}

// readWikiCounts aggregates hourly pagecounts rows (date,time,term,views,...) into one bucket per hour.
func readWikiCounts(reader io.Reader) (*timeseries.BucketedSeries, error) {
	csvReader := csv.NewReader(reader)
	csvReader.FieldsPerRecord = -1

	countsByHour := make(map[int64]float64)
	var minHour, maxHour int64
	total := 0.0
	seen := false

	for {
		record, err := csvReader.Read()
		if err == io.EOF {
			break
		}
		if err != nil {
			return nil, err
		}
		if len(record) < 4 {
			continue
		}

		stamp, err := time.Parse("20060102150405", strings.TrimSpace(record[0])+strings.TrimSpace(record[1]))
		if err != nil {
			continue // header or malformed row
		}
		views, err := strconv.ParseFloat(strings.TrimSpace(record[3]), 64)
		if err != nil {
			continue
		}

		hour := stamp.Unix() / 3600
		countsByHour[hour] += views
		total += views
		if !seen || hour < minHour {
			minHour = hour
		}
		if !seen || hour > maxHour {
			maxHour = hour
		}
		seen = true
	}

	if !seen {
		return nil, fmt.Errorf("no valid hourly count rows parsed from input")
	}

	counts := make([]float64, maxHour-minHour+1)
	for hour, count := range countsByHour {
		counts[hour-minHour] = count
	}

	return &timeseries.BucketedSeries{
		StartTime:      float64(minHour * 3600),
		BucketInterval: 3600.0,
		Counts:         counts,
		TotalEvents:    int(total),
	}, nil
}

func writeDiagnosticCSV(path string, bucketed *timeseries.BucketedSeries, dataset *timeseries.ARDataset, fittedModel lasso.Model) error {
	file, err := os.Create(path)
	if err != nil {
		return err
	}
	defer file.Close()

	writer := csv.NewWriter(file)
	if err := writer.Write([]string{"bucket_start_time", "actual_rate", "predicted_rate"}); err != nil {
		return err
	}

	for bucketIndex := dataset.Horizon; bucketIndex < len(bucketed.Counts); bucketIndex++ {
		count := bucketed.Counts[bucketIndex]
		prediction := fittedModel.Predict(dataset.X[bucketIndex-dataset.Horizon])
		row := []string{
			strconv.FormatFloat(bucketed.StartTime+float64(bucketIndex)*bucketed.BucketInterval, 'f', 6, 64),
			strconv.FormatFloat(count/bucketed.BucketInterval, 'f', 6, 64),
			strconv.FormatFloat(math.Exp(prediction)/bucketed.BucketInterval, 'f', 6, 64),
		}
		if err := writer.Write(row); err != nil {
			return err
		}
	}

	writer.Flush()
	return writer.Error()
}
