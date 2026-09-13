// Copyright (c) 2026 Ted Dunning
// SPDX-License-Identifier: MIT

package test

import (
	"encoding/csv"
	"fmt"
	"math/rand"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"testing"
	"time"
)

func TestEndToEndPipeline(t *testing.T) {
	tmpDir := t.TempDir()

	// 1. Create a config JSON
	configFile := filepath.Join(tmpDir, "config.json")
	configJSON := `{
		"bucket_interval": 10.0,
		"horizon": 5,
		"regularization_penalty": 0.01,
		"epsilon": 1.0,
		"max_iterations": 1000,
		"tolerance": 1e-6
	}`
	if err := os.WriteFile(configFile, []byte(configJSON), 0644); err != nil {
		t.Fatalf("failed to write config: %v", err)
	}

	// 2. Generate synthetic training event data with ISO and Epoch formats
	trainDataFile := filepath.Join(tmpDir, "train_events.csv")
	var trainLines []string
	trainLines = append(trainLines, "timestamp")

	baseTime := time.Date(2023, 1, 1, 0, 0, 0, 0, time.UTC)
	currentTime := baseTime

	rng := rand.New(rand.NewSource(42))
	// 50 buckets of 10s = 500s total duration
	for bucket := 0; bucket < 50; bucket++ {
		// Event rate fluctuates between 2 and 10 events per bucket
		rate := 3 + (bucket%5)*2
		for ev := 0; ev < rate; ev++ {
			offset := float64(bucket)*10.0 + rng.Float64()*10.0
			eventTime := baseTime.Add(time.Duration(offset * float64(time.Second)))
			if bucket%2 == 0 {
				trainLines = append(trainLines, eventTime.Format(time.RFC3339Nano))
			} else {
				trainLines = append(trainLines, fmt.Sprintf("%.4f", float64(eventTime.UnixNano())/1e9))
			}
		}
	}

	trainData := ""
	for _, l := range trainLines {
		trainData += l + "\n"
	}
	if err := os.WriteFile(trainDataFile, []byte(trainData), 0644); err != nil {
		t.Fatalf("failed to write train events: %v", err)
	}

	// 3. Run historical training CLI
	modelFile := filepath.Join(tmpDir, "model.json")
	cmdTrain := exec.Command("go", "run", "../cmd/historical", "-config", configFile, "-input", trainDataFile, "-output", modelFile)
	outTrain, err := cmdTrain.CombinedOutput()
	if err != nil {
		t.Fatalf("cmd/historical failed: %v\nOutput: %s", err, string(outTrain))
	}
	t.Logf("Historical trainer output:\n%s", string(outTrain))

	if _, err := os.Stat(modelFile); os.IsNotExist(err) {
		t.Fatalf("model.json was not created")
	}

	// 4. Generate test/anomaly event data
	testDataFile := filepath.Join(tmpDir, "test_events.csv")
	var testLines []string
	testLines = append(testLines, "time")
	currentTime = baseTime.Add(1000 * time.Second)

	// Stream 20 normal events with ~2s intervals
	for i := 0; i < 20; i++ {
		currentTime = currentTime.Add(time.Duration((1.8 + rng.Float64()*0.4) * float64(time.Second)))
		testLines = append(testLines, currentTime.Format(time.RFC3339))
	}
	// Inject an anomaly: a gap of 60 seconds
	currentTime = currentTime.Add(60 * time.Second)
	testLines = append(testLines, currentTime.Format(time.RFC3339))

	testData := ""
	for _, l := range testLines {
		testData += l + "\n"
	}
	if err := os.WriteFile(testDataFile, []byte(testData), 0644); err != nil {
		t.Fatalf("failed to write test events: %v", err)
	}

	// 5. Run anomaly detector CLI with -n 1
	anomalyOutputFile := filepath.Join(tmpDir, "anomalies.csv")
	cmdAnomaly := exec.Command("go", "run", "../cmd/anomaly", "-model", modelFile, "-n", "1", "-input", testDataFile, "-output", anomalyOutputFile)
	outAnomaly, err := cmdAnomaly.CombinedOutput()
	if err != nil {
		t.Fatalf("cmd/anomaly failed: %v\nOutput: %s", err, string(outAnomaly))
	}
	t.Logf("Anomaly detector output:\n%s", string(outAnomaly))

	// 6. Read and verify anomaly CSV output
	outFile, err := os.Open(anomalyOutputFile)
	if err != nil {
		t.Fatalf("failed to open anomaly output: %v", err)
	}
	defer outFile.Close()

	csvReader := csv.NewReader(outFile)
	records, err := csvReader.ReadAll()
	if err != nil {
		t.Fatalf("failed to read anomaly CSV: %v", err)
	}

	if len(records) < 20 {
		t.Fatalf("expected at least 20 records, got %d", len(records))
	}

	// Header verification
	if records[0][0] != "time" || records[0][1] != "nth_order_diff" || records[0][2] != "expected_rate" || records[0][3] != "anomaly_statistic" {
		t.Fatalf("unexpected CSV headers: %v", records[0])
	}

	// Verify the last record (the anomaly with 60s gap) has a significantly higher anomaly statistic
	lastRecord := records[len(records)-1]
	normalRecord := records[len(records)-2]

	lastDiff, _ := strconv.ParseFloat(lastRecord[1], 64)
	normalDiff, _ := strconv.ParseFloat(normalRecord[1], 64)
	lastStat, _ := strconv.ParseFloat(lastRecord[3], 64)
	normalStat, _ := strconv.ParseFloat(normalRecord[3], 64)

	if lastDiff < 50.0 {
		t.Errorf("expected last diff > 50s, got %f", lastDiff)
	}
	if normalDiff > 10.0 {
		t.Errorf("expected normal diff < 10s, got %f", normalDiff)
	}
	if lastStat <= normalStat {
		t.Errorf("expected anomaly stat (%f) to be larger than normal stat (%f)", lastStat, normalStat)
	}
}
