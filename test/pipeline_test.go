package test

import (
	"encoding/csv"
	"os"
	"os/exec"
	"path/filepath"
	"testing"
)

func TestFullSyntheticPipeline(t *testing.T) {
	tmpDir := t.TempDir()

	trainEvents := filepath.Join(tmpDir, "train_events.csv")
	testEvents := filepath.Join(tmpDir, "test_events.csv")
	configFile := filepath.Join(tmpDir, "config.json")
	modelFile := filepath.Join(tmpDir, "model.json")
	diagnosticFile := filepath.Join(tmpDir, "rates.csv")
	anomalyOutput := filepath.Join(tmpDir, "anomalies.csv")

	// 1. Run generator: 7 days of synthetic data
	cmdGen := exec.Command("go", "run", "../cmd/generate",
		"-k_scale", "500",
		"-k_offset", "100",
		"-total_time", "7d",
		"-format", "iso",
		"-out_90", trainEvents,
		"-out_10", testEvents,
		"-seed", "42",
	)
	outGen, err := cmdGen.CombinedOutput()
	if err != nil {
		t.Fatalf("cmd/generate failed: %v\nOutput: %s", err, string(outGen))
	}
	t.Logf("Generator output:\n%s", string(outGen))

	// 2. Create config JSON
	configJSON := `{
		"bucket_interval": 3600.0,
		"horizon": 24,
		"regularization_penalty": 0.001,
		"epsilon": 1.0,
		"max_iterations": 1000,
		"tolerance": 1e-6
	}`
	if err := os.WriteFile(configFile, []byte(configJSON), 0644); err != nil {
		t.Fatalf("failed to write config: %v", err)
	}

	// 3. Run historical training
	cmdTrain := exec.Command("go", "run", "../cmd/historical",
		"-config", configFile,
		"-input", trainEvents,
		"-output", modelFile,
		"-diagnostic", diagnosticFile,
	)
	outTrain, err := cmdTrain.CombinedOutput()
	if err != nil {
		t.Fatalf("cmd/historical failed: %v\nOutput: %s", err, string(outTrain))
	}
	t.Logf("Historical output:\n%s", string(outTrain))

	// 4. Verify diagnostic output CSV
	diagnosticHandle, err := os.Open(diagnosticFile)
	if err != nil {
		t.Fatalf("failed to open diagnostic output: %v", err)
	}
	defer diagnosticHandle.Close()

	diagnosticRows, err := csv.NewReader(diagnosticHandle).ReadAll()
	if err != nil {
		t.Fatalf("failed to read diagnostic output: %v", err)
	}
	if len(diagnosticRows) <= 25 {
		t.Fatalf("expected more than 25 diagnostic rows, got %d", len(diagnosticRows))
	}
	if got := diagnosticRows[0]; len(got) != 3 || got[0] != "bucket_start_time" || got[1] != "actual_rate" || got[2] != "predicted_rate" {
		t.Fatalf("unexpected diagnostic header: %v", got)
	}
	for rowIndex, row := range diagnosticRows[1:] {
		if row[2] == "" {
			t.Fatalf("expected predicted rate in diagnostic row %d", rowIndex+1)
		}
	}
	// 5. Run anomaly detection on test (10%) events
	cmdAnomaly := exec.Command("go", "run", "../cmd/anomaly",
		"-model", modelFile,
		"-n", "5",
		"-input", testEvents,
		"-output", anomalyOutput,
	)
	outAnomaly, err := cmdAnomaly.CombinedOutput()
	if err != nil {
		t.Fatalf("cmd/anomaly failed: %v\nOutput: %s", err, string(outAnomaly))
	}
	t.Logf("Anomaly output:\n%s", string(outAnomaly))

	// 5. Verify anomaly output CSV
	f, err := os.Open(anomalyOutput)
	if err != nil {
		t.Fatalf("failed to open anomaly output: %v", err)
	}
	defer f.Close()

	reader := csv.NewReader(f)
	rows, err := reader.ReadAll()
	if err != nil {
		t.Fatalf("failed to read anomaly rows: %v", err)
	}

	if len(rows) < 10 {
		t.Fatalf("expected more than 10 rows, got %d", len(rows))
	}

	if rows[0][0] != "time" || rows[0][1] != "nth_order_diff" || rows[0][2] != "expected_rate" || rows[0][3] != "anomaly_statistic" {
		t.Fatalf("unexpected header: %v", rows[0])
	}
}

func TestHistoricalOmitsFinalBucket(t *testing.T) {
	tmpDir := t.TempDir()
	inputFile := filepath.Join(tmpDir, "events.csv")
	configFile := filepath.Join(tmpDir, "config.json")
	modelFile := filepath.Join(tmpDir, "model.json")
	diagnosticFile := filepath.Join(tmpDir, "rates.csv")

	if err := os.WriteFile(inputFile, []byte("0\n10\n20\n30\n"), 0644); err != nil {
		t.Fatalf("failed to write input: %v", err)
	}
	configJSON := `{
		"bucket_interval": 10,
		"horizon": 2,
		"regularization_penalty": 0.01,
		"epsilon": 1,
		"max_iterations": 10
	}`
	if err := os.WriteFile(configFile, []byte(configJSON), 0644); err != nil {
		t.Fatalf("failed to write config: %v", err)
	}

	cmd := exec.Command("go", "run", "../cmd/historical",
		"-config", configFile,
		"-input", inputFile,
		"-output", modelFile,
		"-diagnostic", diagnosticFile,
	)
	if output, err := cmd.CombinedOutput(); err != nil {
		t.Fatalf("cmd/historical failed: %v\nOutput: %s", err, output)
	}

	diagnosticHandle, err := os.Open(diagnosticFile)
	if err != nil {
		t.Fatalf("failed to open diagnostic output: %v", err)
	}
	defer diagnosticHandle.Close()

	rows, err := csv.NewReader(diagnosticHandle).ReadAll()
	if err != nil {
		t.Fatalf("failed to read diagnostic output: %v", err)
	}
	if len(rows) != 2 {
		t.Fatalf("expected one diagnostic row after omitting the final bucket, got %d rows", len(rows)-1)
	}
	if got, want := rows[1][0], "20.000000"; got != want {
		t.Fatalf("expected final diagnostic bucket to start at %s, got %s", want, got)
	}
}
