package main

import (
	"bufio"
	"encoding/csv"
	"flag"
	"fmt"
	"io"
	"os"
	"strconv"

	"interval_anomaly/pkg/detector"
	"interval_anomaly/pkg/model"
	"interval_anomaly/pkg/timeseries"
)

func main() {
	modelPath := flag.String("model", "model.json", "Path to trained model JSON file (required)")
	nParam := flag.Int("n", 1, "Order of difference (n-th order: t[i] - t[i-n]) (required, > 0)")
	inputPath := flag.String("input", "", "Path to input events file (defaults to stdin)")
	outputPath := flag.String("output", "", "Path to output CSV file (defaults to stdout)")
	colIndex := flag.Int("col", 0, "Column index (0-based) for timestamp in CSV input")
	printHeader := flag.Bool("header", true, "Include CSV header row in output")
	flag.Parse()

	if *nParam <= 0 {
		fmt.Fprintf(os.Stderr, "Error: parameter -n must be a positive integer, got %d\n", *nParam)
		flag.Usage()
		os.Exit(1)
	}

	if *modelPath == "" {
		fmt.Fprintf(os.Stderr, "Error: -model parameter is required\n")
		flag.Usage()
		os.Exit(1)
	}

	// Load model
	m, err := model.LoadModel(*modelPath)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error loading model from %s: %v\n", *modelPath, err)
		os.Exit(1)
	}

	fmt.Fprintf(os.Stderr, "Loaded model: interval=%.3fs, horizon=%d, weights=%d\n",
		m.BucketInterval, m.Horizon, len(m.Weights))
	fmt.Fprintf(os.Stderr, "Running anomaly detection with n=%d\n", *nParam)

	// Open input
	var reader io.Reader
	if *inputPath == "" || *inputPath == "-" {
		reader = os.Stdin
	} else {
		file, err := os.Open(*inputPath)
		if err != nil {
			fmt.Fprintf(os.Stderr, "Error opening input file %s: %v\n", *inputPath, err)
			os.Exit(1)
		}
		defer file.Close()
		reader = file
	}

	// Open output writer
	var writer io.Writer
	if *outputPath == "" || *outputPath == "-" {
		writer = os.Stdout
	} else {
		outFile, err := os.Create(*outputPath)
		if err != nil {
			fmt.Fprintf(os.Stderr, "Error creating output file %s: %v\n", *outputPath, err)
			os.Exit(1)
		}
		defer outFile.Close()
		writer = outFile
	}

	csvWriter := csv.NewWriter(writer)
	defer csvWriter.Flush()

	if *printHeader {
		if err := csvWriter.Write([]string{"time", "nth_order_diff", "expected_rate", "anomaly_statistic"}); err != nil {
			fmt.Fprintf(os.Stderr, "Error writing CSV header: %v\n", err)
			os.Exit(1)
		}
	}

	det, err := detector.NewStreamDetector(m, *nParam)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error initializing detector: %v\n", err)
		os.Exit(1)
	}

	scanner := bufio.NewScanner(reader)
	lineNum := 0
	recordsWritten := 0

	for scanner.Scan() {
		lineNum++
		line := scanner.Text()
		ev, err := timeseries.ExtractTimestampFromLine(line, *colIndex)
		if err != nil {
			if lineNum == 1 {
				// Possible header in input CSV
				continue
			}
			continue
		}

		rec, err := det.ProcessEvent(ev)
		if err != nil {
			fmt.Fprintf(os.Stderr, "Error processing event on line %d: %v\n", lineNum, err)
			continue
		}

		if rec != nil {
			row := []string{
				rec.RawTime,
				strconv.FormatFloat(rec.NthOrderDiff, 'f', 6, 64),
				strconv.FormatFloat(rec.ExpectedRate, 'f', 6, 64),
				strconv.FormatFloat(rec.AnomalyStatistic, 'f', 6, 64),
			}
			if err := csvWriter.Write(row); err != nil {
				fmt.Fprintf(os.Stderr, "Error writing CSV row: %v\n", err)
				os.Exit(1)
			}
			recordsWritten++
		}
	}

	if err := scanner.Err(); err != nil {
		fmt.Fprintf(os.Stderr, "Error reading input stream: %v\n", err)
		os.Exit(1)
	}

	csvWriter.Flush()
	if err := csvWriter.Error(); err != nil {
		fmt.Fprintf(os.Stderr, "Error flushing CSV writer: %v\n", err)
		os.Exit(1)
	}

	fmt.Fprintf(os.Stderr, "Detection complete: %d events processed, %d anomaly records output\n",
		lineNum, recordsWritten)
}
