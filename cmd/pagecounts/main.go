// Copyright (c) 2026 Ted Dunning
// SPDX-License-Identifier: MIT

package main

import (
	"bufio"
	"compress/gzip"
	"encoding/csv"
	"flag"
	"fmt"
	"io"
	"io/fs"
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"time"
)

// pagecounts-20160101-000000.gz
var dumpFilePattern = regexp.MustCompile(`^pagecounts-(\d{8})-(\d{6})\.gz$`)

var verbose bool

func main() {
	inputDir := flag.String("input", "dumps.wikipedia.org", "Directory recursively scanned for pagecounts-$date-$time.gz files")
	terms := flag.String("terms", "", "Comma separated list of terms to look for (required)")
	outPath := flag.String("out", "", "Path to output CSV file (defaults to stdout)")
	flag.BoolVar(&verbose, "verbose", false, "Print progress information to stderr")
	flag.Parse()

	wanted, err := parseTerms(*terms)
	if err != nil {
		_, _ = fmt.Fprintf(os.Stderr, "Error: %v\n", err)
		flag.Usage()
		os.Exit(1)
	}

	out := io.Writer(os.Stdout)
	if *outPath != "" && *outPath != "-" {
		outFile, err := os.Create(*outPath)
		if err != nil {
			_, _ = fmt.Fprintf(os.Stderr, "Error creating output file %s: %v\n", *outPath, err)
			os.Exit(1)
		}
		defer outFile.Close()
		out = outFile
	}

	writer := csv.NewWriter(out)
	defer writer.Flush()

	filesScanned := 0
	rowsWritten := 0

	err = filepath.WalkDir(*inputDir, func(path string, entry fs.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if entry.IsDir() {
			return nil
		}
		match := dumpFilePattern.FindStringSubmatch(entry.Name())
		if match == nil {
			return nil
		}

		filesScanned++
		written, err := extractFile(path, match[1], match[2], wanted, writer)
		if err != nil {
			return fmt.Errorf("%s: %w", path, err)
		}
		rowsWritten += written
		return nil
	})
	if err != nil {
		_, _ = fmt.Fprintf(os.Stderr, "Error scanning %s: %v\n", *inputDir, err)
		os.Exit(1)
	}

	writer.Flush()
	if err := writer.Error(); err != nil {
		_, _ = fmt.Fprintf(os.Stderr, "Error writing output: %v\n", err)
		os.Exit(1)
	}

	if verbose {
		_, _ = fmt.Fprintf(os.Stderr, "Scanned %d dump files, wrote %d matching rows\n", filesScanned, rowsWritten)
	}
}

// parseTerms maps every encoding of each requested term back to the term itself.
func parseTerms(terms string) (map[string]string, error) {
	wanted := make(map[string]string)
	for _, term := range strings.Split(terms, ",") {
		term = strings.TrimSpace(term)
		if term == "" {
			continue
		}
		wanted[term] = term
		wanted[urlEncode(term)] = term
	}
	if len(wanted) == 0 {
		return nil, fmt.Errorf("-terms parameter is required")
	}
	return wanted, nil
}

// urlEncode percent-encodes special characters the way page titles are escaped in the dumps.
func urlEncode(term string) string {
	var encoded strings.Builder
	for i := 0; i < len(term); i++ {
		c := term[i]
		switch {
		case c >= 'A' && c <= 'Z', c >= 'a' && c <= 'z', c >= '0' && c <= '9':
			encoded.WriteByte(c)
		case c == '-', c == '_', c == '.', c == '~', c == ':', c == '/':
			encoded.WriteByte(c)
		default:
			_, _ = fmt.Fprintf(&encoded, "%%%02X", c)
		}
	}
	return encoded.String()
}

func extractFile(path, date, timeOfDay string, wanted map[string]string, writer *csv.Writer) (int, error) {
	file, err := os.Open(path)
	if err != nil {
		return 0, err
	}
	defer file.Close()

	gzipReader, err := gzip.NewReader(file)
	if err != nil {
		return 0, err
	}
	defer gzipReader.Close()

	rowsWritten := 0
	scanner := bufio.NewScanner(gzipReader)
	scanner.Buffer(make([]byte, 0, 64*1024), 1024*1024)

	t0 := time.Now()
	for scanner.Scan() {
		fields := strings.Fields(scanner.Text())
		if len(fields) < 3 || fields[0] != "en" {
			continue
		}
		term, ok := wanted[fields[1]]
		if !ok {
			continue
		}

		row := append([]string{date, timeOfDay, term}, fields[2:]...)
		if err := writer.Write(row); err != nil {
			return rowsWritten, err
		}
		rowsWritten++
	}
	writer.Flush()
	if verbose {
		_, _ = fmt.Fprintf(os.Stderr, "Scanned %s in %.2f s\n", path, time.Since(t0).Seconds())
	}

	return rowsWritten, scanner.Err()
}
