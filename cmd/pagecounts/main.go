// Copyright (c) 2026 Ted Dunning
// SPDX-License-Identifier: MIT

package main

import (
	"bufio"
	"bytes"
	"compress/gzip"
	"encoding/csv"
	"flag"
	"fmt"
	"io"
	"io/fs"
	"os"
	"path/filepath"
	"regexp"
	"runtime"
	"strings"
	"sync"
	"time"
)

// pagecounts-20160101-000000.gz
var dumpFilePattern = regexp.MustCompile(`^pagecounts-(\d{8})-(\d{6})\.gz$`)

var verbose bool

// logMu keeps progress lines from interleaving across workers.
var logMu sync.Mutex

func main() {
	inputDir := flag.String("input", "dumps.wikipedia.org", "Directory recursively scanned for pagecounts-$date-$time.gz files")
	terms := flag.String("terms", "", "Comma separated list of terms to look for (required)")
	outPath := flag.String("out", "", "Path to output CSV file (defaults to stdout)")
	threads := flag.Int("threads", defaultThreads(), "Number of dump files to process concurrently")
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

	writer := bufio.NewWriter(out)

	dumps, err := collectDumpFiles(*inputDir)
	if err != nil {
		_, _ = fmt.Fprintf(os.Stderr, "Error scanning %s: %v\n", *inputDir, err)
		os.Exit(1)
	}

	results, rowsWritten, err := processDumps(dumps, wanted, *threads)
	if err != nil {
		_, _ = fmt.Fprintf(os.Stderr, "Error: %v\n", err)
		os.Exit(1)
	}

	// Emit in walk order so output does not depend on completion order.
	for _, chunk := range results {
		if _, err := writer.Write(chunk); err != nil {
			_, _ = fmt.Fprintf(os.Stderr, "Error writing output: %v\n", err)
			os.Exit(1)
		}
	}

	if err := writer.Flush(); err != nil {
		_, _ = fmt.Fprintf(os.Stderr, "Error writing output: %v\n", err)
		os.Exit(1)
	}

	if verbose {
		_, _ = fmt.Fprintf(os.Stderr, "Scanned %d dump files, wrote %d matching rows\n", len(dumps), rowsWritten)
	}
}

// defaultThreads leaves some headroom rather than saturating every core.
func defaultThreads() int {
	threads := runtime.NumCPU() * 8 / 10
	if threads < 1 {
		return 1
	}
	return threads
}

type dumpFile struct {
	path      string
	date      string
	timeOfDay string
}

func collectDumpFiles(root string) ([]dumpFile, error) {
	var dumps []dumpFile
	err := filepath.WalkDir(root, func(path string, entry fs.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if entry.IsDir() {
			return nil
		}
		if match := dumpFilePattern.FindStringSubmatch(entry.Name()); match != nil {
			dumps = append(dumps, dumpFile{path: path, date: match[1], timeOfDay: match[2]})
		}
		return nil
	})
	return dumps, err
}

// processDumps extracts each dump on a worker pool, returning per-file CSV chunks in input order.
func processDumps(dumps []dumpFile, wanted map[string]string, threads int) ([][]byte, int, error) {
	if threads < 1 {
		threads = 1
	}
	if threads > len(dumps) {
		threads = len(dumps)
	}

	results := make([][]byte, len(dumps))
	counts := make([]int, len(dumps))
	queue := make(chan int)

	var mu sync.Mutex
	var firstErr error
	failed := func() bool {
		mu.Lock()
		defer mu.Unlock()
		return firstErr != nil
	}

	var wg sync.WaitGroup
	for w := 0; w < threads; w++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			// Drain the queue even after a failure so the feeder can never block.
			for i := range queue {
				if failed() {
					continue
				}

				var buf bytes.Buffer
				writer := csv.NewWriter(&buf)
				n, err := extractFile(dumps[i].path, dumps[i].date, dumps[i].timeOfDay, wanted, writer)
				if err == nil {
					err = writer.Error()
				}
				if err != nil {
					mu.Lock()
					if firstErr == nil {
						firstErr = fmt.Errorf("%s: %w", dumps[i].path, err)
					}
					mu.Unlock()
					continue
				}

				results[i], counts[i] = buf.Bytes(), n
			}
		}()
	}

	for i := range dumps {
		if failed() {
			break
		}
		queue <- i
	}
	close(queue)
	wg.Wait()

	if firstErr != nil {
		return nil, 0, firstErr
	}

	rowsWritten := 0
	for _, n := range counts {
		rowsWritten += n
	}
	return results, rowsWritten, nil
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
		// Reject on raw bytes first; only "en " lines can match, and Text/Fields
		// would otherwise allocate for every line in the dump.
		line := scanner.Bytes()
		if len(line) < 3 || line[0] != 'e' || line[1] != 'n' || line[2] != ' ' {
			continue
		}

		rest := line[3:]
		titleEnd := bytes.IndexByte(rest, ' ')
		if titleEnd <= 0 {
			continue
		}
		term, ok := wanted[string(rest[:titleEnd])]
		if !ok {
			continue
		}

		data := strings.Fields(string(rest[titleEnd+1:]))
		if len(data) == 0 {
			continue
		}

		row := append([]string{date, timeOfDay, term}, data...)
		if err := writer.Write(row); err != nil {
			return rowsWritten, err
		}
		rowsWritten++
	}
	writer.Flush()
	if verbose {
		logMu.Lock()
		_, _ = fmt.Fprintf(os.Stderr, "Scanned %s in %.2f s\n", path, time.Since(t0).Seconds())
		logMu.Unlock()
	}

	return rowsWritten, scanner.Err()
}
