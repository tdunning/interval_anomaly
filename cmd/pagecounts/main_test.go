// Copyright (c) 2026 Ted Dunning
// SPDX-License-Identifier: MIT

package main

import (
	"bytes"
	"compress/gzip"
	"encoding/csv"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestURLEncode(t *testing.T) {
	if got, want := urlEncode("Cat_(disambiguation)"), "Cat_%28disambiguation%29"; got != want {
		t.Errorf("urlEncode: got %q, want %q", got, want)
	}
	if got, want := urlEncode("Barack_Obama"), "Barack_Obama"; got != want {
		t.Errorf("urlEncode: got %q, want %q", got, want)
	}
}

func TestExtractFile(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "pagecounts-20160101-010000.gz")
	writeGzip(t, path, strings.Join([]string{
		"en Barack_Obama 123 4567",
		"en Cat_%28disambiguation%29 5 60",
		"de Barack_Obama 9 99",
		"en Unrelated 1 2",
	}, "\n"))

	wanted, err := parseTerms("Barack_Obama, Cat_(disambiguation)")
	if err != nil {
		t.Fatalf("parseTerms failed: %v", err)
	}

	var out bytes.Buffer
	writer := csv.NewWriter(&out)
	rows, err := extractFile(path, "20160101", "010000", wanted, writer)
	if err != nil {
		t.Fatalf("extractFile failed: %v", err)
	}
	writer.Flush()

	if rows != 2 {
		t.Errorf("expected 2 matching rows, got %d", rows)
	}
	want := "20160101,010000,Barack_Obama,123,4567\n20160101,010000,Cat_(disambiguation),5,60\n"
	if out.String() != want {
		t.Errorf("unexpected output:\ngot  %q\nwant %q", out.String(), want)
	}
}

func TestExtractFileSampleLines(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "pagecounts-20151126-120000.gz")
	writeGzip(t, path, strings.Join([]string{
		"en Thanksgiving 113 113",
		"en Thanksgiving_%28United_States%29 16 16",
	}, "\n"))

	wanted, err := parseTerms("Thanksgiving,Thanksgiving_(United_States)")
	if err != nil {
		t.Fatalf("parseTerms failed: %v", err)
	}

	var out bytes.Buffer
	writer := csv.NewWriter(&out)
	if _, err := extractFile(path, "20151126", "120000", wanted, writer); err != nil {
		t.Fatalf("extractFile failed: %v", err)
	}
	writer.Flush()

	want := "20151126,120000,Thanksgiving,113,113\n" +
		"20151126,120000,Thanksgiving_(United_States),16,16\n"
	if out.String() != want {
		t.Errorf("unexpected output:\ngot  %q\nwant %q", out.String(), want)
	}
}

func writeGzip(t *testing.T, path, contents string) {
	t.Helper()
	file, err := os.Create(path)
	if err != nil {
		t.Fatalf("failed to create %s: %v", path, err)
	}
	defer file.Close()

	gzipWriter := gzip.NewWriter(file)
	if _, err := gzipWriter.Write([]byte(contents)); err != nil {
		t.Fatalf("failed to write gzip contents: %v", err)
	}
	if err := gzipWriter.Close(); err != nil {
		t.Fatalf("failed to close gzip writer: %v", err)
	}
}
