// Copyright (c) 2026 Ted Dunning
// SPDX-License-Identifier: MIT

package timeseries

import (
	"math"
	"testing"
	"time"
)

func TestParseTime(t *testing.T) {
	tests := []struct {
		name        string
		input       string
		wantSeconds float64
		wantErr     bool
	}{
		{
			name:        "Epoch integer seconds",
			input:       "1609459200",
			wantSeconds: 1609459200,
			wantErr:     false,
		},
		{
			name:        "Epoch float seconds",
			input:       "1609459200.75",
			wantSeconds: 1609459200.75,
			wantErr:     false,
		},
		{
			name:        "ISO 8601 UTC Z",
			input:       "2021-01-01T00:00:00Z",
			wantSeconds: float64(time.Date(2021, 1, 1, 0, 0, 0, 0, time.UTC).Unix()),
			wantErr:     false,
		},
		{
			name:        "RFC3339 with nano",
			input:       "2021-01-01T00:00:00.500000000Z",
			wantSeconds: float64(time.Date(2021, 1, 1, 0, 0, 0, 500000000, time.UTC).UnixNano()) / 1e9,
			wantErr:     false,
		},
		{
			name:        "ISO 8601 with fractional milliseconds",
			input:       "2021-01-01T00:00:00.123Z",
			wantSeconds: 1609459200.123,
			wantErr:     false,
		},
		{
			name:        "Epoch float milliseconds",
			input:       "1609459200.123456",
			wantSeconds: 1609459200.123, // rounded to ms
			wantErr:     false,
		},
		{
			name:        "ISO 8601 with offset",
			input:       "2021-01-01T05:00:00+05:00",
			wantSeconds: float64(time.Date(2021, 1, 1, 0, 0, 0, 0, time.UTC).Unix()),
			wantErr:     false,
		},
		{
			name:        "ISO space separated",
			input:       "2021-01-01 00:00:00",
			wantSeconds: float64(time.Date(2021, 1, 1, 0, 0, 0, 0, time.UTC).Unix()),
			wantErr:     false,
		},
		{
			name:    "Invalid string",
			input:   "not-a-date",
			wantErr: true,
		},
		{
			name:    "Empty string",
			input:   "",
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			ev, err := ParseTime(tt.input)
			if (err != nil) != tt.wantErr {
				t.Fatalf("ParseTime(%q) error = %v, wantErr %v", tt.input, err, tt.wantErr)
			}
			if !tt.wantErr {
				if math.Abs(ev.Seconds-tt.wantSeconds) > 1e-4 {
					t.Errorf("ParseTime(%q) got seconds = %f, want %f", tt.input, ev.Seconds, tt.wantSeconds)
				}
			}
		})
	}
}

func TestExtractTimestampFromLine(t *testing.T) {
	line := "1609459200.5,event_foo,123"
	ev, err := ExtractTimestampFromLine(line, 0)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if math.Abs(ev.Seconds-1609459200.5) > 1e-4 {
		t.Errorf("expected 1609459200.5, got %f", ev.Seconds)
	}

	// Comment line
	_, err = ExtractTimestampFromLine("# Header line", 0)
	if err == nil {
		t.Errorf("expected error for comment line")
	}
}
