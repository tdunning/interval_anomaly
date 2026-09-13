package timeseries

import (
	"fmt"
	"math"
	"strconv"
	"strings"
	"time"
)

// Common ISO 8601 / date-time formats to try when parsing string timestamps.
var isoFormats = []string{
	time.RFC3339Nano,
	time.RFC3339,
	"2006-01-02T15:04:05.999999999",
	"2006-01-02T15:04:05.999999999Z07:00",
	"2006-01-02T15:04:05",
	"2006-01-02 15:04:05.999999999Z07:00",
	"2006-01-02 15:04:05Z07:00",
	"2006-01-02 15:04:05.999999999",
	"2006-01-02 15:04:05",
	"2006-01-02",
}

// Event represents an event occurrence with its time in seconds since epoch
// and its raw string representation with millisecond precision.
type Event struct {
	RawText   string
	Seconds   float64
	Timestamp time.Time
	IsEpoch   bool
}

// RoundToMillis rounds a float duration or epoch seconds to millisecond precision (3 decimal places).
func RoundToMillis(val float64) float64 {
	return math.Round(val*1000.0) / 1000.0
}

// FormatISOWithMillis formats a time.Time in UTC with millisecond precision (e.g. 2006-01-02T15:04:05.000Z).
func FormatISOWithMillis(t time.Time) string {
	return t.UTC().Format("2006-01-02T15:04:05.000Z")
}

// FormatEpochWithMillis formats seconds since epoch with 3 decimal places (millisecond precision).
func FormatEpochWithMillis(sec float64) string {
	return fmt.Sprintf("%.3f", RoundToMillis(sec))
}

// ParseTime parses a time string which can be either a numeric value in seconds
// (since epoch) or an ISO/RFC3339 formatted date-time string, keeping millisecond precision.
func ParseTime(s string) (Event, error) {
	trimmed := strings.TrimSpace(s)
	// Strip surrounding quotes if any (e.g. from CSV fields)
	trimmed = strings.Trim(trimmed, `"'`)
	if trimmed == "" {
		return Event{}, fmt.Errorf("empty timestamp string")
	}

	// 1. Try numeric float / int seconds since epoch
	if sec, err := strconv.ParseFloat(trimmed, 64); err == nil {
		sec = RoundToMillis(sec)
		msecTotal := int64(math.Round(sec * 1000.0))
		t := time.UnixMilli(msecTotal).UTC()
		return Event{
			RawText:   FormatEpochWithMillis(sec),
			Seconds:   sec,
			Timestamp: t,
			IsEpoch:   true,
		}, nil
	}

	// 2. Try ISO formats
	for _, layout := range isoFormats {
		if t, err := time.Parse(layout, trimmed); err == nil {
			t = t.UTC().Round(time.Millisecond)
			sec := float64(t.UnixMilli()) / 1000.0
			return Event{
				RawText:   FormatISOWithMillis(t),
				Seconds:   sec,
				Timestamp: t,
				IsEpoch:   false,
			}, nil
		}
	}

	return Event{}, fmt.Errorf("unable to parse timestamp %q as epoch seconds or ISO format", s)
}

// ExtractTimestampFromLine extracts and parses a timestamp from a single line or CSV row.
// If colIndex >= 0, it treats the line as comma-separated and uses that column index (default 0).
func ExtractTimestampFromLine(line string, colIndex int) (Event, error) {
	line = strings.TrimSpace(line)
	if line == "" || strings.HasPrefix(line, "#") {
		return Event{}, fmt.Errorf("empty or comment line")
	}

	field := line
	if strings.Contains(line, ",") {
		parts := strings.Split(line, ",")
		if colIndex < 0 {
			colIndex = 0
		}
		if colIndex < len(parts) {
			field = parts[colIndex]
		}
	}

	return ParseTime(field)
}
