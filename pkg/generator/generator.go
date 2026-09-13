package generator

import (
	"fmt"
	"math"
	"math/rand"
	"time"

	"interval_anomaly/pkg/timeseries"
)

const (
	// DayInSeconds represents the number of seconds in a standard 24-hour day.
	DayInSeconds = 86400.0
	// HourInSeconds represents the number of seconds in one hour.
	HourInSeconds = 3600.0
	// WeekInSeconds represents 7 days in seconds.
	WeekInSeconds = 7.0 * DayInSeconds
)

// GeneratorConfig holds parameters for synthetic event generation.
type GeneratorConfig struct {
	KScale        float64       // Events per hour (default 500)
	KOffset       float64       // Events per hour (default 100)
	Offset        float64       // Phase offset in days (or fraction of day)
	TotalDuration time.Duration // Total duration to simulate
	BaseTime      time.Time     // Starting timestamp for ISO/epoch output
	TimeFormat    string        // "iso", "rfc3339", "epoch", or "seconds"
	Seed          int64         // RNG seed (0 for time-based)
}

// DefaultConfig returns the default generator configuration.
func DefaultConfig() GeneratorConfig {
	return GeneratorConfig{
		KScale:        500.0,
		KOffset:       100.0,
		Offset:        0.0,
		TotalDuration: 30 * 24 * time.Hour, // 30 days
		BaseTime:      time.Date(2025, 1, 1, 0, 0, 0, 0, time.UTC),
		TimeFormat:    "iso",
		Seed:          0,
	}
}

// K calculates k(t) = k_scale * exp(2) - exp(-2 * sin(2*pi*(t/(7 * day)))) + k_offset
// tSec is elapsed time in seconds.
// Returns rate multiplier in events per hour.
func K(tSec float64, kScale, kOffset float64) float64 {
	sinTerm := math.Sin(2.0 * math.Pi * (tSec / WeekInSeconds))
	return kScale*(math.Exp(2.0) - math.Exp(-2.0*sinTerm)) + kOffset
}

// RatePerHour calculates the instantaneous rate in events per hour at time tSec (seconds).
// rate = k(t) * exp(-2*cos(2π * (t-offset))) + exp(-1.9*cos(2π * (t+offset)))
// where t-offset and t+offset are expressed with diurnal period (1 day = 86400s).
func RatePerHour(tSec float64, kScale, kOffset, offsetDays float64) float64 {
	kt := K(tSec, kScale, kOffset)

	tDays := tSec / DayInSeconds
	cosTerm1 := math.Cos(2.0 * math.Pi * (tDays - offsetDays))
	cosTerm2 := math.Cos(2.0 * math.Pi * (tDays + offsetDays))

	r := kt*(math.Exp(-2.0*cosTerm1) + math.Exp(-1.9*cosTerm2))
	if r < 0 {
		return 0
	}
	return r
}

// RatePerSecond calculates the instantaneous rate in events per second at time tSec.
func RatePerSecond(tSec float64, kScale, kOffset, offsetDays float64) float64 {
	return RatePerHour(tSec, kScale, kOffset, offsetDays) / HourInSeconds
}

// MaxRatePerSecond computes a safe upper bound on RatePerSecond over [0, totalSeconds].
func MaxRatePerSecond(totalSeconds float64, kScale, kOffset, offsetDays float64) float64 {
	// Maximum possible value of k(t):
	// sin(...) can be -1 => -exp(-2 * (-1)) = -exp(2) is subtracted?
	// When sin = -1, -exp(2) is subtracted, so k(t) is smaller.
	// When sin = 1, -exp(-2) is subtracted (~ -0.135), so k(t) reaches max:
	// k_max = kScale * exp(2) - exp(-2) + kOffset
	kMax := kScale*(math.Exp(2.0) - math.Exp(-2.0)) + kOffset
	if kMax < 0 {
		kMax = 0
	}

	// For exp(-2*cos(...)), maximum is exp(2) when cos(...) = -1.
	// For exp(-1.9*cos(...)), maximum is exp(1.9) when cos(...) = -1.
	maxPerHour := kMax*(math.Exp(2.0) + math.Exp(1.9))

	return (maxPerHour / HourInSeconds) * 1.05 // 5% safety margin
}

// GenerateEvents simulates the Non-Homogeneous Poisson Process (NHPP) using
// Lewis-Shedler thinning algorithm over the configured duration.
func GenerateEvents(cfg GeneratorConfig) ([]timeseries.Event, error) {
	if cfg.TotalDuration <= 0 {
		return nil, fmt.Errorf("total duration must be positive, got %v", cfg.TotalDuration)
	}

	var rng *rand.Rand
	if cfg.Seed != 0 {
		rng = rand.New(rand.NewSource(cfg.Seed))
	} else {
		rng = rand.New(rand.NewSource(time.Now().UnixNano()))
	}

	totalSec := cfg.TotalDuration.Seconds()
	lambdaMax := MaxRatePerSecond(totalSec, cfg.KScale, cfg.KOffset, cfg.Offset)
	if lambdaMax <= 0 {
		return nil, fmt.Errorf("invalid maximum rate: %f", lambdaMax)
	}

	baseSec := float64(cfg.BaseTime.UnixNano()) / 1e9

	var events []timeseries.Event
	t := 0.0

	for t < totalSec {
		// Generate exponential inter-arrival time with rate lambdaMax: E ~ Exp(lambdaMax)
		u := rng.Float64()
		for u == 0 {
			u = rng.Float64()
		}
		t += -math.Log(u) / lambdaMax
		if t >= totalSec {
			break
		}

		// Acceptance probability: lambda(t) / lambdaMax
		currentRate := RatePerSecond(t, cfg.KScale, cfg.KOffset, cfg.Offset)
		acceptanceProb := currentRate / lambdaMax
		if rng.Float64() <= acceptanceProb {
			absSec := timeseries.RoundToMillis(baseSec + t)
			msecTotal := int64(math.Round(absSec * 1000.0))
			eventTime := time.UnixMilli(msecTotal).UTC()

			var rawStr string
			switch cfg.TimeFormat {
			case "epoch", "seconds":
				rawStr = timeseries.FormatEpochWithMillis(absSec)
			case "rfc3339", "iso":
				rawStr = timeseries.FormatISOWithMillis(eventTime)
			default:
				rawStr = timeseries.FormatISOWithMillis(eventTime)
			}

			events = append(events, timeseries.Event{
				RawText:   rawStr,
				Seconds:   absSec,
				Timestamp: eventTime,
				IsEpoch:   cfg.TimeFormat == "epoch" || cfg.TimeFormat == "seconds",
			})
		}
	}

	return events, nil
}

// SplitEvents splits an event slice into two parts by ratio (e.g. 0.9 for 90% / 10%).
func SplitEvents(events []timeseries.Event, splitRatio float64) ([]timeseries.Event, []timeseries.Event) {
	if splitRatio <= 0 {
		return nil, events
	}
	if splitRatio >= 1.0 {
		return events, nil
	}

	splitIdx := int(math.Round(float64(len(events)) * splitRatio))
	if splitIdx > len(events) {
		splitIdx = len(events)
	}
	return events[:splitIdx], events[splitIdx:]
}
