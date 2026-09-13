# Interval Anomaly Detection Suite

A Go toolkit for synthetic event generation, historical event frequency modeling using $L_1$-regularized autoregression (Lasso), and real-time anomaly detection on time interval differences.

## Applications

1. **Synthetic Event Generator (`cmd/generate`)**
   - Simulates a non-homogeneous Poisson process (NHPP) using Lewis-Shedler thinning.
   - Event rate model:
     $$\text{rate}(t) = k(t) \cdot \exp(-2\cos(2\pi(t - \text{offset}))) + \exp(-1.9\cos(2\pi(t + \text{offset})))$$
     where
     $$k(t) = k_{\text{scale}} \cdot \exp(2) - \exp(-2\sin(2\pi(t / (7 \cdot \text{day})))) + k_{\text{offset}}$$
   - `k_scale` (default: 500 events/hour) and `k_offset` (default: 100 events/hour) are command-line parameters.
   - Outputs timestamps formatted as ISO 8601/RFC 3339 or epoch seconds.
   - Splits output into two files: first 90% of the data (training set) and the last 10% of the times (test/detection set).

2. **Historical Analysis & AR Trainer (`cmd/historical`)**
   - Reads event timestamps (in seconds since epoch or ISO 8601/RFC 3339 formats).
   - Groups data into fixed-width time buckets of duration $\Delta t$.
   - Constructs an Autoregressive $AR(p)$ dataset predicting $y_t = \log(c_t + \epsilon)$ from preceding bucket counts $c_{t-1}, \dots, c_{t-p}$.
   - Solves sparse $L_1$-regularized linear regression via Coordinate Descent.
   - Outputs textual progress indicators to stderr and regression parameters as a JSON structure.
   - Optionally writes a bucket-level diagnostic CSV containing actual and fitted predicted rates for plotting.

3. **Online / Streaming Anomaly Detector (`cmd/anomaly`)**
   - Takes a trained model JSON and the difference order parameter $n$.
   - Computes $n$-th order time differences $\Delta_n(t_i) = t_i - t_{i-n}$.
   - Evaluates the expected rate as the duration-weighted average of autoregressive estimates for every bucket overlapping the interval from the earliest of the preceding $n$ events through the current event.
   - Computes anomaly statistic $S(t_i) = \frac{t_i - t_{i-n}}{\text{expected\_rate}}$.
   - Outputs CSV formatted stream with `time`, `nth_order_diff`, `expected_rate`, and `anomaly_statistic`.

## Build

```bash
go build -o bin/generate ./cmd/generate
go build -o bin/historical ./cmd/historical
go build -o bin/anomaly ./cmd/anomaly
```

## Usage

### 1. Generating Synthetic Event Data

Generate 30 days of synthetic event data split into 90% (`train_events.csv`) and 10% (`test_events.csv`):
```bash
./bin/generate -k_scale 500 -k_offset 100 -total_time 30d -out_90 train_events.csv -out_10 test_events.csv
```

### 2. Training Historical Model

Create a configuration JSON (e.g. `config.json`):
```json
{
  "bucket_interval": 3600.0,
  "horizon": 24,
  "regularization_penalty": 0.01,
  "epsilon": 1.0,
  "max_iterations": 2000,
  "tolerance": 1e-6
}
```

Run historical training:
```bash
./bin/historical -config config.json -input train_events.csv -output model.json
```

Write bucket-level actual and predicted rates to a CSV for plotting:
```bash
./bin/historical -config config.json -input train_events.csv -output model.json -diagnostic rates.csv
```

The final bucket is omitted from training because it can be incomplete. The diagnostic CSV columns are `bucket_start_time`, `actual_rate`, and `predicted_rate`, in events per second. Output begins after the first `horizon` buckets, when all lag values are available.

Or read from stdin / pipe to stdout:
```bash
cat train_events.csv | ./bin/historical -config config.json > model.json
```

### 3. Detecting Anomalies

Run anomaly detection with order $n = 1$:
```bash
./bin/anomaly -model model.json -n 1 -input test_events.csv -output anomalies.csv
```

Or stream in real time:
```bash
tail -f live_events.csv | ./bin/anomaly -model model.json -n 1
```

## Original Prompts

Write one Go application that analyzes historical event data and another that analyzes new data for anomalies. The input should be in the form of a sequence of times expressed either as times in seconds since an epoch or as dates in ISO format.

The first application (the historical analysis program) should group data into buckets representing constant time intervals and learn a sparse auto-regressive predictor for the frequency of events. The inputs for the predictor should be bucket counts over a specified time horizon and the output should be log(count + epsilon) where epsilon is a specified small constant. The regression should be done using L_1 regularized linear regression. The bucket intervals, the horizon, the regularization penalty size and epsilon should all be taken from a configuration file that contains a JSON structure. The output should be 1) textual progress indicators and 2) a JSON structure containing the parameters of the regression

The second application should accept event times in the same format. It should form n-th order differences (that is t[i] - t[i-n]). The expected value for this number of events per bucket should be computed by using the linear regression from the previous program and the actual time difference should be divided by this expected rate to get an anomaly statistic that should be output. The output format should be CSV like the input and have a time (the t[i] from above), the n-th order difference, the current expected rate, and the resulting statistic. The parameter n should be taken from the command line.

Add a third program that will generate synthetic event data. The rate should look like

rate = k(t) * exp(-2*cos(2π * (t-offset))) + exp(-1.9*cos(2π * (t+offset)))

where k(t) = k_scale * exp(2) - exp(-2 * sin(2*pi*(t/(7 * day)))) + k_offset

k_scale and k_offset should be command line parameters expressed in events per hour.

The output should be a list of times in a format suitable for the historical and anomaly programs. The total time for the output should also be a command line parameter. The output should be split into two files, one with 90% of the data and the other with the last 10% of the times.

The default values should be k_scale = 500, k_offset = 100