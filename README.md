# alpha_assay

Open-source backtesting and paper-trading framework for intraday futures strategies. Bring your own signal; the framework handles data, execution, risk, and reporting.

**Status: pre-alpha.** APIs will change. Not investment advice. Do not run live without completing the Go-Live Checklist.

## What it is

`alpha_assay` is the **shell**: a plugin contract (`BaseStrategy`), a [NautilusTrader](https://nautilustrader.io/)-backed event-driven runner with explicit backtest-to-live parity, adapters for market data ([Databento](https://databento.com/) historical + [IBKR](https://www.interactivebrokers.com/) live via `ib_insync`), and a self-contained Docker Compose stack with private Grafana + Prometheus for observability. Strategies are written as subclasses of `BaseStrategy` and can be kept in your own private repo.

The reference workload is an ES `$TICK` / `$ADD` breadth-fade scalper on 1-minute bars, paper-traded through IBKR. Nothing in the public framework is specific to that strategy.

## Repo layout

```
alpha-assay/
├── src/alpha_assay/         # the importable framework
│   ├── strategy/base.py     # BaseStrategy, Signal (the plugin contract)
│   ├── filters/             # stop_run, session_mask, atr_regime
│   ├── risk/                # hard caps, kill-switch, bracket construction
│   ├── engine/              # NautilusTrader runner + adapter
│   ├── data/                # Databento, IBKR, CSV/parquet replay
│   ├── exec/                # IBKR ib_insync (paper | live flag)
│   ├── observability/       # Prometheus instrumentation
│   ├── config/              # pydantic schema + loader
│   └── cli/                 # backtest, paper, report, kill, reset
├── examples/
│   └── sma_crossover.py     # quickstart strategy
├── apps/
│   └── paper-trader/        # Dockerfile + entrypoint
├── observability/
│   ├── prometheus/          # scrape config
│   └── grafana/             # dashboard JSON + provisioning
├── configs/                 # YAML config for runs
├── data/                    # sample market data + fixtures
├── docs/                    # coding standards, guardrails, metrics, go-live
├── docker-compose.yml       # paper-trader + private prometheus + grafana
└── tests/
```

## Quickstart

```bash
git clone https://github.com/aaronmeza/alpha-assay.git
cd alpha-assay
make setup            # python 3.11 venv + deps
make sample           # produce a synthetic 2-day fixture
make test             # unit + integration tests
make backtest         # run the SMA-crossover example on the sample
```

## CLI usage

`alpha_assay` exposes four subcommands. The Quickstart's `make backtest`
target wraps `alpha_assay backtest` against the synthetic fixture; the
commands below are what you reach for once you have your own data and
strategy.

**Parquet input schema.** `alpha_assay.data.databento_adapter.load_parquet`
enforces a canonical schema: a `timestamp` (or `ts_event`) column, plus
`open`, `high`, `low`, `close`, `volume`. The adapter strictly validates
the OHLC invariants `high >= max(open, close)` and `low <= min(open, close)`.
Bars that violate these are rejected at load time; either clamp them
upstream or run them through `tests/fixtures/make_sample.py` for a
reference shape.

### `alpha_assay backfill`

Pull historical 1-min OHLCV bars from IBKR via
`reqHistoricalDataAsync` and write per-day parquet shards in the
canonical schema accepted by `alpha_assay.data.databento_adapter.load_parquet`.
The shard layout matches the always-on ES-bars recorder, so
historical and live data can land in the same directory and be ingested
interchangeably by `alpha_assay backtest`.

Use this when you need a multi-month historical window for backtests
and you are not (yet) ready to commit to a paid Databento subscription.
IBKR's free CME Real-Time entitlement covers ES futures bars going
back roughly 6 months.

Options:

- `--feed <es-bars>` (required) - currently only `es-bars` is supported.
  IBKR has no historical data for the NYSE breadth indexes
  (`TICK-NYSE`, `AD-NYSE`); see ADR Appendix B.
- `--days <int>` - calendar days back from now to pull. Default 180
  (~6 months).
- `--out <dir>` - per-day shards land here as `YYYY-MM-DD.parquet`.
  Defaults to `data/recorder/es_bars/` so the shards merge cleanly
  with the running recorder's output.
- `--ibkr-host <str>` - TWS / Gateway host. Defaults to `$IBKR_HOST`
  or `127.0.0.1`.
- `--ibkr-port <int>` - TWS / Gateway port. Defaults to `$IBKR_PORT`
  or `4002` (IB Gateway paper).
- `--ibkr-client-id <int>` - IBKR clientId. Default `23` so the
  backfill does not collide with the paper-trader (1), the breadth
  recorder (21), or the ES-bars recorder (22).
- `--es-expiry <YYYYMMDD>` - ES futures `lastTradeDateOrContractMonth`.
  Defaults to `$ES_EXPIRY` or `20260618`. Update on each quarterly roll.
- `--chunk-duration <str>` - per-request IBKR `durationStr`. Default
  `1 W`. For 1-min bars IBKR caps this at roughly 1 week per call.
- `--pace-seconds <int>` - wait between successive IBKR requests.
  Default 10 to stay outside IBKR's 15-second identical-request rule
  and the 6-requests-per-2-seconds rule for the same contract.
- `--symbol <str>` - futures root. Default `ES`.
- `--exchange <str>` - futures exchange. Default `CME`.
- `--currency <str>` - quote currency. Default `USD`.

Behavior:

- Resume-safe. Existing shards are merged with new bars (sort + dedupe
  by timestamp; the new pull's value wins on duplicates). Safe to run
  while the recorder is also writing to `--out`.
- Per-chunk flush. After each chunk lands, the in-memory buffer is
  written to the relevant per-day shards before the next request goes
  out. Ctrl-C still flushes whatever is buffered.
- No `keepUpToDate`. This command issues one-shot historical requests
  only; live streaming is the recorder's job.

Example pulling 30 days against an IB Gateway running on the local
host's default paper port:

```bash
alpha_assay backfill \
  --feed es-bars \
  --days 30 \
  --out data/recorder/es_bars/ \
  --ibkr-host 127.0.0.1 \
  --ibkr-port 4002 \
  --es-expiry 20260618
```

### `alpha_assay backtest`

Run a strategy against historical bars and write trade + summary
artifacts.

Options:

- `--strategy <module:Class>` (required) - dotted path to a
  `BaseStrategy` subclass, e.g. `examples.sma_crossover:SMACrossoverStrategy`.
- `--config <path>` (required) - YAML config (`strategy.params`,
  `risk_caps`, `session`, `execution`).
- `--data <path>` (required) - parquet file in the canonical OHLC schema.
- `--tick-data <path>` (optional) - parquet of per-minute TICK-NYSE
  breadth bars (recorder shard schema). Required when paired with
  `--ad-data` for breadth-aware strategies.
- `--ad-data <path>` (optional) - parquet of per-minute AD-NYSE
  breadth bars. Must be passed together with `--tick-data`; the CLI
  rejects only-one-of-two with a usage error so a strategy never
  silently sees a NaN breadth column.
- `--out <dir>` (required) - output directory; created if absent.

Outputs in `--out`:

- `trades.csv` - raw orders log: `timestamp, signal_ts, fill_ts, side,
  price, quantity, order_type`.
- `session_metrics.json` - `run_status`, `submitted_signals`,
  `orders_submitted`, `signals_filtered_risk_cap`, `final_balance_usd`.

Example (OHLCV-only):

```bash
alpha_assay backtest \
  --strategy examples.sma_crossover:SMACrossoverStrategy \
  --config configs/sma.yaml \
  --data data/es_1m.parquet \
  --out runs/2026-04-27/
```

Example (with NYSE breadth feeds for breadth-aware strategies like
a breadth-aware strategy):

```bash
alpha_assay backtest \
  --strategy mypkg.my_strategy:MyStrategy \
  --config configs/my_strategy.yaml \
  --data data/recorder/es_bars/2026-04-28.parquet \
  --tick-data data/recorder/breadth/TICK_NYSE/2026-04-28.parquet \
  --ad-data data/recorder/breadth/AD_NYSE/2026-04-28.parquet \
  --out runs/2026-04-28-tickfade/
```

The joined loader inner-joins ES OHLCV with TICK and AD on timestamp
and presents the strategy with `data["TICK"]` / `data["ADD"]` columns
alongside OHLCV. Breadth events are streamed through the engine via
the custom-data path (`engine/breadth_adapter.py`) per ADR Appendix C.

### `alpha_assay live-check`

Read-only diagnostic that evaluates the three live-mode locks (env var,
CLI flag, signed checklist file). Useful as a CI gate before flipping a
paper-trader process to live.

Options:

- `--live` - engage the CLI-flag lock. Without this flag, the CLI lock
  is missing; with it, you can confirm whether the other two locks are
  also engaged.

Exit codes:

- `0` - all three locks engaged (`mode=LIVE` would be allowed).
- `2` - at least one lock missing (`mode=PAPER`; live is blocked).

Examples:

```bash
# Inspect which locks are engaged in the current shell:
alpha_assay live-check

# Pretend the CLI lock is set:
alpha_assay live-check --live

# CI gate (in a pre-deploy job):
ALPHA_ASSAY_LIVE=1 alpha_assay live-check --live   # exits 2 unless GO_LIVE_CHECKLIST_SIGNED is staged
```

### `alpha_assay report`

Turn a backtest output directory into a decision-grade report. Produces
a per-trade CSV, an enriched session-metrics JSON, a Markdown summary,
and (optionally) a quantstats HTML tearsheet.

Requires the `report` extra: `pip install -e ".[report]"`.

Options:

- `--in <dir>` (required) - input directory containing `trades.csv` +
  `session_metrics.json` from a previous backtest run.
- `--out <dir>` - where to write the report artifacts. Defaults to `--in`.
- `--format <html|md|both>` - report format. Default: `both`. The
  Markdown report is always generated when `md` or `both`; the HTML
  tearsheet requires quantstats and at least two distinct calendar days
  of activity.
- `--instrument-multiplier <float>` - dollars per point. Default `50.0`
  for ES; use `5.0` for MES.
- `--data <parquet>` - optional canonical-schema parquet used to
  compute per-trade MAE / MFE. Skipped if absent.

Outputs in `--out`:

- `per_trade_metrics.csv` - one row per round-trip trade, with PnL
  (points and USD), hold time, MAE / MFE, and exit reason
  (`target` / `stop` / `flip`).
- `session_metrics.json` - additive enrichment of the input file. All
  original keys preserved; new keys cover Sharpe / Sortino / max
  drawdown, win rate, profit factor, avg win/loss, and the
  live-vs-backtest parity check fields (per-session PnL, trade count, and
  `per_session_avg_fill_time_seconds` - the mean signal-to-fill
  latency in seconds, populated when the runner emits signal_ts on
  each trade).
- `report.md` - human-readable Markdown summary tables.
- `report.html` - quantstats tearsheet when format is `html` or `both`.

Example end-to-end (backtest then report):

```bash
alpha_assay backtest \
  --strategy examples.sma_crossover:SMACrossoverStrategy \
  --config configs/sma.yaml \
  --data data/es_1m.parquet \
  --out runs/2026-04-27/

alpha_assay report \
  --in runs/2026-04-27/ \
  --instrument-multiplier 50.0 \
  --data data/es_1m.parquet \
  --format both
```

## Bring your own strategy

A strategy is any subclass of `alpha_assay.strategy.base.BaseStrategy`:

```python
from alpha_assay.strategy.base import BaseStrategy, Signal
import pandas as pd

class MyStrategy(BaseStrategy):
    def _validate_config(self) -> None:
        if "lookback" not in self.config.get("signal", {}):
            raise ValueError("signal.lookback is required")

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        # return a Series of {-1, 0, +1} aligned to data.index
        ...

    def get_exit_params(self, signal: Signal, data: pd.DataFrame) -> tuple[float, float]:
        # return (stop_points, target_points) within the framework's hard caps
        ...
```

Drop it in your own repo, point a config at it, run:

```bash
alpha_assay backtest --strategy mypkg.mymod:MyStrategy --config configs/my.yaml
```

Your strategy stays in your repo. Only the config and entrypoint reference are public-facing.

## Architecture

```
  market data      +-----------+
  historical  ---> | adapters  |  Databento parquet, CSV, IBKR live
  + live           +-----+-----+
                         v
                  +------+-------+
                  |  engine      |   NautilusTrader: event loop, fills,
                  |  (Nautilus)  |   costs, slippage, session mask, caps
                  +---+----+-----+
                      |    |
           signals <--+    +--> orders
                      v    v
         +-----------------+  +-------------------+
         | BaseStrategy    |  | exec (paper|live) |  IBKR via ib_insync
         | (your code)     |  | three-lock gate   |
         +-----------------+  +-------------------+
                                      |
                               +------+-------+
                               | observability|  Prometheus /metrics
                               +------+-------+
                                      |
                              Grafana dashboard
```

The strategy never talks to the broker directly. The engine receives signals, applies hard risk caps (max stop / min target / min target-to-stop ratio), and routes orders through `exec/`. Paper-to-live is a config flag, not a rewrite.

## Guardrails

See `docs/guardrails.md` and `docs/go-live-checklist.md`. Highlights:

- Live order paths require **three locks**: `ALPHA_ASSAY_LIVE=1` env var, `--live` CLI flag, and a `GO_LIVE_CHECKLIST_SIGNED` file in the working directory.
- Backtests model fees, slippage, and realistic latencies. No "it works" claims without live cost assumptions locked.
- All timestamps normalized to `America/Chicago`; session masks asserted in tests.
- Lookahead bias is a release blocker. CI gates on `tests/test_no_lookahead.py`.
- Hard risk caps (configurable, engine-enforced at config-load time): `max_stop_pts`, `min_target_pts`, `min_target_to_stop_ratio`. Config outside these bounds is rejected before any data loads.

## Observability

The project ships a self-contained Grafana + Prometheus stack via Docker Compose. Drop it in via `docker compose up`; the paper-trader exposes `/metrics` on port 9200, Prometheus scrapes, Grafana renders a provisioned dashboard. Metrics cover signal flow (generated / filtered / fired), equity curve, per-trade MAE / MFE / duration, fill slippage, and health (feed freshness, IBKR connection, kill-switch state).

## License

Apache License 2.0. See [`LICENSE`](LICENSE) and [`NOTICE`](NOTICE).

## Contributing

Issues and PRs welcome once the v0.1 contract stabilizes. Until then, treat this as a reference implementation. Coding standards in `docs/coding-standards.md`.

## Related

The reference proprietary strategy implementation (an ES `$TICK` / `$ADD` breadth-fade) lives in a separate private repo. The public framework has no dependency on it; it is one consumer among many.
