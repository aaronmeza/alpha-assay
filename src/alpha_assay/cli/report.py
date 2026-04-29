# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Aaron Meza
"""`alpha_assay report` command.

Turns a backtest output directory (containing trades.csv +
session_metrics.json, as written by `alpha_assay backtest`) into a
decision-grade report:

- Pairs raw orders into round-trip trades (entry + matched bracket exit).
- Computes per-trade PnL (points and USD), hold time, MAE / MFE.
- Computes standard quant metrics via quantstats (Sharpe, Sortino, MDD,
  Calmar, volatility) plus trade-level summaries (win rate, profit
  factor, avg win/loss).
- Emits live-vs-backtest parity check fields per spec Section 9: per-session
  PnL, trade count, and avg fill time.

Outputs (all written to ``--out``):

- ``per_trade_metrics.csv`` - one row per round-trip trade.
- ``session_metrics.json`` - additive enrichment of the source file
  (every original key preserved; new keys appended).
- ``report.md`` - human-readable summary table.
- ``report.html`` - quantstats tearsheet (when ``--format`` is ``html``
  or ``both``).

The report command is engine-agnostic: it consumes the CSV/JSON files
emitted by the backtest CLI and never imports the engine. This keeps it
fast, reusable across runners, and friendly to manual sanity checks
(point it at any backtest output dir and re-run).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from statistics import median

import click
import pandas as pd

from alpha_assay.data.databento_adapter import load_parquet


@dataclass(frozen=True, slots=True)
class PairedTrade:
    """A round-trip trade reconstructed from the raw orders log.

    All fields are intrinsic to the trade itself; downstream metrics
    (USD PnL, hold seconds, MAE / MFE) derive from these plus the
    instrument multiplier and (optionally) the underlying bar data.

    ``entry_signal_ts`` is the strategy-side timestamp at which the
    signal that triggered this trade was emitted. It is ``None`` for
    backtest output produced before the runner started recording
    signal timestamps; downstream code falls back to a null
    ``fill_latency_seconds`` in that case.
    """

    entry_ts: pd.Timestamp
    exit_ts: pd.Timestamp
    side: str  # "long" or "short"
    entry_price: float
    exit_price: float
    quantity: float
    exit_reason: str  # "target", "stop", "flip", "unknown"
    entry_signal_ts: pd.Timestamp | None = None


def pair_trades(orders: pd.DataFrame) -> list[PairedTrade]:
    """Pair raw orders into round-trip trades.

    Pairing convention (matches the Nautilus runner emission):

    - Each ``market`` order is an entry. ``buy`` entry => long;
      ``sell`` entry => short.
    - The matched exit is the next order in time whose ``order_type``
      is ``limit`` (target) or ``stop_market`` (stop). The exit's
      ``side`` MUST be opposite the entry. Mapping: target hit on a
      long fills as a sell-limit; stop hit on a long fills as a
      buy-stop_market (and vice versa for shorts).
    - Defensive case: if a new ``market`` entry arrives before the
      previous bracket exits (a "flip"), close the prior position at
      the new entry's fill price and start a fresh trade. Mark the
      closed trade's ``exit_reason`` as ``"flip"``. This keeps PnL
      attribution honest even if the runner ever ships position
      reversals through a single market order.

    Returns an empty list if ``orders`` is empty or has no ``market``
    entries. Never raises on malformed input - unmatched entries (no
    exit before EOF) are silently dropped, which mirrors how the
    Nautilus cache behaves on a session-end flat.
    """
    if orders.empty:
        return []

    required = {"timestamp", "side", "price", "quantity", "order_type"}
    missing = required - set(orders.columns)
    if missing:
        raise ValueError(f"orders DataFrame missing required columns: {sorted(missing)}")

    df = orders.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    # `signal_ts` is optional - backtests produced before the runner
    # started emitting it will not have the column. When present,
    # parse it to UTC; when absent, all entries get None.
    if "signal_ts" in df.columns:
        df["signal_ts"] = pd.to_datetime(df["signal_ts"], utc=True, errors="coerce")
    df = df.sort_values("timestamp", kind="stable").reset_index(drop=True)

    paired: list[PairedTrade] = []
    open_entry: dict | None = None

    for _, row in df.iterrows():
        order_type = str(row["order_type"]).lower()
        side = str(row["side"]).lower()
        ts = pd.Timestamp(row["timestamp"])
        price = float(row["price"]) if pd.notna(row["price"]) else float("nan")
        qty = float(row["quantity"])
        raw_signal_ts = row.get("signal_ts") if "signal_ts" in df.columns else None
        if raw_signal_ts is None or pd.isna(raw_signal_ts):
            signal_ts: pd.Timestamp | None = None
        else:
            signal_ts = pd.Timestamp(raw_signal_ts)

        if order_type == "market":
            # New entry. If a previous entry is still open, treat this
            # as a flip: close prior at this market price first.
            if open_entry is not None:
                paired.append(
                    PairedTrade(
                        entry_ts=open_entry["ts"],
                        exit_ts=ts,
                        side=open_entry["side"],
                        entry_price=open_entry["price"],
                        exit_price=price,
                        quantity=open_entry["qty"],
                        exit_reason="flip",
                        entry_signal_ts=open_entry["signal_ts"],
                    )
                )
            entry_side = "long" if side == "buy" else "short"
            open_entry = {
                "ts": ts,
                "side": entry_side,
                "price": price,
                "qty": qty,
                "signal_ts": signal_ts,
            }
        elif order_type in {"limit", "stop_market"}:
            if open_entry is None:
                # Bracket exit with no preceding entry - skip.
                continue
            reason = "target" if order_type == "limit" else "stop"
            paired.append(
                PairedTrade(
                    entry_ts=open_entry["ts"],
                    exit_ts=ts,
                    side=open_entry["side"],
                    entry_price=open_entry["price"],
                    exit_price=price,
                    quantity=open_entry["qty"],
                    exit_reason=reason,
                    entry_signal_ts=open_entry["signal_ts"],
                )
            )
            open_entry = None
        else:
            # Unknown order_type - leave any open entry alone.
            continue

    return paired


def _trade_pnl_points(t: PairedTrade) -> float:
    direction = 1.0 if t.side == "long" else -1.0
    return direction * (t.exit_price - t.entry_price)


def _trade_hold_seconds(t: PairedTrade) -> float:
    return float((t.exit_ts - t.entry_ts).total_seconds())


def _trade_fill_latency_seconds(t: PairedTrade) -> float | None:
    """Seconds between entry-signal emission and entry fill.

    Returns ``None`` when ``entry_signal_ts`` is missing (older backtest
    runs that pre-date signal_ts emission). For a Nautilus backtest
    with cheat-on-open OFF this should equal one bar period (60s for
    1-minute bars); on live paper it should equal the broker
    acknowledgement latency. Spec Section 9 calls for paper-vs-backtest
    alignment within +/- 30s.
    """
    if t.entry_signal_ts is None or pd.isna(t.entry_signal_ts):
        return None
    return float((t.entry_ts - t.entry_signal_ts).total_seconds())


def compute_mae_mfe(
    trade: PairedTrade,
    bars: pd.DataFrame | None,
) -> tuple[float | None, float | None]:
    """Return (MAE, MFE) in price points for a single trade.

    MAE (max adverse excursion) is the worst running PnL during the
    trade; MFE (max favorable excursion) is the best. Both are
    expressed as positive numbers in price points - the sign convention
    is "how many points against / for me" which keeps the columns
    intuitive in the per-trade CSV.

    Returns ``(None, None)`` when ``bars`` is missing or no rows fall
    inside the trade window. The caller decides whether to emit nulls
    or omit the columns.
    """
    if bars is None or bars.empty:
        return (None, None)

    # Bars index is tz-aware Chicago; trade timestamps from the orders
    # log are tz-aware UTC. Normalize both to a tz-aware view before
    # slicing - .loc on a tz-aware index requires a tz-aware key.
    if bars.index.tz is None:
        idx = bars.index.tz_localize("UTC")
    else:
        idx = bars.index.tz_convert("UTC")
    norm = bars.copy()
    norm.index = idx

    entry_ts = pd.Timestamp(trade.entry_ts).tz_convert("UTC")
    exit_ts = pd.Timestamp(trade.exit_ts).tz_convert("UTC")
    window = norm.loc[entry_ts:exit_ts]
    if window.empty:
        return (None, None)

    if trade.side == "long":
        # MAE: lowest low minus entry (negative => against us); flip sign
        adverse = trade.entry_price - float(window["low"].min())
        favorable = float(window["high"].max()) - trade.entry_price
    else:
        adverse = float(window["high"].max()) - trade.entry_price
        favorable = trade.entry_price - float(window["low"].min())

    return (max(adverse, 0.0), max(favorable, 0.0))


def build_per_trade_frame(
    trades: list[PairedTrade],
    instrument_multiplier: float,
    bars: pd.DataFrame | None,
) -> pd.DataFrame:
    """Materialize per-trade rows with PnL, hold time, MAE/MFE, and
    entry fill latency.
    """
    rows: list[dict] = []
    for t in trades:
        pnl_pts = _trade_pnl_points(t)
        pnl_usd = pnl_pts * instrument_multiplier * t.quantity
        hold_s = _trade_hold_seconds(t)
        mae, mfe = compute_mae_mfe(t, bars)
        rows.append(
            {
                "entry_ts": t.entry_ts,
                "exit_ts": t.exit_ts,
                "side": t.side,
                "entry_price": t.entry_price,
                "exit_price": t.exit_price,
                "quantity": t.quantity,
                "pnl_points": pnl_pts,
                "pnl_usd": pnl_usd,
                "hold_seconds": hold_s,
                "mae_points": mae,
                "mfe_points": mfe,
                "exit_reason": t.exit_reason,
                "entry_signal_ts": t.entry_signal_ts,
                "fill_latency_seconds": _trade_fill_latency_seconds(t),
            }
        )
    return pd.DataFrame(rows)


def _safe_div(numerator: float, denominator: float) -> float | None:
    if denominator == 0:
        return None
    return numerator / denominator


def _equity_curve_from_trades(
    per_trade: pd.DataFrame,
    starting_balance: float,
) -> pd.Series:
    """Construct a cumulative-PnL equity curve indexed by exit_ts.

    Used as the input to quantstats. Falls back to a one-point series
    at ``starting_balance`` when no trades exist so quantstats has
    something to work on.
    """
    if per_trade.empty:
        return pd.Series([starting_balance], index=pd.DatetimeIndex([pd.Timestamp.utcnow()]))
    sorted_pt = per_trade.sort_values("exit_ts")
    idx = pd.DatetimeIndex(pd.to_datetime(sorted_pt["exit_ts"], utc=True))
    cum = sorted_pt["pnl_usd"].cumsum().to_numpy()
    return pd.Series(starting_balance + cum, index=idx, name="equity_usd")


def _daily_returns_from_equity(equity: pd.Series) -> pd.Series:
    """Resample the equity curve to daily and compute simple returns.

    quantstats expects a daily Series for its annualization defaults
    (periods=252). Bucketing by calendar day matches how a live-trading
    operator would read the dashboard.
    """
    if equity.empty:
        return pd.Series(dtype=float)
    daily = equity.resample("1D").last().ffill()
    returns = daily.pct_change().dropna()
    return returns


def _reconciled_total_return_pct(
    starting_balance: float,
    final_balance: float,
) -> float | None:
    """Dollar-basis total return: (final - start) / start * 100.

    This is the metric the user actually wants for live-vs-backtest parity:
    the same number the broker statement reports. It is intentionally
    NOT the geometric compound of per-period returns - quantstats'
    ``comp`` answers a different question and diverges from the
    dollar-balance change when the per-period returns Series is built
    from per-trade PnL on a constant base. Keep both numbers separate
    and label them honestly: ``total_return_pct`` is dollar-basis;
    ``compound_return_pct`` is the quantstats compound figure.
    """
    if starting_balance == 0:
        return None
    return ((final_balance - starting_balance) / starting_balance) * 100.0


def _annualized_from_total_and_days(
    total_return_pct: float | None,
    days_elapsed: float | None,
) -> float | None:
    """Annualize a dollar-basis total return over its actual calendar span.

    Uses simple compounding: ``(1 + r)^(365/days) - 1``. Falls back to
    ``None`` when the inputs are missing or the elapsed span is too
    small to annualize sensibly (< 1 day). Sign is preserved so a
    negative total return annualizes to a negative number.
    """
    if total_return_pct is None or days_elapsed is None or days_elapsed < 1.0:
        return None
    r = total_return_pct / 100.0
    base = 1.0 + r
    if base <= 0:
        # 100%+ loss; annualization is undefined. Return the floor.
        return -100.0
    return ((base ** (365.0 / days_elapsed)) - 1.0) * 100.0


def _elapsed_days_from_per_trade(per_trade: pd.DataFrame) -> float | None:
    """Calendar-day span between first entry and last exit, inclusive."""
    if per_trade.empty:
        return None
    entries = pd.to_datetime(per_trade["entry_ts"], utc=True)
    exits = pd.to_datetime(per_trade["exit_ts"], utc=True)
    span = exits.max() - entries.min()
    days = span.total_seconds() / 86400.0
    return days if days > 0 else None


def _quant_metrics(
    equity: pd.Series,
    per_trade: pd.DataFrame,
    starting_balance: float,
    final_balance: float,
) -> dict[str, float | None]:
    """Compute the headline quant metrics.

    ``total_return_pct`` is computed directly from the dollar-balance
    change so it always reconciles with ``final_balance_usd`` on the
    enriched session_metrics. ``annualized_return_pct`` is derived
    from that total over the trade-spanning calendar window.

    The remaining metrics (Sharpe, Sortino, MDD, Calmar, volatility)
    come from quantstats over a daily-resampled equity curve. Each is
    wrapped in try/except so a single quantstats edge case (e.g.
    constant equity => zero variance => ``RuntimeWarning`` cast to
    NaN) does not nuke the whole report.

    A ``compound_return_pct`` field preserves the quantstats
    compound-return number for completeness; it can diverge from
    ``total_return_pct`` when per-trade returns compound differently
    than dollar PnL accumulates, and labelling it separately keeps
    both numbers honest.
    """
    try:
        import quantstats.stats as qs_stats
    except ImportError as exc:  # pragma: no cover - guarded at command entry
        raise click.ClickException(
            "alpha_assay report requires quantstats; install with: " 'pip install -e ".[report]"'
        ) from exc

    returns = _daily_returns_from_equity(equity)
    out: dict[str, float | None] = {
        "sharpe_ratio": None,
        "sortino_ratio": None,
        "max_drawdown": None,
        "max_drawdown_duration_days": None,
        "total_return_pct": None,
        "annualized_return_pct": None,
        "compound_return_pct": None,
        "volatility_annualized_pct": None,
        "calmar_ratio": None,
    }

    # Dollar-basis total return: always computed, regardless of how
    # many daily observations the equity curve has. This is the field
    # live-vs-backtest parity checks against.
    out["total_return_pct"] = _reconciled_total_return_pct(starting_balance, final_balance)

    # Annualized return: derive from total + actual elapsed days.
    days_elapsed = _elapsed_days_from_per_trade(per_trade)
    out["annualized_return_pct"] = _annualized_from_total_and_days(
        out["total_return_pct"], days_elapsed
    )

    if returns.empty or len(returns) < 2:
        # Daily-return-based metrics are degenerate under fewer than
        # two daily observations; leave them as None.
        return out

    def _coerce(value: float | None) -> float | None:
        if value is None:
            return None
        try:
            v = float(value)
        except (TypeError, ValueError):
            return None
        if pd.isna(v):
            return None
        return v

    try:
        out["sharpe_ratio"] = _coerce(qs_stats.sharpe(returns))
    except Exception:
        pass
    try:
        out["sortino_ratio"] = _coerce(qs_stats.sortino(returns))
    except Exception:
        pass
    try:
        out["max_drawdown"] = _coerce(qs_stats.max_drawdown(equity))
    except Exception:
        pass
    try:
        out["calmar_ratio"] = _coerce(qs_stats.calmar(returns))
    except Exception:
        pass
    try:
        out["volatility_annualized_pct"] = _coerce(qs_stats.volatility(returns) * 100.0)
    except Exception:
        pass
    try:
        # Preserve quantstats' compound-return figure under its own key
        # so anything that wants the geometric number can still find it.
        out["compound_return_pct"] = _coerce(qs_stats.comp(returns) * 100.0)
    except Exception:
        pass

    # Drawdown duration: peek at the longest stretch between equity
    # high-water-mark recoveries in days. quantstats' drawdown_details
    # would handle this but its API is heavier than what we need.
    try:
        running_max = equity.cummax()
        drawdown = (equity - running_max) / running_max
        in_dd = drawdown < 0
        if in_dd.any():
            longest = pd.Timedelta(0)
            current_start: pd.Timestamp | None = None
            for ts, flag in in_dd.items():
                if flag and current_start is None:
                    current_start = ts
                elif not flag and current_start is not None:
                    span = ts - current_start
                    if span > longest:
                        longest = span
                    current_start = None
            if current_start is not None:
                span = in_dd.index[-1] - current_start
                if span > longest:
                    longest = span
            out["max_drawdown_duration_days"] = longest.total_seconds() / 86400.0
    except Exception:
        pass

    return out


def _trade_metrics(per_trade: pd.DataFrame) -> dict[str, float | int | None]:
    """Win rate, profit factor, avg win/loss, hold time stats."""
    if per_trade.empty:
        return {
            "n_trades_total": 0,
            "n_trades_long": 0,
            "n_trades_short": 0,
            "win_rate": None,
            "profit_factor": None,
            "avg_win_usd": None,
            "avg_loss_usd": None,
            "avg_trade_pnl_usd": None,
            "largest_win_usd": None,
            "largest_loss_usd": None,
            "avg_hold_seconds": None,
            "median_hold_seconds": None,
            "avg_mae_points": None,
            "avg_mfe_points": None,
        }

    pnls = per_trade["pnl_usd"]
    wins = pnls[pnls > 0]
    losses = pnls[pnls < 0]
    longs = (per_trade["side"] == "long").sum()
    shorts = (per_trade["side"] == "short").sum()
    profit_factor = _safe_div(float(wins.sum()), float(abs(losses.sum())))

    mae_series = per_trade.get("mae_points")
    mfe_series = per_trade.get("mfe_points")
    avg_mae = (
        float(mae_series.dropna().mean())
        if mae_series is not None and not mae_series.dropna().empty
        else None
    )
    avg_mfe = (
        float(mfe_series.dropna().mean())
        if mfe_series is not None and not mfe_series.dropna().empty
        else None
    )

    return {
        "n_trades_total": int(len(per_trade)),
        "n_trades_long": int(longs),
        "n_trades_short": int(shorts),
        "win_rate": float(len(wins)) / float(len(per_trade)) if len(per_trade) else None,
        "profit_factor": profit_factor,
        "avg_win_usd": float(wins.mean()) if not wins.empty else None,
        "avg_loss_usd": float(losses.mean()) if not losses.empty else None,
        "avg_trade_pnl_usd": float(pnls.mean()),
        "largest_win_usd": float(pnls.max()) if not pnls.empty else None,
        "largest_loss_usd": float(pnls.min()) if not pnls.empty else None,
        "avg_hold_seconds": float(per_trade["hold_seconds"].mean()),
        "median_hold_seconds": float(median(per_trade["hold_seconds"])),
        "avg_mae_points": avg_mae,
        "avg_mfe_points": avg_mfe,
    }


def _per_session_metrics(per_trade: pd.DataFrame) -> dict[str, dict[str, float | int | None]]:
    """Bucket trades by calendar day for the live-vs-backtest parity check.

    Spec Section 9 calls for per-session PnL +/- 20%, trade count
    +/- 20%, fill times +/- 30s. We emit those three dicts keyed by
    ``YYYY-MM-DD``.

    ``per_session_avg_fill_time_seconds`` is the mean of
    ``fill_latency_seconds`` across each session's trades. When the
    runner did not emit signal_ts (older runs), the per-trade column
    is all-null and the per-session value falls through to ``None``
    rather than 0.0 - a null-safe signal to downstream parity-check
    code that the data is unavailable.
    """
    if per_trade.empty:
        return {
            "per_session_pnl_usd": {},
            "per_session_trade_count": {},
            "per_session_avg_fill_time_seconds": {},
        }
    df = per_trade.copy()
    # Bucket by exit timestamp (when PnL is realized).
    df["session_date"] = pd.to_datetime(df["exit_ts"], utc=True).dt.date.astype(str)
    pnl_by_session = df.groupby("session_date")["pnl_usd"].sum().to_dict()
    count_by_session = df.groupby("session_date").size().to_dict()
    # Per-session avg fill latency: mean over non-null fill_latency_seconds.
    # Sessions where every trade lacks signal_ts get None.
    fill_time_by_session: dict[str, float | None] = {}
    if "fill_latency_seconds" in df.columns:
        latency_col = pd.to_numeric(df["fill_latency_seconds"], errors="coerce")
        for session_date, group in df.groupby("session_date"):
            session_latencies = latency_col.loc[group.index].dropna()
            fill_time_by_session[str(session_date)] = (
                float(session_latencies.mean()) if not session_latencies.empty else None
            )
    else:
        fill_time_by_session = {str(k): None for k in count_by_session}
    return {
        "per_session_pnl_usd": {k: float(v) for k, v in pnl_by_session.items()},
        "per_session_trade_count": {k: int(v) for k, v in count_by_session.items()},
        "per_session_avg_fill_time_seconds": fill_time_by_session,
    }


def _format_md_report(
    enriched: dict,
    per_trade: pd.DataFrame,
) -> str:
    """Render a Markdown summary of the enriched metrics + per-trade table.

    Markdown is the always-on default because it diff-reviews well in
    PRs and renders inline in GitHub. The HTML tearsheet from
    quantstats is the heavyweight visual companion.
    """
    lines: list[str] = []
    now = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    lines.append("# alpha_assay backtest report")
    lines.append("")
    lines.append(f"_Generated: {now}_")
    lines.append("")

    lines.append("## Run summary")
    lines.append("")
    run_keys = [
        "run_status",
        "submitted_signals",
        "orders_submitted",
        "signals_filtered_risk_cap",
        "starting_balance_usd",
        "final_balance_usd",
        "instrument_multiplier",
    ]
    lines.append("| metric | value |")
    lines.append("| --- | --- |")
    for k in run_keys:
        if k in enriched:
            lines.append(f"| {k} | {enriched[k]} |")
    lines.append("")

    lines.append("## Standard quant metrics")
    lines.append("")
    quant_keys = [
        "sharpe_ratio",
        "sortino_ratio",
        "calmar_ratio",
        "max_drawdown",
        "max_drawdown_duration_days",
        "total_return_pct",
        "annualized_return_pct",
        "compound_return_pct",
        "volatility_annualized_pct",
    ]
    lines.append("| metric | value |")
    lines.append("| --- | --- |")
    for k in quant_keys:
        v = enriched.get(k)
        lines.append(f"| {k} | {v if v is not None else 'n/a'} |")
    lines.append("")

    lines.append("## Trade-level metrics")
    lines.append("")
    trade_keys = [
        "n_trades_total",
        "n_trades_long",
        "n_trades_short",
        "win_rate",
        "profit_factor",
        "avg_win_usd",
        "avg_loss_usd",
        "avg_trade_pnl_usd",
        "largest_win_usd",
        "largest_loss_usd",
        "avg_hold_seconds",
        "median_hold_seconds",
        "avg_mae_points",
        "avg_mfe_points",
    ]
    lines.append("| metric | value |")
    lines.append("| --- | --- |")
    for k in trade_keys:
        v = enriched.get(k)
        lines.append(f"| {k} | {v if v is not None else 'n/a'} |")
    lines.append("")

    lines.append("## Per-session breakdown (live-vs-backtest parity check)")
    lines.append("")
    pnl = enriched.get("per_session_pnl_usd") or {}
    counts = enriched.get("per_session_trade_count") or {}
    fills = enriched.get("per_session_avg_fill_time_seconds") or {}
    if pnl:
        lines.append("| session | pnl_usd | n_trades | avg_fill_time_seconds |")
        lines.append("| --- | --- | --- | --- |")
        for date in sorted(pnl):
            fill_val = fills.get(date)
            fill_cell = f"{fill_val:.2f}" if isinstance(fill_val, (int, float)) else "n/a"
            lines.append(f"| {date} | {pnl[date]:.2f} | {counts.get(date, 0)} | {fill_cell} |")
    else:
        lines.append("_No trades recorded._")
    lines.append("")

    lines.append("## Per-trade table")
    lines.append("")
    if per_trade.empty:
        lines.append("_No round-trip trades reconstructed from orders.csv._")
    else:
        cols = [
            "entry_ts",
            "exit_ts",
            "side",
            "entry_price",
            "exit_price",
            "pnl_points",
            "pnl_usd",
            "hold_seconds",
            "exit_reason",
        ]
        lines.append("| " + " | ".join(cols) + " |")
        lines.append("| " + " | ".join(["---"] * len(cols)) + " |")
        for _, row in per_trade.iterrows():
            cells = []
            for c in cols:
                val = row[c]
                if isinstance(val, float):
                    cells.append(f"{val:.4f}" if c != "pnl_usd" else f"{val:.2f}")
                else:
                    cells.append(str(val))
            lines.append("| " + " | ".join(cells) + " |")

    lines.append("")
    return "\n".join(lines)


def _coerce_session_metrics(metrics: dict) -> dict:
    """Re-cast numeric values that were JSON-serialized as strings.

    The original ``json.dumps(..., default=str)`` writes Decimals and
    Timestamps as strings; balance fields land back as strings, which
    breaks downstream arithmetic. Rehydrate the few numeric fields
    we care about.
    """
    out = dict(metrics)
    for key in (
        "final_balance_usd",
        "starting_balance_usd",
        "submitted_signals",
        "orders_submitted",
        "signals_filtered_risk_cap",
    ):
        v = out.get(key)
        if isinstance(v, str):
            try:
                out[key] = float(v) if "." in v else int(v)
            except ValueError:
                pass
    return out


def _quantstats_html(equity: pd.Series, out_path: Path) -> bool:
    """Best-effort render of the quantstats HTML tearsheet.

    Returns ``True`` on success, ``False`` on failure (the caller
    surfaces a friendly note in the CLI output). Wrapped because
    quantstats relies on yfinance for its benchmark download by
    default and that path can fail in offline CI.
    """
    try:
        import quantstats.reports as qs_reports
    except ImportError:
        return False

    returns = _daily_returns_from_equity(equity)
    if returns.empty or len(returns) < 2:
        return False

    try:
        qs_reports.html(
            returns, output=str(out_path), download_filename=None, title="alpha_assay backtest"
        )
    except TypeError:
        # Older signatures don't accept ``download_filename``; retry.
        try:
            qs_reports.html(returns, output=str(out_path), title="alpha_assay backtest")
        except Exception:
            return False
    except Exception:
        return False
    return out_path.exists()


@click.command()
@click.option(
    "--in",
    "in_dir",
    required=True,
    type=click.Path(exists=True, file_okay=False),
    help="Input directory holding trades.csv + session_metrics.json.",
)
@click.option(
    "--out",
    "out_dir",
    type=click.Path(file_okay=False),
    default=None,
    help="Output directory (default: same as --in).",
)
@click.option(
    "--format",
    "fmt",
    type=click.Choice(["html", "md", "both"]),
    default="both",
    help="Report format. 'html' adds a quantstats tearsheet.",
)
@click.option(
    "--instrument-multiplier",
    type=float,
    default=50.0,
    help="Dollars per point (default 50.0 for ES).",
)
@click.option(
    "--data",
    "data_path",
    type=click.Path(exists=True, dir_okay=False),
    default=None,
    help="Optional parquet path used to compute per-trade MAE/MFE.",
)
@click.option(
    "--starting-balance",
    "starting_balance_opt",
    type=float,
    default=100_000.0,
    show_default=True,
    help=(
        "Account starting balance used as the basis for total_return_pct. "
        "If session_metrics.json already contains starting_balance_usd, that "
        "value wins; this flag is the fallback."
    ),
)
def report(
    in_dir: str,
    out_dir: str | None,
    fmt: str,
    instrument_multiplier: float,
    data_path: str | None,
    starting_balance_opt: float,
) -> None:
    """Build a per-trade + per-session report from a backtest output dir."""
    try:
        import quantstats  # noqa: F401  -- guard the friendly error
    except ImportError as exc:
        raise click.ClickException(
            "alpha_assay report requires quantstats; install with: " 'pip install -e ".[report]"'
        ) from exc

    in_path = Path(in_dir)
    out_path = Path(out_dir) if out_dir else in_path
    out_path.mkdir(parents=True, exist_ok=True)

    trades_csv = in_path / "trades.csv"
    metrics_json = in_path / "session_metrics.json"
    if not trades_csv.exists():
        raise click.ClickException(f"missing trades.csv in {in_path}")
    if not metrics_json.exists():
        raise click.ClickException(f"missing session_metrics.json in {in_path}")

    orders = pd.read_csv(trades_csv)
    base_metrics = _coerce_session_metrics(json.loads(metrics_json.read_text()))

    bars: pd.DataFrame | None = None
    if data_path:
        bars = load_parquet(data_path)

    paired = pair_trades(orders)
    per_trade = build_per_trade_frame(paired, instrument_multiplier, bars)

    # Source of truth for starting_balance: prefer the value embedded
    # in session_metrics.json (so a future runner that records its own
    # starting balance wins), fall back to the CLI flag. Coerce to
    # float defensively - JSON round-trips can land it as a string.
    starting_balance = starting_balance_opt
    sb_from_metrics = base_metrics.get("starting_balance_usd")
    if sb_from_metrics is not None:
        try:
            starting_balance = float(sb_from_metrics)
        except (TypeError, ValueError):
            pass

    # Final balance: mandatory for the dollar-basis return. If absent
    # (e.g. an old session_metrics.json), fall back to the equity
    # curve's tail so we at least produce a number.
    def _pnl_sum_or_zero(df: pd.DataFrame) -> float:
        if df.empty or "pnl_usd" not in df.columns:
            return 0.0
        return float(df["pnl_usd"].sum())

    final_balance_raw = base_metrics.get("final_balance_usd")
    if final_balance_raw is not None:
        try:
            final_balance = float(final_balance_raw)
        except (TypeError, ValueError):
            final_balance = starting_balance + _pnl_sum_or_zero(per_trade)
    else:
        final_balance = starting_balance + _pnl_sum_or_zero(per_trade)

    equity = _equity_curve_from_trades(per_trade, starting_balance)

    quant = _quant_metrics(equity, per_trade, starting_balance, final_balance)
    trade_summary = _trade_metrics(per_trade)
    per_session = _per_session_metrics(per_trade)

    enriched = dict(base_metrics)
    enriched["instrument_multiplier"] = instrument_multiplier
    enriched["starting_balance_usd"] = starting_balance
    enriched.update(quant)
    enriched.update(trade_summary)
    enriched.update(per_session)

    # Write artifacts.
    per_trade_path = out_path / "per_trade_metrics.csv"
    per_trade.to_csv(per_trade_path, index=False)

    enriched_path = out_path / "session_metrics.json"
    enriched_path.write_text(json.dumps(enriched, default=str, indent=2))

    md_text = _format_md_report(enriched, per_trade)
    md_path = out_path / "report.md"
    if fmt in {"md", "both"}:
        md_path.write_text(md_text)

    html_written = False
    if fmt in {"html", "both"}:
        html_path = out_path / "report.html"
        html_written = _quantstats_html(equity, html_path)

    click.echo(f"Wrote {len(per_trade)} round-trip trades to {per_trade_path}")
    click.echo(f"Enriched session_metrics.json at {enriched_path}")
    if fmt in {"md", "both"}:
        click.echo(f"Markdown report at {md_path}")
    if fmt in {"html", "both"}:
        if html_written:
            click.echo(f"HTML tearsheet at {out_path / 'report.html'}")
        else:
            click.echo("HTML tearsheet skipped (quantstats render failed or insufficient data)")


__all__ = [
    "PairedTrade",
    "_annualized_from_total_and_days",
    "_elapsed_days_from_per_trade",
    "_reconciled_total_return_pct",
    "build_per_trade_frame",
    "compute_mae_mfe",
    "pair_trades",
    "report",
]
