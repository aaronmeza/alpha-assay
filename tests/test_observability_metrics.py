from prometheus_client import Counter, Gauge, Histogram

from alpha_assay.observability import metrics as M


def test_metrics_module_exports_every_spec_series():
    expected = {
        "bars_processed_total": Counter,
        "signals_generated_total": Counter,
        "signals_filtered_total": Counter,
        "signals_fired_total": Counter,
        "orders_submitted_total": Counter,
        "orders_filled_total": Counter,
        "fill_slippage_points": Histogram,
        "trade_pnl_points": Histogram,
        "trade_mae_points": Histogram,
        "trade_mfe_points": Histogram,
        "trade_duration_seconds": Histogram,
        "trades_total": Counter,
        "equity_points": Gauge,
        "session_pnl_points": Gauge,
        "drawdown_points": Gauge,
        "position_contracts": Gauge,
        "feed_freshness_seconds": Gauge,
        "ibkr_connected": Gauge,
        "kill_switch_armed": Gauge,
        "kill_switch_trips_total": Counter,
        "in_session": Gauge,
        "signal_eval_seconds": Histogram,
        "bar_to_order_seconds": Histogram,
    }
    for name, cls in expected.items():
        obj = getattr(M, name)
        assert isinstance(obj, cls), f"{name} should be {cls.__name__}, got {type(obj).__name__}"


def test_all_metric_names_prefixed_alpha_assay():
    for attr in dir(M):
        obj = getattr(M, attr)
        if isinstance(obj, (Counter, Gauge, Histogram)):
            assert obj._name.startswith(
                "alpha_assay_"
            ), f"metric {attr} has name {obj._name!r}; must start with alpha_assay_"


def test_signals_filtered_total_has_reason_label():
    labelnames = M.signals_filtered_total._labelnames
    assert "strategy" in labelnames
    assert "filter_name" in labelnames
    assert "reason" in labelnames


def test_orders_submitted_total_has_type_label():
    assert "type" in M.orders_submitted_total._labelnames


def test_feed_freshness_has_feed_label():
    assert "feed" in M.feed_freshness_seconds._labelnames


def test_kill_switch_trips_has_reason_label():
    assert "reason" in M.kill_switch_trips_total._labelnames


def test_start_metrics_server_exposes_http_endpoint():
    """Smoke: start the exporter on a free port, GET /metrics, assert
    our prefix appears in the response body. Uses a dedicated port to
    avoid collisions with any ambient prometheus_client state.
    """
    import socket
    import time
    import urllib.request

    from alpha_assay.observability import start_metrics_server

    # Pick a free port deterministically.
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()

    start_metrics_server(port=port)
    # The server starts in a background thread; give it a moment.
    time.sleep(0.1)
    with urllib.request.urlopen(f"http://127.0.0.1:{port}/metrics", timeout=2.0) as resp:
        body = resp.read().decode()
    assert "alpha_assay_" in body, f"/metrics response should contain our prefix; got first 200 chars: {body[:200]!r}"
