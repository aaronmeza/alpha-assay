import pandas as pd

from alpha_assay.filters.stop_run import stop_run


def _bars(closes, volumes):
    idx = pd.date_range("2026-04-21 09:00", periods=len(closes), freq="1min", tz="America/Chicago")
    return pd.DataFrame({"close": closes, "volume": volumes}, index=idx)


def test_flat_volume_and_price_no_spike():
    bars = _bars([100] * 20, [50] * 20)
    mask = stop_run(bars, lookback=10)
    assert not mask.any()


def test_volume_spike_with_down_move_is_flush():
    closes = [100] * 15 + [95, 95, 95, 95, 95]  # sudden drop at index 15
    volumes = [50] * 15 + [500, 100, 100, 100, 100]  # volume spike at 15
    bars = _bars(closes, volumes)
    mask = stop_run(bars, lookback=10, vol_z_threshold=2.0, price_z_threshold=2.0)
    # Bar 15 spiked; subsequent bars are not spikes themselves
    assert bool(mask.iloc[15]) is True


def test_price_spike_no_volume_not_flush():
    closes = [100] * 15 + [110, 110, 110, 110, 110]  # price spike
    volumes = [50] * 20  # flat volume
    bars = _bars(closes, volumes)
    mask = stop_run(bars, lookback=10, vol_z_threshold=2.0, price_z_threshold=2.0)
    assert not mask.any()


def test_needs_both_price_and_volume_spike():
    closes = [100] * 15 + [95, 95, 95, 95, 95]
    volumes = [50] * 20
    bars = _bars(closes, volumes)
    mask = stop_run(bars, lookback=10, vol_z_threshold=2.0, price_z_threshold=2.0)
    assert not mask.any()


def test_output_aligned_to_index():
    bars = _bars([100] * 20, [50] * 20)
    mask = stop_run(bars, lookback=10)
    assert list(mask.index) == list(bars.index)
    assert mask.dtype == bool
