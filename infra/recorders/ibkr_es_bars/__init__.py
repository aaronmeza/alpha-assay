# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Aaron Meza
"""forward ES 1-min bar recorder.

Sibling to the breadth recorder (``ibkr_breadth``). Subscribes to
ES futures 1-min OHLCV bars via :class:`alpha_assay.data.ibkr_adapter.IBKRAdapter`
and writes daily parquet shards in the canonical schema accepted by
:func:`alpha_assay.data.databento_adapter.load_parquet`.
"""
