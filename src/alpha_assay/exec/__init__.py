# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Aaron Meza
"""Order-submission WRITE path for AlphaAssay.

This subpackage houses execution adapters. The IBKR flavor lives in
``alpha_assay.exec.ibkr`` and is the only one wired for v0.1.

Paper execution is the default. Flipping to live requires ALL THREE
locks engaged (env var + CLI flag + on-disk checklist file). See spec
Section 10 Risk 3 and the module docstring of ``ibkr.py``.

Note on naming: this subpackage is called ``exec`` even though that
matches the builtin ``exec()`` name; Python's attribute lookup on the
``alpha_assay`` package resolves the subpackage fine in import
contexts. No rename required.
"""
