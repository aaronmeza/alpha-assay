from pathlib import Path

from nautilus_trader.common.config import LoggingConfig

from alpha_assay.engine import logging as engine_logging


def test_logging_config_exports_bypass_logging():
    assert isinstance(engine_logging.LOGGING_CONFIG, LoggingConfig)
    # Can't introspect the internal bool directly; round-trip via repr.
    assert (
        "bypass_logging=True" in repr(engine_logging.LOGGING_CONFIG)
        or getattr(engine_logging.LOGGING_CONFIG, "bypass_logging", False) is True
    )


def test_reset_for_test_is_callable_noop():
    # reset_for_test is a no-op hook the test suite can call between
    # BacktestEngine instantiations. Must not raise.
    engine_logging.reset_for_test()
    engine_logging.reset_for_test()


def test_module_documents_rust_singleton_lesson():
    # Regression guard: the module docstring must cite the ADR Appendix A
    # constraint so the next reader understands why bypass_logging=True
    # is non-negotiable.
    source = Path(engine_logging.__file__).read_text()
    assert "Rust" in source or "singleton" in source, (
        "engine/logging.py must document the Rust-singleton constraint " "(ADR Appendix A)"
    )
    assert "Appendix A" in source, "engine/logging.py must reference ADR Appendix A for provenance"
