import csv
import datetime
import pathlib
import subprocess

ROOT = pathlib.Path(__file__).resolve().parent.parent

EXPECTED_COLUMNS = [
    "timestamp",
    "ES_open",
    "ES_high",
    "ES_low",
    "ES_close",
    "ES_volume",
    "TICK",
    "ADD",
]


def test_make_sample_produces_fixture():
    fixture = ROOT / "tests" / "fixtures" / "sample_2d.csv"
    if fixture.exists():
        fixture.unlink()
    result = subprocess.run(
        ["make", "sample"], cwd=ROOT, capture_output=True, text=True, timeout=60
    )
    assert result.returncode == 0, result.stderr
    assert fixture.exists()
    assert fixture.stat().st_size > 0

    with fixture.open(newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = list(reader)

    assert header == EXPECTED_COLUMNS, f"unexpected columns: {header}"

    row_count = len(rows)
    assert (
        300 <= row_count <= 2000
    ), f"row count {row_count} outside expected band [300, 2000] for --days 2"

    first_ts = rows[0][0]
    # Must parse as tz-aware ISO 8601 (offset like +00:00 / -05:00 / -06:00).
    parsed = datetime.datetime.fromisoformat(first_ts)
    assert parsed.tzinfo is not None, f"first timestamp {first_ts!r} is not tz-aware"
