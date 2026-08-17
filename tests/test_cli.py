"""CLI wiring tests. No pipeline execution, since the data is gitignored."""

from __future__ import annotations

import pytest
from typer.testing import CliRunner

from tfire.cli import app

runner = CliRunner()


def test_every_command_is_registered() -> None:
    listed = runner.invoke(app, ["--help"]).output
    for command in (
        "build-positives",
        "build-grid",
        "build-samples",
        "extract-features",
        "fetch-era5",
        "fetch-landsat",
        "predict",
        "check-access",
    ):
        assert command in listed, f"{command} is missing from the command list"
        assert runner.invoke(app, [command, "--help"]).exit_code == 0


@pytest.mark.parametrize(
    ("command", "flag", "value"),
    [
        ("check-access", "--source", "nope"),
        ("extract-features", "--category", "nope"),
        ("fetch-era5", "--year", "1899"),
        ("fetch-landsat", "--year", "1899"),
    ],
)
def test_an_unknown_name_is_rejected_before_any_work_starts(
    command: str, flag: str, value: str
) -> None:
    result = runner.invoke(app, [command, flag, value])
    assert result.exit_code != 0
    assert value in result.output
