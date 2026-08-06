"""CLI wiring tests. No pipeline execution, since the data is gitignored."""

from __future__ import annotations

from typer.testing import CliRunner

from tfire.cli import app

runner = CliRunner()


def test_every_command_is_registered() -> None:
    listed = runner.invoke(app, ["--help"]).output
    for command in ("build-positives", "build-grid", "check-access"):
        assert command in listed, f"{command} is missing from the command list"
        assert runner.invoke(app, [command, "--help"]).exit_code == 0


def test_check_access_rejects_an_unknown_source() -> None:
    result = runner.invoke(app, ["check-access", "--source", "nope"])
    assert result.exit_code != 0
    assert "nope" in result.output
