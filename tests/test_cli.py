from typer.testing import CliRunner

from reactive_portrait_animation.cli import app

runner = CliRunner()


def test_info_command() -> None:
    result = runner.invoke(app, ["info"])

    assert result.exit_code == 0
    assert "Environment:" in result.stdout
    assert "LLM provider:" in result.stdout


def test_demo_command() -> None:
    result = runner.invoke(app, ["demo"])

    assert result.exit_code == 0
    assert "Observation:" in result.stdout
    assert "Speech chunks:" in result.stdout
