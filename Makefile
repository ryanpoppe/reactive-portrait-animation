.PHONY: help install install-dev test lint fmt fmt-check typecheck check run-info run-demo clean

help:
	@python -c "print('Available targets:'); print('  install      Install project dependencies with uv'); print('  install-dev  Install all dependency groups with uv'); print('  test         Run pytest'); print('  lint         Run ruff checks'); print('  fmt          Format code with ruff'); print('  fmt-check    Check formatting without modifying files'); print('  typecheck    Run mypy'); print('  check        Run lint, format check, typecheck, and tests'); print('  run-info     Show active RPA configuration'); print('  run-demo     Run the mock RPA demo'); print('  clean        Remove local caches and build artifacts')"

install:
	uv sync

install-dev:
	uv sync --all-groups

test:
	uv run pytest

lint:
	uv run ruff check .

fmt:
	uv run ruff format .

fmt-check:
	uv run ruff format . --check

typecheck:
	uv run mypy .

check: lint fmt-check typecheck test

run-info:
	uv run rpa info

run-demo:
	uv run rpa demo

clean:
	python -c "from pathlib import Path; import shutil; paths=['.pytest_cache','.mypy_cache','.ruff_cache','htmlcov','build','dist']; files=['.coverage']; [shutil.rmtree(Path(p), ignore_errors=True) for p in paths]; [Path(f).unlink(missing_ok=True) for f in files]"
