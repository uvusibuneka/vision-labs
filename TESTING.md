# Monorepo: how to run tests (uv + ruff + pytest)

Repo root is `КЗ/` (this folder). All labs live under `src/labN`,
all tests under `tests/labN`.

## Setup (once, from repo root)

```bash
uv sync --group dev
```

## Run tests

```bash
# all labs
uv run pytest
# only LW1
uv run pytest tests/lab1 -v
```

Current Red stage for LW1: `51 failed, 1 passed` is expected
(failures are `NotImplementedError` in `src/lab1/erosion.py`).

## Lint / format

```bash
uv run ruff check .
uv run ruff format --check .
```
