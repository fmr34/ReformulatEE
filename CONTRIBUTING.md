# Contributing to ReformulatEE

## Getting Started

```bash
git clone https://github.com/fmr34/ReformulatEE.git
cd ReformulatEE
pip install -e ".[dev]"
git checkout -b feature/your-feature-name
```

## Code Style

```bash
black src/          # format
ruff check src/     # lint
ruff check --fix src/
```

New code should include type hints and docstrings.

## Tests

```bash
pytest tests/ -v
```

## Pull Request Checklist

- [ ] Tests pass (`pytest`)
- [ ] Code formatted (`black`) and linted (`ruff`)
- [ ] Docstrings updated for new functions
- [ ] README updated if adding user-facing features

## Key Modules

| Module | Purpose |
|--------|---------|
| `src/rl/inference.py` | Main pipeline: generation → scoring → filtering |
| `src/ee/reward.py` | Epistemic effectiveness scoring |
| `src/classifier/tractability_local.py` | Ridge classifier for tractability |
| `src/corpus/index.py` | BM25 + semantic hybrid search |
| `src/db/historico.py` | History and feedback persistence |

## Reporting Issues

Include: description, steps to reproduce, expected vs. actual behavior, Python version and OS.

## Questions?

Open a [GitHub Discussion](https://github.com/fmr34/ReformulatEE/discussions) or file an [Issue](https://github.com/fmr34/ReformulatEE/issues).
