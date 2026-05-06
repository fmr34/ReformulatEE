# Contributing to ReformulatEE

Thank you for your interest in contributing! This document provides guidelines and instructions for contributing to ReformulatEE.

## 🤝 Ways to Contribute

### Code Contributions
- Bug fixes and issue resolution
- Performance improvements
- New features (semantic search improvements, new scoring metrics, etc.)
- Documentation improvements

### Dataset & Research Contributions
- Expanding the curated research questions dataset
- Annotating new domain-specific datasets
- Human evaluation of reformulated questions
- Domain expertise for corpus expansion

### Other Contributions
- Writing guides and tutorials
- Creating test cases
- Language support (new translation models)
- Integration examples (Colab notebooks, deployment guides)

## 📋 Getting Started

### 1. Fork & Clone

```bash
git clone https://github.com/fmr34/reformulatee.git
cd reformulatee
git remote add upstream https://github.com/original-author/reformulatee.git
```

### 2. Setup Development Environment

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

### 3. Create a Feature Branch

```bash
git checkout -b feature/your-feature-name
```

## 🧪 Development Workflow

### Code Style

We use **Black** (formatting) and **Ruff** (linting):

```bash
# Format code
black src/

# Check linting
ruff check src/

# Fix common issues
ruff check --fix src/
```

### Type Hints

New code should include type hints:

```python
from typing import Optional, Dict, List

def compute_ee(question: str, corpus_index: Any) -> Dict[str, float]:
    """Compute epistemic effectiveness scores."""
    pass
```

### Testing

Run existing tests:

```bash
pytest tests/ -v
```

Add tests for new features:

```bash
# tests/test_your_feature.py
import pytest
from src.your_module import your_function

def test_your_function():
    result = your_function("input")
    assert result == "expected"
```

### Documentation

Document new functions and modules:

```python
def reformulate_question(q: str, n: int = 8) -> List[str]:
    """
    Generate n candidate reformulations of a research question.
    
    Args:
        q: Original research question
        n: Number of candidates to generate (default: 8)
    
    Returns:
        List of reformulated questions, sorted by epistemic effectiveness
    
    Raises:
        ValueError: If n < 1 or question is empty
    
    Examples:
        >>> candidates = reformulate_question("What is life?", n=5)
        >>> len(candidates)
        5
    """
    pass
```

## 🔄 Pull Request Process

### Before Submitting

1. **Sync with upstream:**
   ```bash
   git fetch upstream
   git rebase upstream/main
   ```

2. **Run tests locally:**
   ```bash
   pytest tests/ -v
   ruff check src/
   black --check src/
   ```

3. **Commit with clear messages:**
   ```bash
   git commit -m "Add semantic re-ranking to hybrid search (#123)"
   ```

### PR Checklist

- [ ] Tests pass locally (`pytest`)
- [ ] Code is formatted (`black`)
- [ ] Linting passes (`ruff`)
- [ ] Type hints added (where applicable)
- [ ] Docstrings updated
- [ ] README updated (if new features)
- [ ] Linked to related issues

### PR Template

```markdown
## Description
Brief description of changes

## Related Issue
Closes #123

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Performance improvement
- [ ] Documentation
- [ ] Dataset expansion

## Testing
Describe how you tested these changes

## Metrics (if applicable)
- EE improvement: +X%
- Speed improvement: Yms → Zms
- Memory reduction: X MB → Y MB
```

## 🎯 Priority Areas for Contribution

### High Impact
1. **Human evaluation pipeline** — assess reformulation quality
2. **Domain-specific corpus expansion** — add papers in specific fields
3. **Evaluation metrics** — implement ROUGE, BERTScore comparisons
4. **Dataset curation** — expand `data/rl/dpo_*.jsonl` files

### Medium Impact
1. Language support (Spanish, French, German, etc.)
2. Semantic search improvements (re-ranking algorithms)
3. Performance optimizations
4. Additional unit tests

### Low Priority
1. UI improvements (unless breaking)
2. Documentation typos
3. Minor refactorings

## 🐛 Reporting Issues

### Bug Reports

Include:
- Clear description of the bug
- Steps to reproduce
- Expected vs actual behavior
- Python version, OS, environment
- Minimal reproducible example (if possible)

```markdown
## Bug Description
The semantic search returns duplicate papers.

## Steps to Reproduce
1. Run `app.py`
2. Enter a question: "What is consciousness?"
3. Check "Ver candidatos"
4. Observe: Papers #3 and #5 are the same

## Expected
Each candidate should be unique

## Environment
- Python: 3.11
- OS: Windows 11
- GPU: Intel Arc
```

### Feature Requests

Include:
- Clear use case / motivation
- Proposed implementation (optional)
- Examples of desired behavior

```markdown
## Feature Request
Add support for custom scoring weights

## Motivation
Currently, EE is hardcoded as 0.05·R + 0.05·T + 0.90·NT.
Researchers want to tune these weights for their domain.

## Proposed Solution
Add `BETA1`, `BETA2`, `BETA3` parameters to config
```

## 📚 Architecture Overview for Contributors

### Key Modules

| Module | Purpose |
|---|---|
| `src/rl/inference.py` | Main reformulation pipeline (generation → scoring → filtering) |
| `src/ee/reward.py` | Epistemic effectiveness scoring logic |
| `src/classifier/tractability_local.py` | Ridge regression for tractability prediction |
| `src/corpus/index.py` | BM25 + semantic hybrid search |
| `src/dataset/prepare_dpo.py` | Dataset consolidation for fine-tuning |
| `src/db/historico.py` | User history & feedback persistence |

### Data Flow

```
User Input (pt/en)
    ↓
[Translation: pt → en via MarianMT]
    ↓
[Generation: 8 candidates via HF Inference API]
    ↓
[Scoring: compute EE(q) for each candidate]
    │
    ├─ Respondibilidade: BM25 corpus search
    ├─ Tratabilidade: local Ridge classifier
    └─ Não-trivialidade: semantic probe
    ↓
[Stage 1 Filter: keep only EE(q) > EE(q0) + ε]
    ↓
[Selection: return highest-scoring candidate]
    ↓
[Translation: en → pt (if needed)]
    ↓
[Persist: save to historico.db with feedback]
```

## 🚀 Deployment & Release

Contributors preparing releases should:

1. Update version in `pyproject.toml`
2. Update `CHANGELOG.md` (if exists)
3. Tag release: `git tag v1.0.0`
4. Push to GitHub: `git push origin --tags`

Maintainers will handle:
- GitHub release creation
- PyPI publication
- HF Spaces auto-deployment

## 📞 Questions?

- **GitHub Discussions** — Questions, ideas, announcements
- **GitHub Issues** — Bug reports, feature requests
- **Email** — https://github.com/fmr34 for non-technical inquiries

---

Thank you for contributing to ReformulatEE! 🙏
