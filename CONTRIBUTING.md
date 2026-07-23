# Contributing to HowdyAI

Thank you for considering contributing to HowdyAI! The following guidelines help keep contributions consistent and the review process smooth.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [How Can I Contribute?](#how-can-i-contribute)
- [Development Setup](#development-setup)
- [Pull Request Process](#pull-request-process)
- [Style Guidelines](#style-guidelines)

## Code of Conduct

This project and everyone participating in it is governed by the [Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code.

## How Can I Contribute?

### Reporting Bugs

Before submitting a bug report, please check the existing issues to avoid duplicates. When filing a bug, use the [bug report template](.github/ISSUE_TEMPLATE/bug_report.md) and include:

- A clear, descriptive title.
- Steps to reproduce the behavior.
- The expected vs. actual behavior.
- Your OS, Python version, and any relevant environment details.

### Suggesting Enhancements

Feature requests are welcome. Use the [feature request template](.github/ISSUE_TEMPLATE/feature_request.md) and describe:

- The problem you are trying to solve.
- Your proposed solution and any alternatives you considered.

### Improving Documentation

Documentation improvements — including fixing typos, clarifying explanations, and improving examples — are always welcome and do not require an issue first.

## Development Setup

```bash
git clone <repository-url>
cd HowdyAI
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install pre-commit
pre-commit install
```

Copy `.env.example` to `.env` and fill in your API keys before running any tests.

## Pull Request Process

1. Fork the repository and create your branch from `main`.
2. Write or update tests for any code you change. Coverage must remain at or above 70%.
3. Ensure all pre-commit hooks pass: `pre-commit run --all-files`.
4. Ensure the full test suite passes: `pytest tests/ --cov=src --cov=main`.
5. Fill in the [pull request template](.github/PULL_REQUEST_TEMPLATE.md) when opening your PR.
6. PRs are merged after at least one approving review.

## Style Guidelines

- **Python**: Code is linted with `ruff` and formatted with `autopep8`. Both are enforced by pre-commit hooks.
- **Commit messages**: Use the imperative present tense ("Add feature" not "Added feature").
- **Tests**: All new logic should be accompanied by a corresponding test module in `tests/`.
