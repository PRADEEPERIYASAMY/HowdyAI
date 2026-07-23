# Support

## Getting Help

If you have a question or run into trouble with HowdyAI, try the following resources in order:

### 1. Check the Documentation

The [README](README.md) covers installation, configuration, usage, and the full system architecture. Read through it first — most common questions are answered there.

### 2. Check Existing Issues

Search the [GitHub Issues](../../issues) to see if your question or problem has already been raised or resolved.

### 3. Open a New Issue

If you cannot find an answer, open a new issue using the appropriate template:

- **Bug:** Use the [bug report template](.github/ISSUE_TEMPLATE/bug_report.md) and include your OS, Python version, and steps to reproduce.
- **Feature request:** Use the [feature request template](.github/ISSUE_TEMPLATE/feature_request.md).
- **General question:** Open a blank issue with a clear description of what you need help with.

## Health Check

Before filing a bug related to API connectivity or retrieval failures, run the built-in health check to validate all system dependencies:

```bash
python eval/health_check.py
```

This validates that your OpenAI and Brave API keys are correctly configured and that ChromaDB is accessible.

## Response Time

This is a personal project maintained on a best-effort basis. Issues will be reviewed as time permits.
