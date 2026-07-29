# Verification

`python -m forecast_select check-project` checks:

- the Uptrend Selector artifact exists;
- it reproduces the registered 926/1500 result;
- the historical locked evaluation remains preserved;
- the active model does not claim to use the locked evaluation.

This is artifact-integrity verification only. There is no live
production-performance claim.

## Frozen evaluation exception

`artifacts/audit/locked_evaluation.parquet` was renamed for readability but
its bytes were not rewritten. Its internal historical labels still contain
the original version identifiers. Keeping those labels preserves the frozen
file's SHA-256 hash and makes the audit evidence independently verifiable;
they are not active model names.
