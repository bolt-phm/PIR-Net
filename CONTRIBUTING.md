# Contributing

Thank you for your interest in improving PIR-Net.

## Development Workflow

1. Fork the repository and create a feature branch from `main`.
2. Keep changes scoped to a single purpose (code, docs, or tooling).
3. Run local checks before opening a pull request:

```bash
python -m compileall -q .
python -m tools.smoke_test --mode all --exp_dir experiments/pirnet_ablation/222
```

4. Open a pull request with a clear summary, rationale, and validation evidence.

## Commit Message Convention

Use concise, professional commit messages, for example:

- `feat: add baseline model configuration`
- `fix: correct config parsing for utf-8-sig`
- `docs: update reproducibility notes`
- `chore: auxiliary GUI`

## Coding Standards

- Use Python 3.10+ compatible syntax.
- Preserve existing experiment directory structure.
- Avoid introducing PIR-specific preprocessing into external baseline pipelines.
- Keep public documentation synchronized with code changes.

## Pull Request Checklist

- [ ] Code builds and smoke checks pass locally.
- [ ] Documentation is updated when behavior changes.
- [ ] New scripts include minimal usage instructions.
- [ ] No dataset binaries are committed outside approved LFS artifacts.