# Contributing to PIR-Net

Thank you for your interest in contributing to this project.

## 1. Scope of Contributions

Contributions are welcome in the following areas:

1. Model implementation and training stability.
2. Reproducibility tooling and documentation.
3. Baseline and generalization benchmarking pipelines.
4. Validation GUI integration and usability.

## 2. Development Workflow

1. Create a feature branch from `main`.
2. Keep each pull request focused on a single technical objective.
3. Include clear validation evidence for behavioral changes.
4. Update related documentation in the same pull request.

## 3. Local Validation Before PR

Run the following checks:

```bash
python -m compileall -q .
python -m tools.smoke_test --mode all --exp_dir experiments/pirnet_ablation/222
```

If your contribution changes experiment execution paths, also run the affected experiment-level smoke training command(s).

## 4. Commit Message Convention

Use concise and professional Conventional Commit-style messages:

1. `feat: add zenodo cross-condition launcher`
2. `fix: resolve config utf-8 parsing issue`
3. `docs: standardize experiment execution guide`
4. `chore: update baseline script defaults`

## 5. Engineering Standards

1. Preserve existing experiment directory structure.
2. Do not inject PIR-specific preprocessing into external baselines.
3. Avoid committing generated runtime artifacts (`logs/`, `checkpoints/`, `results/`).
4. Keep public-facing text precise, technical, and reproducible.

## 6. Pull Request Checklist

- [ ] Code compiles and smoke checks pass locally.
- [ ] Documentation is updated for any user-visible behavior change.
- [ ] New scripts include usage instructions.
- [ ] No private data or credentials are included.
