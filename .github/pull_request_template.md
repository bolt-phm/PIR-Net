## Summary

Describe the change in 3-5 concise bullet points.

## Validation

- [ ] `python -m compileall -q .`
- [ ] `python -m tools.smoke_test --mode all --exp_dir experiments/pirnet_ablation/222`
- [ ] Relevant training/evaluation command executed (if applicable)

## Reproducibility Impact

State whether this PR changes data processing, split strategy, model behavior, or baseline fairness constraints.

## Checklist

- [ ] Documentation updated (README/USAGE_GUIDE/notes)
- [ ] No accidental dataset or checkpoint artifacts committed
- [ ] Commit messages are concise and professional