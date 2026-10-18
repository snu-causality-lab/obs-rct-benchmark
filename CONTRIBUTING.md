# Contributing

Keep the benchmark usable without any unpublished method or external research workspace.

A DGP contribution needs a primary source or full generating equations, an explicit target population and CATE, the parameters that differ from the source, and checks of observed sampling against the stated truth. A code translation and an exact paper reproduction are distinct claims. Document the source's version and redistribution terms in `sources.lock.toml` and `THIRD_PARTY_NOTICES.md`.

A method receives only the observed training view. Use fixed declared settings or permitted training/tuning data to choose parameters. Evaluation truth, latent confounders and evaluation outcomes belong to the evaluator. Python object separation prevents accidental leakage; it is not a security sandbox.

Result contributions need the executed configuration, source version, package versions, expected and completed IDs, failures and an explicit metric definition. Preserve paired replicate IDs. Report missingness before ranking methods. Keep datasets, model checkpoints and large output arrays outside Git.

Before submitting a change, run `python -m pytest` and `python scripts/check_release.py`. Optional upstream integrations have separate dependency requirements described in the README. Test the smallest relevant path first; schedule substantial experiments on appropriate compute resources.
