# Third-party sources and notices

The initial distribution contains benchmark code and documentation, with no third-party source code, paper PDFs, datasets, or model weights vendored. The repository's [MIT license](LICENSE) applies to its original material. It does not relicense the referenced papers, upstream repositories, optional dependencies, or data obtained separately.

The DGP implementations express mathematical mechanisms in the benchmark's data contract. They are translations, with the changes recorded in [the paper cards](docs/PAPER_DGPS.md); they are not distributions of the authors' experiment scripts. Citation and permission to redistribute source code are separate matters. In particular, a publicly readable repository with no declared code license is a reference, not a reusable code dependency.

| Public reference | Pinned source | License evidence at that reference | Use here |
| --- | --- | --- | --- |
| CorNet | [`38b9c37`](https://github.com/tobhatt/CorNet/tree/38b9c37d5a86bc87950945d6ddb1ceb5354a45ba) | No license file found | Mathematical mechanism and architecture reference |
| Experimental Grounding | [`a25cf3d`](https://github.com/CausalML/RemovingHiddenConfounding/tree/a25cf3d2524a01c8a8d821f25cdeeb312c34a9cd) | No license file found | Mathematical mechanism reference |
| Integrative R-learner | [`a46ce2c`](https://github.com/lwu9/intRlearner/tree/a46ce2c1cc814865898b863d6a8e130bbc2c802b) | No license file found; `DESCRIPTION` has a placeholder license field | Mathematical mechanism reference |
| Elastic Integrative | [`a8c9a06`](https://github.com/Gaochenyin/ElasticIntegrative/tree/a8c9a06371485c25caca5cc8396ea5ba47f381b6) | [MIT, `LICENSE.md`](https://github.com/Gaochenyin/ElasticIntegrative/blob/a8c9a06371485c25caca5cc8396ea5ba47f381b6/LICENSE.md), copyright 2022 elasticHTE authors | Mathematical mechanism reference; source not vendored |
| Power-likelihood | [Publisher supplement](https://academic.oup.com/biometrics/article/81/1/ujaf008/8016472), archive SHA-256 in the lock | No code license file found in the inspected supplement | Mathematical mechanism reference |
| CVCI | [`d74cb7c`](https://github.com/xyang23/cross_validated_causal/tree/d74cb7c928938fd1a2978481f9aab6ac654c7d08) | No license file found | Mathematical mechanism reference |
| DataFusionPPI, optional | [`04c7f34`](https://github.com/CausalDataScience/DataFusionPPI/tree/04c7f3447b02fa09f875e68e8c47100a7207d546) | [MIT, `LICENSE`](https://github.com/CausalDataScience/DataFusionPPI/blob/04c7f3447b02fa09f875e68e8c47100a7207d546/LICENSE) | Separately installed official method, called through an adapter |

[sources.lock.toml](sources.lock.toml) records full revisions, selected file hashes, primary paper links, and source use. The six paper sources support eight translations and one explicit target-population extension; DataFusionPPI is an optional method reference and does not add a DGP source. The lock records inspected references rather than asserting that these are the latest upstream versions.

Python runtime dependencies are installed separately under their own licenses. Installing the optional DataFusionPPI dependency obtains its source from its public upstream. Its license notice must accompany any later redistribution of that package or substantial portions of its source. No permission to bundle an upstream paper or dataset follows from this inventory.

Any contribution that copies third-party code or bundles data must add its origin, exact version, applicable license, required notices, and distribution permission before inclusion. If permission cannot be established, retain a citation and write an equation translation or provide an external installation step.
