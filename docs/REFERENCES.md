# Selected references

This bibliography identifies sources used in the project's design or implementation. Each section explains the source's role, from architectural motivation to algorithm and software use. Machine-readable entries are in [references.bib](references.bib).

## Reader design

The common-reader design recorded on 6 October 2026 explicitly used the following references as architectural motivation:

- **Graph Metanetworks for Processing Diverse Neural Architectures** (Lim et al., ICLR 2024): graph representations and neural-network parameter symmetries. [Source](https://openreview.net/forum?id=ijK5hyxs0n)
- **Neural Functional Transformers** (Zhou et al., 2023): attention-based equivariant weight-space layers. [Source](https://arxiv.org/abs/2305.13546)
- **Set Transformer** (Lee et al., ICML 2019): attention over unordered records. [Source](https://proceedings.mlr.press/v97/lee19d.html)
- **Perceiver IO** (Jaegle et al., 2022): separating structured input encoding from queried output decoding. [Source](https://arxiv.org/abs/2107.14795)

These sources motivate the target-record representation and paired attention reader. Synaptic Lens's empirical findings and evaluation scope are documented in [Selected research results](RESULTS.md).

## Algorithms and software

- **Adam: A Method for Stochastic Optimization** (Kingma and Ba, ICLR 2015). [Source](https://arxiv.org/abs/1412.6980)
- **Decoupled Weight Decay Regularization** (Loshchilov and Hutter, ICLR 2019). [Source](https://arxiv.org/abs/1711.05101)
- **PyTorch documentation**. [Source](https://docs.pytorch.org/docs/stable/index.html)

The project uses PyTorch, including Adam-family optimization. The research training implementation constructs AdamW and saves/restores optimizer state. These citations credit the algorithms and software used. The exact first-adoption date and historical documentation versions are not established. Historical implementation milestones need not contain every component of the current project.

Speculative proposals, news briefings, unadopted alternatives and operational migration references are deliberately excluded. This selected bibliography documents the project's design and implementation sources. Paper citations do not replace applicable software licenses or third-party notices.
