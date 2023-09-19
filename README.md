[![Multi-Modality](agorabanner.png)](https://discord.gg/qUtxnK2NMf)

# Nerf
My personal implementation of the NERF paper, with much better code. Because the original implementation has ugly code and i don't understand absolutely anything there.


[Paper Link](https://arxiv.org/abs/2003.08934)

# Appreciation
* Lucidrains
* Agorians


# Install
`pip install nerf-torch`

# Usage
```python
import torch
from nerf.model import Nerf

model = Nerf()

x = torch.randn(1, 6)

output = model(x)

print(output)
```


# License
MIT

# Citations

```bibtex
@misc{2003.08934,
Author = {Ben Mildenhall and Pratul P. Srinivasan and Matthew Tancik and Jonathan T. Barron and Ravi Ramamoorthi and Ren Ng},
Title = {NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis},
Year = {2020},
Eprint = {arXiv:2003.08934},
}
```