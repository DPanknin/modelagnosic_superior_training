# modelagnostic_superior_training

---
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

modelagnostic_superior_training implements a model-agnostic active learning approach for regression. In particular, it implements sampling from the model-agnostic superior training density, which is calculated via the Spatially Adaptive Bandwidth Estimation in Regression (SABER) approach. SABER is a sparse Mixture of Gaussian processes model with implementation is GPyTorch and PyTorch.

## Examples, Tutorials, and Documentation

See our [**examples**](https://github.com/DPanknin/modelagnostic_superior_training/blob/main/examples) on how to apply the active learning framework on regression problems.

## Installation

**Requirements**:
- Python >= 3.9
- PyTorch >= 1.13.1

Install modelagnostic_superior_training using pip or conda:

```bash
pip install git+https://github.com/DPanknin/modelagnosic_superior_training.git
```


## License
modelagnostic_superior_training is [MIT licensed](https://github.com/DPanknin/modelagnostic_superior_training/blob/main/LICENSE).
