# diskannpy

[![DiskANN Paper](https://img.shields.io/badge/Paper-NeurIPS%3A_DiskANN-blue)](https://papers.nips.cc/paper/9527-rand-nsg-fast-accurate-billion-point-nearest-neighbor-search-on-a-single-node.pdf)
[![DiskANN Paper](https://img.shields.io/badge/Paper-Arxiv%3A_Fresh--DiskANN-blue)](https://arxiv.org/abs/2105.09613)
[![DiskANN Paper](https://img.shields.io/badge/Paper-Filtered--DiskANN-blue)](https://harsha-simhadri.org/pubs/Filtered-DiskANN23.pdf)
[![DiskANN Main](https://github.com/microsoft/DiskANN/actions/workflows/push-test.yml/badge.svg?branch=main)](https://github.com/microsoft/DiskANN/actions/workflows/push-test.yml)
[![PyPI version](https://img.shields.io/pypi/v/diskannpy.svg)](https://pypi.org/project/diskannpy/)
[![Downloads shield](https://pepy.tech/badge/diskannpy)](https://pepy.tech/project/diskannpy)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Installation
Packages published to PyPI will always be built using the latest numpy major.minor release (at this time, 1.25).

Conda distributions for versions 1.19-1.25 will be completed as a future effort.  In the meantime, feel free to
clone this repository and build it yourself.

## Local Build Instructions
Please see the [Project README](https://github.com/microsoft/DiskANN/blob/main/README.md) for system dependencies and requirements.

After ensuring you've followed the directions to build the project library and executables, you will be ready to also
build `diskannpy` with these additional instructions.

### Changing Numpy Version
In the root folder of DiskANN, there is a file `pyproject.toml`. You will need to edit the version of numpy in both the
`[build-system.requires]` section, as well as the `[project.dependencies]` section.  The version numbers must match.

#### Linux
```bash
python3.11 -m venv venv # versions from python3.9 and up should work
source venv/bin/activate
pip install build
python -m build
```

#### Windows
```powershell
py -3.11 -m venv venv # versions from python3.9 and up should work
venv\Scripts\Activate.ps1
pip install build
python -m build
```

The built wheel will be placed in the `dist` directory in your DiskANN root. Install it using `pip install dist/<wheel name>.whl`

## Citations
Please cite this software in your work as:
```
@misc{diskann-github,
   author = {Simhadri, Harsha Vardhan and Krishnaswamy, Ravishankar and Srinivasa, Gopal and Subramanya, Suhas Jayaram and Antonijevic, Andrija and Pryce, Dax and Kaczynski, David and Williams, Shane and Gollapudi, Siddarth and Sivashankar, Varun and Karia, Neel and Singh, Aditi and Jaiswal, Shikhar and Mahapatro, Neelam and Adams, Philip and Tower, Bryan and Patel, Yash}},
   title = {{DiskANN: Graph-structured Indices for Scalable, Fast, Fresh and Filtered Approximate Nearest Neighbor Search}},
   url = {https://github.com/Microsoft/DiskANN},
   version = {0.6.1},
   year = {2023}
}
```
