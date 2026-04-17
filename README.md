# `PyRoFFI`: A Python Library for Robot Kinematics Using Spherical Approximations


[![Format Check](https://github.com/CoMMALab/pyroffi/actions/workflows/formatting.yml/badge.svg)](https://github.com/CoMMALab/pyroffi/actions/workflows/formatting.yml)
[![Pyright](https://github.com/CoMMALab/pyroffi/actions/workflows/pyright.yml/badge.svg)](https://github.com/CoMMALab/pyroffi/actions/workflows/pyright.yml)
[![Pytest](https://github.com/CoMMALab/pyroffi/actions/workflows/pytest.yml/badge.svg)](https://github.com/CoMMALab/pyroffi/actions/workflows/pytest.yml)
[![PyPI - Version](https://img.shields.io/pypi/v/pyroffi)](https://pypi.org/project/pyroffi/)

By Weihang Guo, Sai Coumar

PyRoFFI is a toolkit optimized with Jax JIT tracing for accelerated kinematics research with NVIDIA GPUs. This repository is expands on [pyroki](https://github.com/chungmin99/pyroki) and is backward-compatible. PyRoFFI replaces core kinematics kernels in PyRoKi with custom CUDA implementations through the Jax FFI interface to retain the benefits of Jax's compiler optimizations while adding the flexibility of CUDA for low-level GPU optimization. Additionally, we add an extended suite of optimization solvers for inverse kinematics and trajectory optimization. We provide accelerated kernels for the following kinematics functions:

- Forward Kinematics
- Inverse Kinematics: HJCD-IK, LS, SQP, MPPI, IKFlow
- Trajectory Optimization: SCO, CHOMP, L-BFGS, LS


Additional Features:
- SRDF parsing
- Runtime Neural SDFs
- VAMP Compatible Primitive Sets
- Improved Jax performance

## Installation
```
git clone https://github.com/commalab/pyroffi.git
cd pyroffi
pip install -e .
```
