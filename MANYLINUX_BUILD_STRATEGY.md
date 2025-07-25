# Manylinux Build Strategy

## Problem
Google Colab requires wheels compatible with `manylinux_2_35_x86_64` or earlier. Our previous builds were producing `manylinux_2_39_x86_64` wheels, which are incompatible.

## Solution
We're using `cibuildwheel` with `manylinux2014` images to build wheels that are compatible with a wide range of Linux distributions, including Google Colab.

### Key Changes

1. **cibuildwheel Configuration**
   - Using `manylinux2014` images (provides `manylinux_2_17` compatibility)
   - Using `yum` package manager (CentOS 7 based)
   - Installing `cmake3` and creating symlink for compatibility

2. **Build Matrix**
   - Python versions: 3.9, 3.10, 3.11, 3.12, 3.13
   - Platforms: Linux (x86_64), macOS
   - No Windows support (not required)

3. **Dependencies**
   - Linux: gcc-c++, boost-devel, zeromq-devel, openblas-devel, cmake3
   - macOS: boost, zeromq, openblas, cmake (via Homebrew)

4. **Environment Variables**
   - `CMAKE_BUILD_PARALLEL_LEVEL=8`: Speed up builds
   - `Python_FIND_VIRTUALENV=ONLY`: Help CMake find Python in cibuildwheel env
   - `Python3_FIND_VIRTUALENV=ONLY`: Alternative variable for compatibility

## Testing Strategy

1. **CI Pipeline**: `test-manylinux.yml`
   - Triggers on PR to main, manual dispatch, or push to `fix/manylinux-*` branches
   - Builds wheels using cibuildwheel
   - Tests installation on Ubuntu 22.04 (simulating Colab)

2. **Local Testing**
   ```bash
   # Download built wheels
   # Test in fresh environment
   python -m venv test_env
   source test_env/bin/activate
   pip install leann_core-*.whl leann_backend_hnsw-*manylinux*.whl leann-*.whl
   python -c "from leann import LeannBuilder; print('Success!')"
   ```

## References
- [cibuildwheel documentation](https://cibuildwheel.readthedocs.io/)
- [manylinux standards](https://github.com/pypa/manylinux)
- [PEP 599 - manylinux2014](https://peps.python.org/pep-0599/) 