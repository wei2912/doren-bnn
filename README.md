# doren-bnn

PyTorch framework for building, testing and benchmarking Binarised Neural Networks (BNNs), which can be converted into Fully Homomorphic Encryption (FHE) versions to be executed on [CONCRETE](https://www.zama.ai/concrete-framework).

**Warning**: Much of the code pertaining to FHE is not functional, due to issues with [bootstrapping in CONCRETE](https://docs.zama.ai/concrete/v/0.1/advanced-operations/bootstrapping-a-ciphertext). Future versions of CONCRETE may address this problem, but would require significant modifications to the code.

## Development Setup

### Package Building

First [install Poetry](https://python-poetry.org/docs/#installation). Dependencies for
the `doren_bnn` package can now be installed in a virtual environment with the following
command:

```bash
poetry install
```

### `nbdime` Setup

To view Jupyter notebooks with [`nbdime`](https://github.com/jupyter/nbdime), configure
the diff/merge drivers for this repository with the following command:

```bash
nbdime config-git --enable
```

### `pre-commit` Setup

To enable the use of Git pre-commit hooks, install the
[`pre-commit`](https://pre-commit.com/) package with the following command:

```bash
pre-commit install
```

### CONCRETE Setup

To set up the [Concrete library](https://www.zama.ai/concrete-framework), please follow
the instructions at [https://docs.zama.ai/concrete/lib/user/installation.html]. Take
note that the library is only supported on macOS and Linux.

**Note**: Currently, it seems that compilation only works on Nightly Rust and not
stable.

## Compiling Packages

### `doren_bnn_concrete`

To run the executable in the library, run the following commands from the root
directory:

```bash
cd doren_bnn_concrete
RUSTCFLAGS="target-cpu=native" cargo run --release
```

To build and install the Python bindings, run the following commands from the root
directory:

```bash
cd doren_bnn_concrete
RUSTCFLAGS="target-cpu=native" maturin develop --release
```

You can now import the package `doren_bnn_concrete` from the root directory.
