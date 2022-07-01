# doren-bnn

DOReN-inspired Binary Neural Network architecture.

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

To set up the [CONCRETE library](https://www.zama.ai/concrete-framework), please follow
the instructions at [https://docs.zama.ai/concrete/lib/user/installation.html]. Take
note that the library is only supported on macOS and Linux.

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
