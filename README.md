# doren-bnn
DOReN-inspired Binary Neural Network architecture.

## Development Setup

### Virtual Environment Setup

In the root repository, create a new [virtual environment](https://docs.python.org/3/library/venv.html) in `.venv`:

```bash
python3 -m venv .venv
```

The virtual environment can now be activated using the following command.

```bash
source .venv/bin/activate
```

### Package Building

First install `flit`:

```bash
python3 -m pip install flit
```

The `doren-bnn` package can now be built and installed locally with the following commands:

```bash
flit install
```

### `pre-commit` setup

To enable the use of Git pre-commit hooks, install the [`pre-commit`](https://pre-commit.com/) package with the following command:

```bash
pre-commit install
```
