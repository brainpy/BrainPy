# Contributing to BrainPy

Thank you for your interest in contributing to BrainPy! We welcome contributions from the community and are grateful for your support in making BrainPy better.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [How Can I Contribute?](#how-can-i-contribute)
- [Development Setup](#development-setup)
- [Contribution Workflow](#contribution-workflow)
- [Coding Guidelines](#coding-guidelines)
- [Testing](#testing)
- [Documentation](#documentation)
- [Community](#community)

## Code of Conduct

This project and everyone participating in it is governed by our [Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code. Please report unacceptable behavior to brainpy@foxmail.com.

## How Can I Contribute?

### Reporting Bugs

Before creating bug reports, please check existing issues to avoid duplicates. When creating a bug report, include:

- **Clear title and description**
- **Steps to reproduce** the problem
- **Expected vs actual behavior**
- **BrainPy version** and environment details (Python version, OS, JAX version)
- **Code samples** or test cases that demonstrate the issue
- **Error messages** and stack traces

Submit bug reports via [GitHub Issues](https://github.com/brainpy/BrainPy/issues/new).

### Suggesting Enhancements

Enhancement suggestions are tracked as GitHub issues. When creating an enhancement suggestion, include:

- **Clear use case** - explain the problem you're trying to solve
- **Proposed solution** - describe how you envision the feature working
- **Alternative approaches** you've considered
- **Impact** - who would benefit and how

### Contributing Code

We welcome code contributions! This includes:

- Bug fixes
- New features
- Performance improvements
- Documentation improvements
- Test coverage improvements

## Development Setup

### Prerequisites

- Python 3.10 or higher
- Git
- pip or conda

### Setting Up Your Environment

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/YOUR_USERNAME/BrainPy.git
   cd BrainPy
   ```

3. **Create a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

4. **Install development dependencies**:
   ```bash
   pip install -e ".[dev,cpu]"  # Use [dev,cuda12] for CUDA support
   ```

5. **Set up pre-commit hooks** (if available):
   ```bash
   pre-commit install
   ```

## Contribution Workflow

### 1. Create a Branch

Create a new branch for your work:
```bash
git checkout -b feature/your-feature-name
# or
git checkout -b fix/issue-number-description
```

Branch naming conventions:
- `feature/` - new features
- `fix/` - bug fixes
- `docs/` - documentation changes
- `refactor/` - code refactoring
- `test/` - test improvements

### 2. Make Your Changes

- Write clean, readable code
- Follow the coding guidelines below
- Add tests for new functionality
- Update documentation as needed
- Keep commits focused and atomic

### 3. Test Your Changes

Run the test suite:
```bash
pytest tests/
```

Run specific tests:
```bash
pytest tests/test_specific.py -v
```

### 4. Commit Your Changes

Write clear, descriptive commit messages:
```bash
git commit -m "Fix issue with delayed connection handling (#123)

- Add validation for delay parameters
- Update tests to cover edge cases
- Fix documentation example
"
```

### 5. Push and Create a Pull Request

```bash
git push origin feature/your-feature-name
```

Then create a pull request on GitHub with:
- **Clear title** describing the change
- **Description** explaining what and why
- **Reference to related issues** (e.g., "Fixes #123")
- **Test results** or screenshots if applicable

## Coding Guidelines

### Python Style

- Follow [PEP 8](https://pep8.org/) style guide
- Use 2 or 4 spaces for indentation (be consistent with surrounding code)
- Maximum line length: 100-120 characters
- Use descriptive variable and function names

### Code Structure

- Keep functions focused and single-purpose
- Use type hints where appropriate
- Add docstrings for public APIs (follow NumPy docstring format)
- Avoid unnecessary complexity

### Example Docstring

```python
def simulate_network(network, duration, dt=0.1):
    """Simulate a neural network for a specified duration.

    Parameters
    ----------
    network : Network
        The neural network to simulate
    duration : float
        Total simulation time in milliseconds
    dt : float, optional
        Time step in milliseconds (default: 0.1)

    Returns
    -------
    results : dict
        Simulation results containing spike times and state variables

    Examples
    --------
    >>> net = Network()
    >>> results = simulate_network(net, duration=1000.0)
    """
    pass
```

## Testing

### Writing Tests

- Place tests in the `tests/` directory
- Use descriptive test function names: `test_feature_does_what_expected`
- Test edge cases and error conditions
- Use fixtures for common setups

### Test Coverage

Aim for high test coverage on new code:
```bash
pytest --cov=brainpy tests/
```

## Documentation

### Updating Documentation

- Update relevant documentation when changing features
- Add examples for new functionality
- Keep the API reference up to date
- Fix typos and improve clarity

For comprehensive contribution guidelines, see our [detailed documentation](https://brainpy.readthedocs.io/en/latest/tutorial_advanced/contributing.html).

## Community

### Getting Help

- **Documentation**: https://brainpy.readthedocs.io/
- **GitHub Discussions**: https://github.com/brainpy/BrainPy/discussions
- **Issues**: https://github.com/brainpy/BrainPy/issues

### Recognition

Contributors are recognized in:
- Release notes
- Contributors file
- Project documentation

Thank you for contributing to BrainPy! Your efforts help make computational neuroscience more accessible and powerful for everyone.
