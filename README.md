# Compgrapher

[![CI](https://github.com/nstapc/compgrapher/actions/workflows/ci.yml/badge.svg)](https://github.com/nstapc/compgrapher/actions/workflows/ci.yml)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Generates floating bar graphs from compensation market data to facilitate employer comparisons.

## Example Output

<img src="output/svg/Example%20Software%20Engineer.svg" width="60%" alt="Example Software Engineer Compensation">

<img src="output/svg/Example%20Data%20Scientist.svg" width="60%" alt="Example Data Scientist Compensation">


## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/nstapc/compgrapher.git
cd compgrapher

# Set up virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Basic Usage

```bash
# Generate PNG graphs from sample data
python main.py -i input/csv/example_table.csv --output png

# Generate HTML report with charts
python main.py -i your_data.csv --output html png

# Specify the client to highlight
python main.py -c "Your Company" -i data.csv --output html pdf png
```

## Documentation

### Input Data Format

The data file (CSV, XLS, XLSX, or ODS) should have a 'POSITION TITLE' column and columns for each employer with salary data. The data alternates between high and low salary ranges on consecutive rows for each position.

Example structure:

| Row | POSITION TITLE | Employer A | Employer B | Employer C |
|-----|----------------|------------|------------|------------|
| 1   | Software Engineer | 100 | 95 | 105 |
| 2   |                | 80 | 75 | 85 |
| 3   | Data Scientist | 120 | 110 | 115 |
| 4   |                | 90 | 85 | 95 |

- Odd rows: High salary values
- Even rows: Low salary values

### Command Line Options

```
python main.py [OPTIONS]

Options:
  --client NAME       Name of the employer to highlight (default: first in data)
  -i FILE             Path to data file (.csv, .xls, .xlsx, .ods)
  --output FORMAT(s)  Output format(s): html, pdf, png, svg, jpg, jpeg, webp, eps
  --validate          Run data validation checks
  --config FILE       Path to YAML configuration file
  -v, --verbose       Enable verbose output
  -q, --quiet         Suppress output except errors
  -V, --version       Show version information
```

### Output Formats

| Format | Description |
|--------|-------------|
| `html` | Interactive HTML report with embedded charts and statistics |
| `png`  | High-quality raster images |
| `svg`  | Scalable vector graphics (ideal for documents) |
| `pdf`  | Print-ready PDF files |
| `jpg`/`jpeg` | JPEG images |
| `webp` | Modern web-optimized format |
| `eps`  | Encapsulated PostScript (for publishing) |

### Configuration File

Create a `config.yaml` file to set default options:

```yaml
defaults:
  input: input/csv/my_data.csv
  output:
    - png
    - html
  validate: true

graph:
  colors:
    client: '#4CAF50'
    default: '#FFFFFF'
  display:
    show_grid: true
    show_labels: false
```

Use with: `python main.py --config config.yaml`

## Testing

```bash
# Install test dependencies
pip install pytest pytest-cov

# Run tests
pytest

# Run with coverage report
pytest --cov=. --cov-report=html
```

## Project Structure

```
compgrapher/
├── main.py              # Main entry point
├── cli.py               # Enhanced command-line interface
├── data_parser.py       # Data loading and validation
├── graph_generator.py   # Graph generation with statistics
├── config.yaml          # Default configuration
├── pyproject.toml       # Modern Python packaging
├── requirements.txt     # Dependencies
├── tests/               # Unit tests
│   ├── __init__.py
│   └── test_data_parser.py
├── .github/
│   └── workflows/
│       └── ci.yml       # CI/CD pipeline
├── input/               # Input data files
│   ├── csv/
│   ├── ods/
│   └── xls/
└── output/              # Generated outputs
    ├── html/
    ├── png/
    ├── svg/
    └── pdf/
```

## Development

### Setting Up Development Environment

```bash
# Install with development dependencies
pip install -e .[dev]

# Install pre-commit hooks
pre-commit install

# Run linting
black .
isort .
flake8 .
```

### Running Tests

```bash
# Run all tests
pytest

# Run with verbose output
pytest -v

# Run specific test file
pytest tests/test_data_parser.py
```