# EDA Toolkit

A comprehensive toolkit for Exploratory Data Analysis (EDA) with Python. This package provides a structured approach to data exploration, cleaning, visualization, and statistical analysis.

## Features

- 🔍 **Data Loading & Cleaning**: Robust data loading utilities with automated cleaning functions
- 📊 **Comprehensive Visualizations**: Pre-built plotting functions for all common chart types
- 🧮 **Statistical Analysis**: Built-in statistical functions for data insights
- 🔧 **Feature Engineering**: Tools for creating and transforming features
- 📓 **Jupyter Notebooks**: Ready-to-use notebook templates for different analysis stages
- 🧪 **Testing Suite**: Comprehensive tests to ensure reliability

## Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd eda

# Install dependencies
pip install -r requirements.txt

# Install the package in development mode
pip install -e .
```

## Quick Start

```python
from src.data_loader import DataLoader
from plots.histogram import create_histogram
from src.statistical_analysis import StatisticalAnalyzer

# Load data
loader = DataLoader()
data = loader.load_csv('data/raw/your_data.csv')

# Create visualizations
create_histogram(data, 'column_name', save_path='figures/exploratory/')

# Perform statistical analysis
analyzer = StatisticalAnalyzer(data)
summary = analyzer.describe_all()
```

## Project Structure

```
eda/
├── setup.py                    # Package installation configuration
├── requirements.txt            # Project dependencies
├── main.py                     # Main entry point
├── config/                     # Configuration settings
├── data/                       # Data storage (raw, processed, external, interim)
├── src/                        # Core EDA utilities
├── plots/                      # Visualization modules
├── figures/                    # Generated plots and visualizations
├── notebooks/                  # Jupyter notebook templates
├── tests/                      # Test suite
├── docs/                       # Documentation
└── examples/                   # Usage examples
```

## Usage Examples

See the `examples/` directory for detailed usage examples and the `notebooks/` directory for interactive analysis templates.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests: `pytest tests/`
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.
