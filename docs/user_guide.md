# EDA Toolkit User Guide

This guide provides comprehensive instructions for using the EDA (Exploratory Data Analysis) toolkit effectively.

## Table of Contents

1. [Getting Started](#getting-started)
2. [Basic Workflow](#basic-workflow)
3. [Data Loading](#data-loading)
4. [Data Cleaning](#data-cleaning)
5. [Statistical Analysis](#statistical-analysis)
6. [Feature Engineering](#feature-engineering)
7. [Visualizations](#visualizations)
8. [Advanced Usage](#advanced-usage)
9. [Best Practices](#best-practices)
10. [Troubleshooting](#troubleshooting)

## Getting Started

### Installation

1. Clone or download the EDA toolkit to your project directory
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Install the package in development mode:
   ```bash
   pip install -e .
   ```

### Directory Structure

```
eda/
├── src/                    # Core modules
├── plots/                  # Visualization modules
├── notebooks/              # Jupyter notebook templates
├── data/                   # Data storage
├── figures/                # Generated plots
├── examples/               # Usage examples
└── docs/                   # Documentation
```

### Quick Start

```python
from src.data_loader import DataLoader
from src.statistical_analysis import StatisticalAnalyzer
from plots.histogram import create_histogram

# Load data
loader = DataLoader()
data = loader.load_csv('your_data.csv')

# Analyze
analyzer = StatisticalAnalyzer(data)
stats = analyzer.describe_all()

# Visualize
create_histogram(data, 'column_name', save_path='figures/histogram.png')
```

## Basic Workflow

### 1. Data Loading and Exploration

Start by loading your data and getting a basic understanding:

```python
from src.data_loader import DataLoader
from src.statistical_analysis import StatisticalAnalyzer

# Load data
loader = DataLoader()
data = loader.load_csv('data/raw/your_file.csv')

# Basic info
print(f"Shape: {data.shape}")
print(data.info())
print(data.head())

# Statistical overview
analyzer = StatisticalAnalyzer(data)
stats = analyzer.describe_all()
print(stats['numeric_summary'])
```

### 2. Data Quality Assessment

Check for data quality issues:

```python
from src.data_cleaner import DataCleaner

cleaner = DataCleaner()

# Missing values
missing_info = cleaner.detect_missing_patterns(data)
print(missing_info['by_column'])

# Duplicates
duplicate_info = cleaner.detect_duplicates(data)
print(f"Duplicates: {duplicate_info['count']}")

# Outliers
outlier_info = cleaner.detect_outliers(data)
for col, info in outlier_info.items():
    if info['count'] > 0:
        print(f"{col}: {info['count']} outliers ({info['percentage']:.1f}%)")
```

### 3. Data Cleaning

Clean the data based on your findings:

```python
# Define cleaning strategy
cleaning_config = {
    'handle_missing': True,
    'missing_strategy': {
        'numerical_col': 'median',
        'categorical_col': 'mode'
    },
    'remove_duplicates': True,
    'handle_outliers': True,
    'outlier_strategy': 'cap',
    'standardize_columns': True
}

# Apply cleaning pipeline
data_clean = cleaner.clean_pipeline(data, cleaning_config)
```

### 4. Exploratory Visualization

Create visualizations to understand your data:

```python
from plots.histogram import create_histogram, create_multiple_histograms
from plots.box_plot import create_box_plot
from plots.correlation_matrix import create_correlation_heatmap
from plots.scatter_plot import create_scatter_plot

# Distribution analysis
numeric_cols = data_clean.select_dtypes(include=['number']).columns.tolist()
create_multiple_histograms(data_clean, numeric_cols, 
                          save_path='figures/distributions.png')

# Outlier analysis
create_box_plot(data_clean, numeric_cols, 
               save_path='figures/boxplots.png')

# Correlation analysis
create_correlation_heatmap(data_clean, numeric_cols,
                          save_path='figures/correlations.png')

# Relationships
create_scatter_plot(data_clean, 'x_column', 'y_column', 
                   hue='category_column',
                   save_path='figures/scatter.png')
```

## Data Loading

### Supported Formats

The DataLoader supports various file formats:

```python
loader = DataLoader()

# CSV files
data = loader.load_csv('file.csv')
data = loader.load_csv('file.csv', encoding='latin-1', nrows=1000)

# Excel files
data = loader.load_excel('file.xlsx')
data = loader.load_excel('file.xlsx', sheet_name='Sheet2')

# JSON files
data = loader.load_json('file.json')

# Parquet files
data = loader.load_parquet('file.parquet')

# Auto-detection
data = loader.auto_detect_and_load('file.csv')  # Detects format automatically
```

### Loading Multiple Files

```python
# Load and combine multiple files
data = loader.load_multiple_files('data/raw/*.csv', 'csv')
print(data['source_file'].value_counts())  # See file origins
```

### File Information

```python
# Get file info without loading
info = loader.get_file_info('large_file.csv')
print(f"Size: {info['size_mb']:.1f} MB")
print(f"Estimated rows: {info.get('estimated_rows', 'Unknown')}")
```

## Data Cleaning

### Missing Values

```python
cleaner = DataCleaner()

# Analyze missing patterns
missing_info = cleaner.detect_missing_patterns(data)

# Custom strategies for different columns
strategies = {
    'age': 'median',           # Use median for numeric
    'category': 'mode',        # Use mode for categorical
    'income': 'mean',          # Use mean
    'description': 'drop'      # Drop rows with missing values
}

data_clean = cleaner.handle_missing_values(data, strategy=strategies)
```

### Outlier Detection and Treatment

```python
# Detect outliers using different methods
outliers_iqr = cleaner.detect_outliers(data, method='iqr', threshold=1.5)
outliers_z = cleaner.detect_outliers(data, method='zscore', threshold=3)

# Handle outliers
data_clean = cleaner.handle_outliers(data, outliers_iqr, strategy='cap')
# Strategies: 'remove', 'cap', 'transform'
```

### Duplicate Handling

```python
# Check for duplicates
duplicate_info = cleaner.detect_duplicates(data)

# Remove duplicates
data_clean = cleaner.remove_duplicates(data, keep='first')

# Check specific columns only
duplicate_info = cleaner.detect_duplicates(data, subset=['id', 'name'])
```

## Statistical Analysis

### Descriptive Statistics

```python
analyzer = StatisticalAnalyzer(data)

# Comprehensive statistics
stats = analyzer.describe_all()

# Access different summaries
print(stats['numeric_summary'])      # Basic numeric stats
print(stats['numeric_extended'])     # Skewness, kurtosis, etc.
print(stats['categorical_summary'])  # Categorical variable stats
print(stats['missing_summary'])      # Missing value summary
```

### Correlation Analysis

```python
# Correlation matrix
corr_results = analyzer.correlation_analysis(method='pearson')
print(corr_results['correlation_matrix'])

# High correlations
high_corr = corr_results['high_correlations']
if not high_corr.empty:
    print("Highly correlated pairs:")
    print(high_corr)

# Different correlation methods
spearman_corr = analyzer.correlation_analysis(method='spearman')
kendall_corr = analyzer.correlation_analysis(method='kendall')
```

### Distribution Analysis

```python
# Test for normality
numeric_cols = data.select_dtypes(include=['number']).columns.tolist()
normality = analyzer.normality_tests(numeric_cols)
print(normality[['column', 'shapiro_normal', 'dagostino_normal']])

# Distribution properties
dist_analysis = analyzer.distribution_analysis(numeric_cols)
for col, props in dist_analysis.items():
    print(f"{col}: {props['shape']} distribution")
```

### Hypothesis Testing

```python
# Configure hypothesis tests
tests_config = [
    {
        'test': 't_test_1sample',
        'columns': ['age'],
        'pop_mean': 35
    },
    {
        'test': 't_test_2sample',
        'columns': ['group1_score', 'group2_score']
    },
    {
        'test': 'anova',
        'columns': ['score'],
        'groups': 'category'
    }
]

results = analyzer.hypothesis_tests(tests_config)
print(results)
```

## Feature Engineering

### Creating New Features

```python
from src.feature_engineer import FeatureEngineer

engineer = FeatureEngineer()

# DateTime features
data_featured = engineer.create_datetime_features(
    data, ['date_column']
)

# Binned features
binning_config = {
    'age': 5,  # 5 equal-width bins
    'income': [0, 30000, 60000, 100000, float('inf')]  # Custom bins
}
data_featured = engineer.create_binned_features(data_featured, binning_config)

# Interaction features
feature_pairs = [('age', 'experience'), ('education_years', 'salary')]
data_featured = engineer.create_interaction_features(
    data_featured, feature_pairs, operations=['multiply', 'add']
)

# Polynomial features
data_featured = engineer.create_polynomial_features(
    data_featured, ['age', 'experience'], degree=2
)
```

### Encoding Categorical Variables

```python
# One-hot encoding for low cardinality
data_encoded = engineer.encode_categorical_features(
    data, encoding_type='onehot', max_categories=10
)

# Label encoding for high cardinality
data_encoded = engineer.encode_categorical_features(
    data, columns=['high_cardinality_col'], encoding_type='label'
)
```

### Feature Scaling

```python
# Standard scaling
data_scaled = engineer.scale_features(data, scaler_type='standard')

# Min-max scaling
data_scaled = engineer.scale_features(data, scaler_type='minmax')

# Robust scaling (less sensitive to outliers)
data_scaled = engineer.scale_features(data, scaler_type='robust')
```

## Visualizations

### Individual Plot Functions

```python
from plots.histogram import create_histogram, create_grouped_histogram
from plots.box_plot import create_box_plot, create_violin_plot
from plots.scatter_plot import create_scatter_plot, create_bubble_plot
from plots.bar_plot import create_bar_plot, create_count_plot
from plots.line_plot import create_line_plot, create_time_series_plot

# Histograms
create_histogram(data, 'age', bins=30, save_path='figures/age_dist.png')
create_grouped_histogram(data, 'salary', 'department', 
                        save_path='figures/salary_by_dept.png')

# Box plots
create_box_plot(data, ['age', 'salary'], save_path='figures/boxplots.png')
create_violin_plot(data, 'salary', groupby='department')

# Scatter plots
create_scatter_plot(data, 'age', 'salary', hue='department',
                   add_trendline=True, save_path='figures/age_salary.png')

# Bar plots
create_count_plot(data, 'department', save_path='figures/dept_counts.png')
create_bar_plot(data, 'department', 'salary', save_path='figures/avg_salary.png')
```

### Customizing Plots

```python
from plots.utils import PlotConfig

# Configure global plot settings
config = PlotConfig()
config.set_style(style='whitegrid', palette='Set2', context='notebook')

# Custom plot parameters
create_histogram(data, 'age',
                title='Age Distribution of Employees',
                xlabel='Age (years)',
                ylabel='Frequency',
                bins=25,
                alpha=0.7,
                color='skyblue',
                figsize=(10, 6))
```

## Advanced Usage

### Custom Analysis Pipeline

```python
def custom_eda_pipeline(data_path, output_dir):
    """Custom EDA pipeline"""
    
    # 1. Load data
    loader = DataLoader()
    data = loader.auto_detect_and_load(data_path)
    
    # 2. Clean data
    cleaner = DataCleaner()
    config = {
        'handle_missing': True,
        'remove_duplicates': True,
        'handle_outliers': True
    }
    data_clean = cleaner.clean_pipeline(data, config)
    
    # 3. Analyze
    analyzer = StatisticalAnalyzer(data_clean)
    stats = analyzer.describe_all()
    
    # 4. Generate report
    analyzer.export_results(f'{output_dir}/analysis_report.xlsx')
    
    # 5. Create visualizations
    numeric_cols = data_clean.select_dtypes(include=['number']).columns.tolist()
    
    if len(numeric_cols) > 1:
        create_correlation_heatmap(data_clean, numeric_cols,
                                 save_path=f'{output_dir}/correlations.png')
    
    create_multiple_histograms(data_clean, numeric_cols,
                             save_path=f'{output_dir}/distributions.png')
    
    return data_clean, stats

# Usage
clean_data, analysis_stats = custom_eda_pipeline('data.csv', 'output/')
```

### Batch Processing

```python
import glob
from pathlib import Path

def process_multiple_datasets(data_pattern, output_dir):
    """Process multiple datasets"""
    
    loader = DataLoader()
    results = {}
    
    for file_path in glob.glob(data_pattern):
        file_name = Path(file_path).stem
        print(f"Processing {file_name}...")
        
        # Load and analyze
        data = loader.auto_detect_and_load(file_path)
        analyzer = StatisticalAnalyzer(data)
        stats = analyzer.describe_all()
        
        results[file_name] = stats
        
        # Save individual reports
        analyzer.export_results(f'{output_dir}/{file_name}_analysis.xlsx')
    
    return results

# Process all CSV files in a directory
results = process_multiple_datasets('data/*.csv', 'reports/')
```

## Best Practices

### 1. Data Loading

- Always check data shape and types after loading
- Use appropriate encodings for text files
- Consider memory usage for large files
- Validate data integrity after loading

### 2. Data Cleaning

- Document all cleaning decisions
- Keep track of data transformations
- Validate cleaning results
- Save cleaned data for reproducibility

### 3. Analysis

- Start with descriptive statistics
- Check assumptions before statistical tests
- Use appropriate correlation methods
- Consider domain knowledge in interpretation

### 4. Visualization

- Choose appropriate plot types for your data
- Use consistent styling across plots
- Add meaningful titles and labels
- Save plots in high resolution for reports

### 5. Reproducibility

- Use random seeds for consistent results
- Document parameter choices
- Version control your analysis code
- Save intermediate results

## Troubleshooting

### Common Issues

#### Import Errors
```python
# If you get import errors, add the src directory to Python path
import sys
sys.path.append('path/to/eda/src')
sys.path.append('path/to/eda/plots')
```

#### Memory Issues
```python
# For large files, use chunking
loader = DataLoader(low_memory=True)
data = loader.load_csv('large_file.csv', chunksize=10000)
```

#### Plotting Issues
```python
# If plots don't display, save them instead
create_histogram(data, 'column', save_path='plot.png')

# For Jupyter notebooks, use inline backend
%matplotlib inline
```

### Performance Tips

1. **Use appropriate data types**: Convert object columns to category when possible
2. **Sample large datasets**: Use data.sample(n=10000) for initial exploration
3. **Vectorize operations**: Use pandas/numpy operations instead of loops
4. **Cache results**: Save intermediate analysis results

### Getting Help

1. Check the API reference for detailed function documentation
2. Look at the examples directory for usage patterns
3. Review the test files to understand expected behavior
4. Check the notebook templates for complete workflows
