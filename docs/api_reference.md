# API Reference

This document provides detailed information about the EDA toolkit's classes and functions.

## Core Modules

### DataLoader

The `DataLoader` class provides utilities for loading various data formats.

#### Constructor

```python
DataLoader(encoding='utf-8', low_memory=False)
```

**Parameters:**
- `encoding` (str): Default encoding for text files
- `low_memory` (bool): Whether to use low memory mode for large files

#### Methods

##### load_csv(filepath, **kwargs)

Load CSV file with robust error handling.

**Parameters:**
- `filepath` (str|Path): Path to CSV file
- `**kwargs`: Additional arguments for pd.read_csv

**Returns:** pandas DataFrame

##### load_excel(filepath, sheet_name=None, **kwargs)

Load Excel file.

**Parameters:**
- `filepath` (str|Path): Path to Excel file
- `sheet_name` (str): Name of sheet to load (default: first sheet)
- `**kwargs`: Additional arguments for pd.read_excel

**Returns:** pandas DataFrame

##### auto_detect_and_load(filepath, **kwargs)

Automatically detect file type and load appropriately.

**Parameters:**
- `filepath` (str|Path): Path to file
- `**kwargs`: Additional arguments for loading functions

**Returns:** pandas DataFrame

---

### DataCleaner

The `DataCleaner` class provides comprehensive data cleaning operations.

#### Methods

##### detect_missing_patterns(df)

Analyze missing data patterns.

**Parameters:**
- `df` (DataFrame): Input DataFrame

**Returns:** Dictionary with missing data analysis

##### handle_missing_values(df, strategy=None, threshold=0.5)

Handle missing values based on specified strategies.

**Parameters:**
- `df` (DataFrame): Input DataFrame
- `strategy` (dict): Dictionary mapping column names to strategies
- `threshold` (float): Threshold for dropping columns

**Returns:** Cleaned DataFrame

##### detect_outliers(df, columns=None, method='iqr', threshold=1.5)

Detect outliers in numeric columns.

**Parameters:**
- `df` (DataFrame): Input DataFrame
- `columns` (list): Columns to check
- `method` (str): Method to use ('iqr', 'zscore', 'modified_zscore')
- `threshold` (float): Threshold for outlier detection

**Returns:** Dictionary with outlier information

---

### StatisticalAnalyzer

The `StatisticalAnalyzer` class provides statistical analysis operations.

#### Constructor

```python
StatisticalAnalyzer(df)
```

**Parameters:**
- `df` (DataFrame): Input DataFrame for analysis

#### Methods

##### describe_all()

Generate comprehensive descriptive statistics.

**Returns:** Dictionary containing different types of descriptive statistics

##### correlation_analysis(method='pearson', columns=None)

Perform correlation analysis.

**Parameters:**
- `method` (str): Correlation method ('pearson', 'spearman', 'kendall')
- `columns` (list): List of columns to analyze

**Returns:** Dictionary with correlation results

##### normality_tests(columns=None, alpha=0.05)

Test for normality using multiple tests.

**Parameters:**
- `columns` (list): List of columns to test
- `alpha` (float): Significance level

**Returns:** DataFrame with normality test results

---

### FeatureEngineer

The `FeatureEngineer` class provides feature engineering operations.

#### Methods

##### create_datetime_features(df, datetime_columns)

Create features from datetime columns.

**Parameters:**
- `df` (DataFrame): Input DataFrame
- `datetime_columns` (list): List of datetime column names

**Returns:** DataFrame with new datetime features

##### create_interaction_features(df, feature_pairs, operations=['multiply'])

Create interaction features between pairs of columns.

**Parameters:**
- `df` (DataFrame): Input DataFrame
- `feature_pairs` (list): List of tuples containing column pairs
- `operations` (list): List of operations

**Returns:** DataFrame with interaction features

##### encode_categorical_features(df, columns=None, encoding_type='onehot')

Encode categorical features.

**Parameters:**
- `df` (DataFrame): Input DataFrame
- `columns` (list): List of categorical columns
- `encoding_type` (str): Type of encoding ('onehot', 'label', 'target')

**Returns:** DataFrame with encoded features

---

## Plotting Modules

### Histogram Functions

##### create_histogram(data, column, bins='auto', **kwargs)

Create histogram for exploring data distributions.

**Parameters:**
- `data` (DataFrame): Input DataFrame
- `column` (str): Column name to plot
- `bins` (int|str|list): Number of bins, binning strategy, or bin edges
- `**kwargs`: Additional plotting arguments

**Returns:** Matplotlib figure object

##### create_multiple_histograms(data, columns, **kwargs)

Create multiple histograms in a grid layout.

**Parameters:**
- `data` (DataFrame): Input DataFrame
- `columns` (list): List of column names to plot
- `**kwargs`: Additional plotting arguments

**Returns:** Matplotlib figure object

---

### Box Plot Functions

##### create_box_plot(data, columns, groupby=None, **kwargs)

Create box plot for exploring distributions and outliers.

**Parameters:**
- `data` (DataFrame): Input DataFrame
- `columns` (str|list): Column name(s) to plot
- `groupby` (str): Column to group by
- `**kwargs`: Additional plotting arguments

**Returns:** Matplotlib figure object

---

### Scatter Plot Functions

##### create_scatter_plot(data, x, y, hue=None, **kwargs)

Create scatter plot for exploring relationships between variables.

**Parameters:**
- `data` (DataFrame): Input DataFrame
- `x` (str): Column name for x-axis
- `y` (str): Column name for y-axis
- `hue` (str): Column name for color coding
- `**kwargs`: Additional plotting arguments

**Returns:** Matplotlib figure object

---

### Correlation Matrix Functions

##### create_correlation_heatmap(data, columns=None, method='pearson', **kwargs)

Create correlation heatmap for exploring relationships between variables.

**Parameters:**
- `data` (DataFrame): Input DataFrame
- `columns` (list): List of columns to include
- `method` (str): Correlation method
- `**kwargs`: Additional plotting arguments

**Returns:** Matplotlib figure object

---

## Configuration

### PlotConfig

The `PlotConfig` class manages consistent plotting styles and themes.

#### Methods

##### set_style(style=None, palette=None, context=None)

Set plotting style and theme.

**Parameters:**
- `style` (str): Seaborn style
- `palette` (str): Color palette
- `context` (str): Plotting context

---

## Utility Functions

### save_plot(fig, filepath, dpi=300, **kwargs)

Save plot with consistent settings.

**Parameters:**
- `fig`: Matplotlib figure object
- `filepath` (str): Path to save the plot
- `dpi` (int): Resolution
- `**kwargs`: Additional save parameters

### create_subplot_grid(nrows, ncols, figsize=None, **kwargs)

Create subplot grid with consistent styling.

**Parameters:**
- `nrows` (int): Number of rows
- `ncols` (int): Number of columns
- `figsize` (tuple): Figure size tuple
- `**kwargs`: Additional subplot arguments

**Returns:** Tuple of figure and axes array
