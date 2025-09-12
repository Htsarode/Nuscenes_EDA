from pathlib import Path

# Project root
PROJECT_ROOT = Path(__file__).parent.parent

# Data directories
DATA_ROOT = PROJECT_ROOT / "Data"
RAW_DATA = DATA_ROOT / "Raw"
EXTERNAL_DATA = DATA_ROOT / "External"
INTERIM_DATA = DATA_ROOT / "Interim"
PROCESSED_DATA = DATA_ROOT / "Processed"

# Output directories
FIGURES_ROOT = PROJECT_ROOT / "figures"
EXPLORATORY_FIGURES = FIGURES_ROOT / "exploratory"
FINAL_FIGURES = FIGURES_ROOT / "final"
TEMP_FIGURES = FIGURES_ROOT / "temp"
PLOTS_ROOT = PROJECT_ROOT / "plots"

# nuScenes config
NUSCENES_CONFIG = {
	'version': 'v1.0-mini',
	'dataroot': str(RAW_DATA / 'nuscenes' / 'v1.0-mini'),
	'verbose': True
}

def create_directories():
	"""Create all necessary directories"""
	directories = [
		DATA_ROOT, EXTERNAL_DATA, INTERIM_DATA, PROCESSED_DATA, RAW_DATA,
		FIGURES_ROOT, EXPLORATORY_FIGURES, FINAL_FIGURES, TEMP_FIGURES,
		PLOTS_ROOT
	]
	for d in directories:
		d.mkdir(parents=True, exist_ok=True)
