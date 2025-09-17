"""
Statistical Analysis for nuScenes EDA
This module provides comprehensive statistical analysis functionality for nuScenes dataset
specifically designed to support the 22 analysis types in this EDA system.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Any, Optional
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from nuscenes.nuscenes import NuScenes
import warnings
warnings.filterwarnings('ignore')

from config.settings import EXPLORATORY_FIGURES


class NuScenesStatisticalAnalyzer:
    """
    Statistical analyzer specifically designed for nuScenes dataset
    """
    
    def __init__(self, dataroot: str = None, version: str = "v1.0-mini"):
        """
        Initialize statistical analyzer with nuScenes dataset access
        
        Args:
            dataroot: Path to nuScenes dataset
            version: Dataset version
        """
        self.dataroot = dataroot
        self.version = version
        self.nusc = None
        if dataroot:
            try:
                self.nusc = NuScenes(version=version, dataroot=dataroot, verbose=False)
            except Exception as e:
                print(f"Warning: Could not initialize nuScenes: {e}")
    
    def analyze_pedestrian_statistics(self) -> Dict[str, Any]:
        """
        Comprehensive statistical analysis for pedestrian-related data (Analyses 1-6)
        
        Returns:
            Dictionary containing pedestrian statistical analysis
        """
        if not self.nusc:
            return {}
        
        # Collect pedestrian data
        pedestrian_data = []
        pedestrian_categories = [
            'human.pedestrian.adult',
            'human.pedestrian.child',
            'human.pedestrian.construction_worker', 
            'human.pedestrian.police_officer'
        ]
        
        for ann in self.nusc.sample_annotation:
            if ann['category_name'] in pedestrian_categories:
                x, y, z = ann['translation']
                distance = np.sqrt(x**2 + y**2)
                
                # Get visibility level
                visibility = self.nusc.get('visibility', ann['visibility_token'])
                
                # Get behavioral attributes
                is_standing = any(attr for attr in self.nusc.attribute 
                                 if attr['token'] in ann['attribute_tokens'] and 'standing' in attr['name'])
                
                pedestrian_data.append({
                    'category': ann['category_name'],
                    'x': x, 'y': y, 'z': z,
                    'distance': distance,
                    'visibility': visibility['level'],
                    'is_standing': is_standing,
                    'num_lidar_pts': ann['num_lidar_pts'],
                    'num_radar_pts': ann['num_radar_pts']
                })
        
        if not pedestrian_data:
            return {"error": "No pedestrian data found"}
        
        df = pd.DataFrame(pedestrian_data)
        
        # Statistical analyses
        analysis = {
            'total_pedestrians': len(df),
            'category_distribution': df['category'].value_counts().to_dict(),
            'behavioral_stats': {
                'standing_count': df['is_standing'].sum(),
                'standing_percentage': (df['is_standing'].sum() / len(df)) * 100
            },
            'spatial_statistics': {
                'distance_mean': df['distance'].mean(),
                'distance_std': df['distance'].std(),
                'distance_median': df['distance'].median(),
                'distance_range': [df['distance'].min(), df['distance'].max()]
            },
            'visibility_distribution': df['visibility'].value_counts().to_dict(),
            'detection_quality': {
                'avg_lidar_points': df['num_lidar_pts'].mean(),
                'avg_radar_points': df['num_radar_pts'].mean(),
                'detection_correlation': df['num_lidar_pts'].corr(df['distance'])
            },
            'position_analysis': {
                'x_range': [df['x'].min(), df['x'].max()],
                'y_range': [df['y'].min(), df['y'].max()],
                'z_range': [df['z'].min(), df['z'].max()],
                'quadrant_distribution': self._analyze_quadrants(df[['x', 'y']])
            }
        }
        
        return analysis
    
    def analyze_vehicle_statistics(self) -> Dict[str, Any]:
        """
        Comprehensive statistical analysis for vehicle-related data (Analyses 7-9)
        
        Returns:
            Dictionary containing vehicle statistical analysis
        """
        if not self.nusc:
            return {}
        
        vehicle_data = []
        vehicle_categories = [
            'vehicle.car', 'vehicle.truck', 'vehicle.bus.bendy', 'vehicle.bus.rigid',
            'vehicle.motorcycle', 'vehicle.bicycle', 'vehicle.emergency.ambulance',
            'vehicle.emergency.police', 'vehicle.construction', 'vehicle.trailer'
        ]
        
        for ann in self.nusc.sample_annotation:
            if ann['category_name'] in vehicle_categories:
                x, y, z = ann['translation']
                w, l, h = ann['size']
                volume = w * l * h
                distance = np.sqrt(x**2 + y**2)
                
                vehicle_data.append({
                    'category': ann['category_name'],
                    'x': x, 'y': y, 'z': z,
                    'width': w, 'length': l, 'height': h,
                    'volume': volume,
                    'distance': distance,
                    'num_lidar_pts': ann['num_lidar_pts'],
                    'num_radar_pts': ann['num_radar_pts']
                })
        
        if not vehicle_data:
            return {"error": "No vehicle data found"}
        
        df = pd.DataFrame(vehicle_data)
        
        # Group vehicles by type
        df['vehicle_type'] = df['category'].apply(self._classify_vehicle_type)
        
        analysis = {
            'total_vehicles': len(df),
            'category_distribution': df['category'].value_counts().to_dict(),
            'type_distribution': df['vehicle_type'].value_counts().to_dict(),
            'size_statistics': {
                'volume_stats': {
                    'mean': df['volume'].mean(),
                    'std': df['volume'].std(),
                    'median': df['volume'].median(),
                    'range': [df['volume'].min(), df['volume'].max()]
                },
                'dimension_stats': {
                    'width': {'mean': df['width'].mean(), 'std': df['width'].std()},
                    'length': {'mean': df['length'].mean(), 'std': df['length'].std()},
                    'height': {'mean': df['height'].mean(), 'std': df['height'].std()}
                }
            },
            'spatial_distribution': {
                'distance_stats': {
                    'mean': df['distance'].mean(),
                    'std': df['distance'].std(),
                    'median': df['distance'].median()
                },
                'position_variance': {
                    'x_variance': df['x'].var(),
                    'y_variance': df['y'].var(),
                    'z_variance': df['z'].var()
                }
            },
            'detection_analysis': {
                'avg_lidar_points': df['num_lidar_pts'].mean(),
                'avg_radar_points': df['num_radar_pts'].mean(),
                'size_detection_correlation': df['volume'].corr(df['num_lidar_pts'])
            }
        }
        
        return analysis
    
    def analyze_environmental_statistics(self) -> Dict[str, Any]:
        """
        Statistical analysis for environmental conditions (Analyses 10-13)
        
        Returns:
            Dictionary containing environmental statistical analysis
        """
        if not self.nusc:
            return {}
        
        scene_data = []
        for scene in self.nusc.scene:
            log = self.nusc.get('log', scene['log_token'])
            scene_name = scene['name'].lower()
            
            # Extract conditions from scene name
            weather = self._extract_weather_condition(scene_name)
            time_of_day = self._extract_time_condition(scene_name)
            
            # Calculate scene duration and object count
            first_sample = self.nusc.get('sample', scene['first_sample_token'])
            last_sample = self.nusc.get('sample', scene['last_sample_token'])
            duration = (last_sample['timestamp'] - first_sample['timestamp']) / 1e6
            
            # Count total annotations in scene
            total_annotations = 0
            current_sample = scene['first_sample_token']
            while current_sample:
                sample = self.nusc.get('sample', current_sample)
                total_annotations += len(sample['anns'])
                current_sample = sample['next']
            
            scene_data.append({
                'scene_name': scene['name'],
                'location': log.get('location', 'unknown'),
                'weather': weather,
                'time_of_day': time_of_day,
                'duration': duration,
                'total_annotations': total_annotations,
                'annotation_density': total_annotations / duration if duration > 0 else 0,
                'num_samples': scene['nbr_samples']
            })
        
        df = pd.DataFrame(scene_data)
        
        analysis = {
            'total_scenes': len(df),
            'location_distribution': df['location'].value_counts().to_dict(),
            'weather_distribution': df['weather'].value_counts().to_dict(),
            'time_distribution': df['time_of_day'].value_counts().to_dict(),
            'temporal_statistics': {
                'duration_stats': {
                    'mean': df['duration'].mean(),
                    'std': df['duration'].std(),
                    'median': df['duration'].median(),
                    'range': [df['duration'].min(), df['duration'].max()]
                },
                'sample_stats': {
                    'mean_samples': df['num_samples'].mean(),
                    'total_samples': df['num_samples'].sum()
                }
            },
            'object_density_analysis': {
                'mean_density': df['annotation_density'].mean(),
                'density_by_weather': df.groupby('weather')['annotation_density'].mean().to_dict(),
                'density_by_time': df.groupby('time_of_day')['annotation_density'].mean().to_dict()
            },
            'geographical_diversity': {
                'unique_locations': df['location'].nunique(),
                'location_scene_count': df['location'].value_counts().to_dict()
            }
        }
        
        return analysis
    
    def analyze_ego_vehicle_statistics(self) -> Dict[str, Any]:
        """
        Statistical analysis for ego vehicle motion (Analyses 18-20)
        
        Returns:
            Dictionary containing ego vehicle statistical analysis
        """
        if not self.nusc:
            return {}
        
        # Process ego poses to extract motion statistics
        ego_poses = sorted(self.nusc.ego_pose, key=lambda x: x['timestamp'])
        
        if len(ego_poses) < 2:
            return {"error": "Insufficient ego pose data"}
        
        motion_data = []
        for i in range(1, len(ego_poses)):
            prev_pose = ego_poses[i-1]
            curr_pose = ego_poses[i]
            
            # Calculate displacement
            prev_pos = np.array(prev_pose['translation'])
            curr_pos = np.array(curr_pose['translation'])
            displacement = np.linalg.norm(curr_pos - prev_pos)
            
            # Calculate time difference
            dt = (curr_pose['timestamp'] - prev_pose['timestamp']) / 1e6
            
            # Calculate speed
            speed = displacement / dt if dt > 0 else 0
            
            motion_data.append({
                'timestamp': curr_pose['timestamp'],
                'displacement': displacement,
                'speed_ms': speed,
                'speed_kmh': speed * 3.6,
                'time_delta': dt,
                'x': curr_pos[0],
                'y': curr_pos[1],
                'z': curr_pos[2]
            })
        
        df = pd.DataFrame(motion_data)
        
        # Calculate acceleration
        df['acceleration'] = df['speed_ms'].diff() / df['time_delta']
        
        analysis = {
            'total_poses': len(df),
            'speed_statistics': {
                'mean_speed_kmh': df['speed_kmh'].mean(),
                'max_speed_kmh': df['speed_kmh'].max(),
                'min_speed_kmh': df['speed_kmh'].min(),
                'std_speed_kmh': df['speed_kmh'].std(),
                'speed_percentiles': {
                    '25th': df['speed_kmh'].quantile(0.25),
                    '50th': df['speed_kmh'].quantile(0.50),
                    '75th': df['speed_kmh'].quantile(0.75),
                    '95th': df['speed_kmh'].quantile(0.95)
                }
            },
            'acceleration_statistics': {
                'mean_acceleration': df['acceleration'].mean(),
                'max_acceleration': df['acceleration'].max(),
                'min_acceleration': df['acceleration'].min(),
                'std_acceleration': df['acceleration'].std()
            },
            'motion_patterns': {
                'stationary_time_pct': (df['speed_ms'] < 0.5).sum() / len(df) * 100,
                'slow_speed_time_pct': ((df['speed_ms'] >= 0.5) & (df['speed_ms'] < 5)).sum() / len(df) * 100,
                'normal_speed_time_pct': ((df['speed_ms'] >= 5) & (df['speed_ms'] < 15)).sum() / len(df) * 100,
                'high_speed_time_pct': (df['speed_ms'] >= 15).sum() / len(df) * 100
            },
            'trajectory_statistics': {
                'total_distance': df['displacement'].sum(),
                'position_variance': {
                    'x_variance': df['x'].var(),
                    'y_variance': df['y'].var(),
                    'z_variance': df['z'].var()
                }
            }
        }
        
        return analysis
    
    def analyze_rare_class_statistics(self) -> Dict[str, Any]:
        """
        Statistical analysis for rare class occurrences (Analysis 22)
        
        Returns:
            Dictionary containing rare class statistical analysis
        """
        if not self.nusc:
            return {}
        
        # Count all categories
        category_counts = {}
        total_annotations = 0
        
        for ann in self.nusc.sample_annotation:
            total_annotations += 1
            category = ann['category_name']
            category_counts[category] = category_counts.get(category, 0) + 1
        
        # Statistical analysis of class distribution
        counts = list(category_counts.values())
        
        # Define rare classes (less than 1% of total)
        rare_threshold = total_annotations * 0.01
        rare_classes = {k: v for k, v in category_counts.items() if v < rare_threshold}
        common_classes = {k: v for k, v in category_counts.items() if v >= rare_threshold}
        
        analysis = {
            'total_annotations': total_annotations,
            'total_categories': len(category_counts),
            'category_counts': category_counts,
            'distribution_statistics': {
                'mean_count': np.mean(counts),
                'median_count': np.median(counts),
                'std_count': np.std(counts),
                'min_count': np.min(counts),
                'max_count': np.max(counts),
                'gini_coefficient': self._calculate_gini_coefficient(counts)
            },
            'rare_class_analysis': {
                'rare_classes': rare_classes,
                'rare_class_count': len(rare_classes),
                'rare_class_percentage': len(rare_classes) / len(category_counts) * 100,
                'total_rare_instances': sum(rare_classes.values()),
                'rare_instances_percentage': sum(rare_classes.values()) / total_annotations * 100
            },
            'common_class_analysis': {
                'common_classes': common_classes,
                'common_class_count': len(common_classes),
                'total_common_instances': sum(common_classes.values())
            }
        }
        
        return analysis
    
    def generate_comprehensive_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive statistical report for all analysis types
        
        Returns:
            Complete statistical analysis report
        """
        report = {
            'pedestrian_statistics': self.analyze_pedestrian_statistics(),
            'vehicle_statistics': self.analyze_vehicle_statistics(),
            'environmental_statistics': self.analyze_environmental_statistics(),
            'ego_vehicle_statistics': self.analyze_ego_vehicle_statistics(),
            'rare_class_statistics': self.analyze_rare_class_statistics()
        }
        
        # Generate summary statistics
        report['summary'] = {
            'total_analyses': 5,
            'dataset_version': self.version,
            'analysis_timestamp': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        return report
    
    # Helper methods
    def _classify_vehicle_type(self, category_name: str) -> str:
        """Classify vehicle into broader categories"""
        if 'car' in category_name:
            return 'Car'
        elif 'truck' in category_name:
            return 'Truck'
        elif 'bus' in category_name:
            return 'Bus'
        elif 'motorcycle' in category_name:
            return 'Motorcycle'
        elif 'bicycle' in category_name:
            return 'Bicycle'
        elif 'emergency' in category_name:
            return 'Emergency'
        else:
            return 'Other'
    
    def _extract_weather_condition(self, scene_name: str) -> str:
        """Extract weather condition from scene name"""
        if 'rain' in scene_name:
            return 'rain'
        elif 'night' in scene_name:
            return 'night'
        else:
            return 'clear'
    
    def _extract_time_condition(self, scene_name: str) -> str:
        """Extract time condition from scene name"""
        if 'night' in scene_name:
            return 'night'
        else:
            return 'day'
    
    def _analyze_quadrants(self, position_df: pd.DataFrame) -> Dict[str, int]:
        """Analyze position distribution by quadrants"""
        quadrants = {
            'Q1 (+,+)': ((position_df['x'] >= 0) & (position_df['y'] >= 0)).sum(),
            'Q2 (-,+)': ((position_df['x'] < 0) & (position_df['y'] >= 0)).sum(),
            'Q3 (-,-)': ((position_df['x'] < 0) & (position_df['y'] < 0)).sum(),
            'Q4 (+,-)': ((position_df['x'] >= 0) & (position_df['y'] < 0)).sum()
        }
        return quadrants
    
    def _calculate_gini_coefficient(self, values: List[float]) -> float:
        """Calculate Gini coefficient for measuring inequality"""
        values = sorted(values)
        n = len(values)
        if n == 0:
            return 0
        
        cumsum = np.cumsum(values)
        return (n + 1 - 2 * np.sum(cumsum) / cumsum[-1]) / n
        
    def analyze_object_distribution(self, annotations_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze distribution of objects in the dataset
        
        Args:
            annotations_df: DataFrame containing annotations
            
        Returns:
            Dictionary containing analysis results
        """
        if annotations_df.empty:
            return {}
        
        analysis = {}
        
        # Category distribution
        category_counts = annotations_df['category_name'].value_counts()
        analysis['category_distribution'] = category_counts.to_dict()
        
        # Size analysis by category
        size_analysis = annotations_df.groupby('category_name').agg({
            'size_width': ['mean', 'std', 'min', 'max'],
            'size_length': ['mean', 'std', 'min', 'max'],
            'size_height': ['mean', 'std', 'min', 'max']
        }).round(3)
        analysis['size_statistics'] = size_analysis
        
        # Point cloud density analysis
        point_analysis = annotations_df.groupby('category_name').agg({
            'num_lidar_pts': ['mean', 'std', 'min', 'max', 'median'],
            'num_radar_pts': ['mean', 'std', 'min', 'max', 'median']
        }).round(3)
        analysis['point_statistics'] = point_analysis
        
        # Visibility analysis
        if 'visibility_level' in annotations_df.columns:
            visibility_analysis = annotations_df.groupby(['category_name', 'visibility_level']).size().unstack(fill_value=0)
            analysis['visibility_distribution'] = visibility_analysis
        
        return analysis
    
    def analyze_sensor_coverage(self, sensor_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze sensor data coverage and distribution
        
        Args:
            sensor_df: DataFrame containing sensor data
            
        Returns:
            Dictionary containing sensor analysis
        """
        if sensor_df.empty:
            return {}
        
        analysis = {}
        
        # Sensor modality distribution
        modality_counts = sensor_df['sensor_modality'].value_counts()
        analysis['modality_distribution'] = modality_counts.to_dict()
        
        # Channel distribution per modality
        channel_dist = sensor_df.groupby('sensor_modality')['sensor_channel'].value_counts()
        analysis['channel_distribution'] = channel_dist.to_dict()
        
        # Key frame analysis
        keyframe_analysis = sensor_df.groupby(['sensor_modality', 'sensor_channel']).agg({
            'is_key_frame': ['sum', 'count']
        })
        keyframe_analysis.columns = ['key_frames', 'total_frames']
        keyframe_analysis['keyframe_ratio'] = keyframe_analysis['key_frames'] / keyframe_analysis['total_frames']
        analysis['keyframe_statistics'] = keyframe_analysis.round(3)
        
        # Timestamp analysis (if available)
        if 'timestamp' in sensor_df.columns:
            sensor_df['timestamp_dt'] = pd.to_datetime(sensor_df['timestamp'], unit='us')
            time_analysis = sensor_df.groupby('sensor_modality')['timestamp_dt'].agg(['min', 'max', 'count'])
            time_analysis['duration_seconds'] = (time_analysis['max'] - time_analysis['min']).dt.total_seconds()
            analysis['temporal_coverage'] = time_analysis
        
        return analysis
    
    def analyze_spatial_distribution(self, annotations_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze spatial distribution of objects
        
        Args:
            annotations_df: DataFrame containing annotations
            
        Returns:
            Dictionary containing spatial analysis
        """
        if annotations_df.empty:
            return {}
        
        analysis = {}
        
        # Translation statistics
        translation_stats = annotations_df.groupby('category_name').agg({
            'translation_x': ['mean', 'std', 'min', 'max'],
            'translation_y': ['mean', 'std', 'min', 'max'],
            'translation_z': ['mean', 'std', 'min', 'max']
        }).round(3)
        analysis['translation_statistics'] = translation_stats
        
        # Distance from ego vehicle
        annotations_df['distance_from_ego'] = np.sqrt(
            annotations_df['translation_x']**2 + 
            annotations_df['translation_y']**2 + 
            annotations_df['translation_z']**2
        )
        
        distance_stats = annotations_df.groupby('category_name')['distance_from_ego'].agg([
            'mean', 'std', 'min', 'max', 'median'
        ]).round(3)
        analysis['distance_statistics'] = distance_stats
        
        # Angular distribution (in ego vehicle frame)
        annotations_df['angle_from_ego'] = np.arctan2(
            annotations_df['translation_y'], 
            annotations_df['translation_x']
        ) * 180 / np.pi
        
        angle_stats = annotations_df.groupby('category_name')['angle_from_ego'].agg([
            'mean', 'std', 'min', 'max'
        ]).round(3)
        analysis['angular_statistics'] = angle_stats
        
        return analysis
    
    def analyze_scene_diversity(self, scenes_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze diversity across scenes
        
        Args:
            scenes_df: DataFrame containing scene metadata
            
        Returns:
            Dictionary containing scene diversity analysis
        """
        if scenes_df.empty:
            return {}
        
        analysis = {}
        
        # Location distribution
        if 'log_location' in scenes_df.columns:
            location_counts = scenes_df['log_location'].value_counts()
            analysis['location_distribution'] = location_counts.to_dict()
        
        # Sample count statistics
        if 'nbr_samples' in scenes_df.columns:
            sample_stats = scenes_df['nbr_samples'].describe()
            analysis['samples_per_scene'] = sample_stats.to_dict()
        
        # Time diversity (if date information available)
        if 'log_date_captured' in scenes_df.columns:
            scenes_df['log_date'] = pd.to_datetime(scenes_df['log_date_captured'])
            date_range = scenes_df['log_date'].max() - scenes_df['log_date'].min()
            analysis['temporal_span_days'] = date_range.days
            
            # Monthly distribution
            scenes_df['month'] = scenes_df['log_date'].dt.month
            monthly_dist = scenes_df['month'].value_counts().sort_index()
            analysis['monthly_distribution'] = monthly_dist.to_dict()
        
        return analysis
    
    def generate_correlation_analysis(self, annotations_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate correlation analysis for numerical features
        
        Args:
            annotations_df: DataFrame containing annotations
            
        Returns:
            Dictionary containing correlation analysis
        """
        if annotations_df.empty:
            return {}
        
        # Select numerical columns
        numerical_cols = [
            'translation_x', 'translation_y', 'translation_z',
            'size_width', 'size_length', 'size_height',
            'num_lidar_pts', 'num_radar_pts'
        ]
        
        available_cols = [col for col in numerical_cols if col in annotations_df.columns]
        
        if len(available_cols) < 2:
            return {}
        
        analysis = {}
        
        # Correlation matrix
        correlation_matrix = annotations_df[available_cols].corr()
        analysis['correlation_matrix'] = correlation_matrix
        
        # Strong correlations (absolute value > 0.5)
        strong_corr = []
        for i in range(len(correlation_matrix.columns)):
            for j in range(i+1, len(correlation_matrix.columns)):
                corr_val = correlation_matrix.iloc[i, j]
                if abs(corr_val) > 0.5:
                    strong_corr.append({
                        'feature_1': correlation_matrix.columns[i],
                        'feature_2': correlation_matrix.columns[j],
                        'correlation': round(corr_val, 3)
                    })
        
        analysis['strong_correlations'] = strong_corr
        
        return analysis
    
    def detect_outliers(self, annotations_df: pd.DataFrame, method: str = 'iqr') -> Dict[str, Any]:
        """
        Detect outliers in the dataset
        
        Args:
            annotations_df: DataFrame containing annotations
            method: Outlier detection method ('iqr', 'zscore')
            
        Returns:
            Dictionary containing outlier analysis
        """
        if annotations_df.empty:
            return {}
        
        numerical_cols = [
            'translation_x', 'translation_y', 'translation_z',
            'size_width', 'size_length', 'size_height',
            'num_lidar_pts', 'num_radar_pts'
        ]
        
        available_cols = [col for col in numerical_cols if col in annotations_df.columns]
        
        if not available_cols:
            return {}
        
        outlier_analysis = {}
        
        for col in available_cols:
            col_data = annotations_df[col].dropna()
            outliers = []
            
            if method == 'iqr':
                Q1 = col_data.quantile(0.25)
                Q3 = col_data.quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                outliers = col_data[(col_data < lower_bound) | (col_data > upper_bound)]
                
            elif method == 'zscore':
                z_scores = np.abs(stats.zscore(col_data))
                outliers = col_data[z_scores > 3]
            
            outlier_analysis[col] = {
                'count': len(outliers),
                'percentage': round(len(outliers) / len(col_data) * 100, 2),
                'values': outliers.tolist()[:10]  # First 10 outlier values
            }
        
        return outlier_analysis


class StatisticalAnalyzer:
    """
    General statistical analyzer with basic functionality
    """
    
    def __init__(self):
        pass
    
    def describe_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate descriptive statistics for dataset
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with descriptive statistics
        """
        return df.describe(include='all')
    
    def missing_value_analysis(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Analyze missing values in dataset
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with missing value statistics
        """
        missing_data = df.isnull().sum()
        missing_percent = 100 * df.isnull().sum() / len(df)
        
        missing_table = pd.DataFrame({
            'Missing Count': missing_data,
            'Missing Percentage': missing_percent
        })
        
        return missing_table.sort_values('Missing Count', ascending=False)
    
    def data_type_analysis(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Analyze data types in dataset
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with data type information
        """
        type_info = pd.DataFrame({
            'Column': df.columns,
            'Data Type': df.dtypes,
            'Non-Null Count': df.count(),
            'Null Count': df.isnull().sum(),
            'Unique Values': df.nunique()
        })
        
        return type_info
    
    def correlation_matrix(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate correlation matrix for numerical columns
        
        Args:
            df: Input DataFrame
            
        Returns:
            Correlation matrix
        """
        numerical_df = df.select_dtypes(include=[np.number])
        return numerical_df.corr()
    
    def distribution_analysis(self, df: pd.DataFrame, column: str) -> Dict[str, Any]:
        """
        Analyze distribution of a specific column
        
        Args:
            df: Input DataFrame
            column: Column name to analyze
            
        Returns:
            Dictionary with distribution statistics
        """
        if column not in df.columns:
            return {}
        
        col_data = df[column].dropna()
        
        if df[column].dtype in ['object', 'category']:
            # Categorical analysis
            value_counts = col_data.value_counts()
            return {
                'type': 'categorical',
                'unique_values': len(value_counts),
                'most_frequent': value_counts.index[0],
                'frequency': value_counts.iloc[0],
                'distribution': value_counts.to_dict()
            }
        else:
            # Numerical analysis
            return {
                'type': 'numerical',
                'mean': col_data.mean(),
                'median': col_data.median(),
                'std': col_data.std(),
                'min': col_data.min(),
                'max': col_data.max(),
                'skewness': stats.skew(col_data),
                'kurtosis': stats.kurtosis(col_data)
            }