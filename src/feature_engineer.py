"""
Feature Engineering for nuScenes EDA
This module provides comprehensive feature engineering functionality for nuScenes dataset
specifically designed to support the 22 analysis types in this EDA system.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from scipy.spatial.distance import cdist
from sklearn.preprocessing import StandardScaler, LabelEncoder
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.geometry_utils import transform_matrix
import warnings
warnings.filterwarnings('ignore')


class NuScenesFeatureEngineer:
    """
    Feature engineer specifically designed for nuScenes dataset
    """
    
    def __init__(self, dataroot: str = None, version: str = "v1.0-mini"):
        """
        Initialize feature engineer with nuScenes dataset access
        
        Args:
            dataroot: Path to nuScenes dataset  
            version: Dataset version
        """
        self.label_encoders = {}
        self.scalers = {}
        self.dataroot = dataroot
        self.version = version
        self.nusc = None
        if dataroot:
            try:
                self.nusc = NuScenes(version=version, dataroot=dataroot, verbose=False)
            except Exception as e:
                print(f"Warning: Could not initialize nuScenes: {e}")
    
    def engineer_pedestrian_features(self) -> Dict[str, Any]:
        """
        Engineer features specific to pedestrian analyses (1-6)
        
        Returns:
            Dictionary containing pedestrian-specific features
        """
        if not self.nusc:
            return {}
        
        pedestrian_features = []
        
        # Get pedestrian categories
        pedestrian_categories = [
            'human.pedestrian.adult',
            'human.pedestrian.child', 
            'human.pedestrian.construction_worker',
            'human.pedestrian.police_officer'
        ]
        
        for ann in self.nusc.sample_annotation:
            if ann['category_name'] in pedestrian_categories:
                # Basic spatial features
                x, y, z = ann['translation']
                w, l, h = ann['size']
                
                # Distance and angle from ego vehicle
                distance_2d = np.sqrt(x**2 + y**2)
                distance_3d = np.sqrt(x**2 + y**2 + z**2)
                angle_rad = np.arctan2(y, x)
                angle_deg = np.degrees(angle_rad)
                
                # Behavioral features from attributes
                attributes = ann['attribute_tokens']
                is_standing = any(attr for attr in self.nusc.attribute 
                                 if attr['token'] in attributes and 'standing' in attr['name'])
                is_moving = any(attr for attr in self.nusc.attribute 
                               if attr['token'] in attributes and 'moving' in attr['name'])
                
                # Visibility features
                visibility = self.nusc.get('visibility', ann['visibility_token'])
                visibility_level = visibility['level']
                
                # Road proximity features (simplified)
                is_on_road = abs(x) < 20 and abs(y) < 3  # Rough road detection
                
                pedestrian_features.append({
                    'token': ann['token'],
                    'category': ann['category_name'],
                    'x': x, 'y': y, 'z': z,
                    'width': w, 'length': l, 'height': h,
                    'distance_2d': distance_2d,
                    'distance_3d': distance_3d,
                    'angle_deg': angle_deg,
                    'is_standing': is_standing,
                    'is_moving': is_moving,
                    'visibility_level': visibility_level,
                    'is_on_road': is_on_road,
                    'num_lidar_pts': ann['num_lidar_pts'],
                    'num_radar_pts': ann['num_radar_pts'],
                    'detection_confidence': np.log1p(ann['num_lidar_pts']) / distance_2d if distance_2d > 0 else 0
                })
        
        return {
            'features': pedestrian_features,
            'total_pedestrians': len(pedestrian_features),
            'standing_count': sum(1 for p in pedestrian_features if p['is_standing']),
            'moving_count': sum(1 for p in pedestrian_features if p['is_moving']),
            'average_distance': np.mean([p['distance_2d'] for p in pedestrian_features]) if pedestrian_features else 0
        }
    
    def engineer_vehicle_features(self) -> Dict[str, Any]:
        """
        Engineer features specific to vehicle analyses (7-9)
        
        Returns:
            Dictionary containing vehicle-specific features
        """
        if not self.nusc:
            return {}
        
        vehicle_features = []
        
        # Vehicle categories
        vehicle_categories = [
            'vehicle.car', 'vehicle.truck', 'vehicle.bus.bendy', 'vehicle.bus.rigid',
            'vehicle.motorcycle', 'vehicle.bicycle', 'vehicle.emergency.ambulance',
            'vehicle.emergency.police', 'vehicle.construction', 'vehicle.trailer'
        ]
        
        for ann in self.nusc.sample_annotation:
            if ann['category_name'] in vehicle_categories:
                x, y, z = ann['translation']
                w, l, h = ann['size']
                
                # Spatial features
                distance_2d = np.sqrt(x**2 + y**2)
                distance_3d = np.sqrt(x**2 + y**2 + z**2)
                angle_deg = np.degrees(np.arctan2(y, x))
                
                # Size-based features
                volume = w * l * h
                aspect_ratio = l / w if w > 0 else 0
                
                # Vehicle type classification
                vehicle_type = self._classify_vehicle_type(ann['category_name'])
                vehicle_size = self._classify_vehicle_size(volume)
                
                # Lane position (simplified)
                lane_position = 'left' if y > 2 else 'right' if y < -2 else 'center'
                
                # Detection quality
                detection_score = (ann['num_lidar_pts'] + ann['num_radar_pts']) / volume if volume > 0 else 0
                
                vehicle_features.append({
                    'token': ann['token'],
                    'category': ann['category_name'],
                    'vehicle_type': vehicle_type,
                    'vehicle_size': vehicle_size,
                    'x': x, 'y': y, 'z': z,
                    'width': w, 'length': l, 'height': h,
                    'volume': volume,
                    'aspect_ratio': aspect_ratio,
                    'distance_2d': distance_2d,
                    'distance_3d': distance_3d,
                    'angle_deg': angle_deg,
                    'lane_position': lane_position,
                    'num_lidar_pts': ann['num_lidar_pts'],
                    'num_radar_pts': ann['num_radar_pts'],
                    'detection_score': detection_score
                })
        
        return {
            'features': vehicle_features,
            'total_vehicles': len(vehicle_features),
            'type_distribution': self._get_distribution(vehicle_features, 'vehicle_type'),
            'size_distribution': self._get_distribution(vehicle_features, 'vehicle_size'),
            'average_volume': np.mean([v['volume'] for v in vehicle_features]) if vehicle_features else 0
        }
    
    def engineer_environmental_features(self) -> Dict[str, Any]:
        """
        Engineer features for environmental analyses (10-13)
        
        Returns:
            Dictionary containing environmental features
        """
        if not self.nusc:
            return {}
        
        environmental_features = []
        
        for scene in self.nusc.scene:
            log = self.nusc.get('log', scene['log_token'])
            
            # Scene name analysis for weather/time conditions
            scene_name = scene['name'].lower()
            
            # Weather classification
            weather = self._extract_weather_from_name(scene_name)
            time_of_day = self._extract_time_from_name(scene_name)
            
            # Location features
            location = log.get('location', 'unknown')
            vehicle_type = log.get('vehicle', 'unknown')
            
            # Sample count and duration
            first_sample = self.nusc.get('sample', scene['first_sample_token'])
            last_sample = self.nusc.get('sample', scene['last_sample_token'])
            duration = (last_sample['timestamp'] - first_sample['timestamp']) / 1e6  # seconds
            
            # Object density in scene
            scene_annotations = []
            current_sample = scene['first_sample_token']
            while current_sample:
                sample = self.nusc.get('sample', current_sample)
                scene_annotations.extend(sample['anns'])
                current_sample = sample['next']
            
            environmental_features.append({
                'scene_token': scene['token'],
                'scene_name': scene['name'],
                'location': location,
                'vehicle_type': vehicle_type,
                'weather': weather,
                'time_of_day': time_of_day,
                'duration_seconds': duration,
                'total_annotations': len(scene_annotations),
                'annotation_density': len(scene_annotations) / duration if duration > 0 else 0,
                'num_samples': scene['nbr_samples']
            })
        
        return {
            'features': environmental_features,
            'total_scenes': len(environmental_features),
            'weather_distribution': self._get_distribution(environmental_features, 'weather'),
            'time_distribution': self._get_distribution(environmental_features, 'time_of_day'),
            'location_distribution': self._get_distribution(environmental_features, 'location')
        }
    
    def engineer_ego_vehicle_features(self) -> Dict[str, Any]:
        """
        Engineer features for ego vehicle analyses (18-20)
        
        Returns:
            Dictionary containing ego vehicle motion features
        """
        if not self.nusc:
            return {}
        
        ego_features = []
        
        # Process ego poses chronologically
        ego_poses = sorted(self.nusc.ego_pose, key=lambda x: x['timestamp'])
        
        for i in range(1, len(ego_poses)):
            prev_pose = ego_poses[i-1]
            curr_pose = ego_poses[i]
            
            # Position and orientation
            prev_pos = np.array(prev_pose['translation'])
            curr_pos = np.array(curr_pose['translation'])
            
            # Calculate motion features
            displacement = curr_pos - prev_pos
            distance = np.linalg.norm(displacement)
            dt = (curr_pose['timestamp'] - prev_pose['timestamp']) / 1e6  # seconds
            
            # Speed and acceleration
            speed = distance / dt if dt > 0 else 0
            
            # Direction of motion
            motion_angle = np.degrees(np.arctan2(displacement[1], displacement[0]))
            
            # Rotation analysis
            prev_rot = prev_pose['rotation']
            curr_rot = curr_pose['rotation']
            
            ego_features.append({
                'timestamp': curr_pose['timestamp'],
                'position_x': curr_pos[0],
                'position_y': curr_pos[1],
                'position_z': curr_pos[2],
                'displacement': distance,
                'speed_ms': speed,
                'speed_kmh': speed * 3.6,
                'motion_angle_deg': motion_angle,
                'time_delta': dt
            })
        
        if ego_features:
            # Calculate acceleration
            for i in range(1, len(ego_features)):
                prev_speed = ego_features[i-1]['speed_ms']
                curr_speed = ego_features[i]['speed_ms']
                dt = ego_features[i]['time_delta']
                acceleration = (curr_speed - prev_speed) / dt if dt > 0 else 0
                ego_features[i]['acceleration'] = acceleration
        
        return {
            'features': ego_features,
            'total_poses': len(ego_features),
            'speed_stats': {
                'mean_speed_kmh': np.mean([f['speed_kmh'] for f in ego_features]) if ego_features else 0,
                'max_speed_kmh': np.max([f['speed_kmh'] for f in ego_features]) if ego_features else 0,
                'min_speed_kmh': np.min([f['speed_kmh'] for f in ego_features]) if ego_features else 0
            },
            'motion_pattern': self._analyze_motion_pattern(ego_features)
        }
    
    def _classify_vehicle_type(self, category_name: str) -> str:
        """Classify vehicle into broader type categories"""
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
        elif 'construction' in category_name:
            return 'Construction'
        else:
            return 'Other'
    
    def _classify_vehicle_size(self, volume: float) -> str:
        """Classify vehicle size based on volume"""
        if volume < 10:
            return 'Small'
        elif volume < 50:
            return 'Medium'
        else:
            return 'Large'
    
    def _extract_weather_from_name(self, scene_name: str) -> str:
        """Extract weather conditions from scene name"""
        if 'rain' in scene_name:
            return 'rain'
        elif 'night' in scene_name:
            return 'night'
        else:
            return 'clear'
    
    def _extract_time_from_name(self, scene_name: str) -> str:
        """Extract time of day from scene name"""
        if 'night' in scene_name:
            return 'night'
        else:
            return 'day'
    
    def _get_distribution(self, features: List[Dict], key: str) -> Dict[str, int]:
        """Get distribution of values for a specific key"""
        distribution = {}
        for feature in features:
            value = feature.get(key, 'unknown')
            distribution[value] = distribution.get(value, 0) + 1
        return distribution
    
    def _analyze_motion_pattern(self, ego_features: List[Dict]) -> str:
        """Analyze overall motion pattern of ego vehicle"""
        if not ego_features:
            return 'stationary'
        
        speeds = [f['speed_ms'] for f in ego_features]
        avg_speed = np.mean(speeds)
        
        if avg_speed < 1:
            return 'stationary'
        elif avg_speed < 5:
            return 'slow'
        elif avg_speed < 15:
            return 'normal'
        else:
            return 'fast'
    
    def engineer_spatial_features(self, annotations_df: pd.DataFrame) -> pd.DataFrame:
        """
        Engineer spatial features from annotations
        
        Args:
            annotations_df: DataFrame containing annotations
            
        Returns:
            DataFrame with additional spatial features
        """
        df = annotations_df.copy()
        
        # Distance from ego vehicle
        df['distance_from_ego'] = np.sqrt(
            df['translation_x']**2 + 
            df['translation_y']**2 + 
            df['translation_z']**2
        )
        
        # 2D distance (ignoring height)
        df['distance_2d_from_ego'] = np.sqrt(
            df['translation_x']**2 + 
            df['translation_y']**2
        )
        
        # Angular position relative to ego vehicle
        df['angle_from_ego_rad'] = np.arctan2(
            df['translation_y'], 
            df['translation_x']
        )
        df['angle_from_ego_deg'] = df['angle_from_ego_rad'] * 180 / np.pi
        
        # Quadrant classification
        df['quadrant'] = np.select([
            (df['translation_x'] >= 0) & (df['translation_y'] >= 0),
            (df['translation_x'] < 0) & (df['translation_y'] >= 0),
            (df['translation_x'] < 0) & (df['translation_y'] < 0),
            (df['translation_x'] >= 0) & (df['translation_y'] < 0)
        ], ['Q1', 'Q2', 'Q3', 'Q4'], default='Unknown')
        
        # Distance bins
        df['distance_bin'] = pd.cut(
            df['distance_from_ego'],
            bins=[0, 10, 30, 50, 100, np.inf],
            labels=['Very Close', 'Close', 'Medium', 'Far', 'Very Far']
        )
        
        # Height categories
        if 'translation_z' in df.columns:
            df['height_category'] = pd.cut(
                df['translation_z'],
                bins=[-np.inf, -1, 0, 1, 2, np.inf],
                labels=['Below Ground', 'Ground Level', 'Low', 'Medium', 'High']
            )
        
        return df
    
    def engineer_size_features(self, annotations_df: pd.DataFrame) -> pd.DataFrame:
        """
        Engineer size-related features from annotations
        
        Args:
            annotations_df: DataFrame containing annotations
            
        Returns:
            DataFrame with additional size features
        """
        df = annotations_df.copy()
        
        # Volume calculation
        df['volume'] = df['size_width'] * df['size_length'] * df['size_height']
        
        # Aspect ratios
        df['width_length_ratio'] = df['size_width'] / (df['size_length'] + 1e-8)
        df['width_height_ratio'] = df['size_width'] / (df['size_height'] + 1e-8)
        df['length_height_ratio'] = df['size_length'] / (df['size_height'] + 1e-8)
        
        # Surface area approximation (assuming box shape)
        df['surface_area'] = 2 * (
            df['size_width'] * df['size_length'] +
            df['size_width'] * df['size_height'] +
            df['size_length'] * df['size_height']
        )
        
        # Maximum dimension
        df['max_dimension'] = df[['size_width', 'size_length', 'size_height']].max(axis=1)
        df['min_dimension'] = df[['size_width', 'size_length', 'size_height']].min(axis=1)
        
        # Size categories by volume
        df['size_category'] = pd.cut(
            df['volume'],
            bins=[0, 1, 10, 50, 200, np.inf],
            labels=['Tiny', 'Small', 'Medium', 'Large', 'Huge']
        )
        
        return df
    
    def engineer_point_cloud_features(self, annotations_df: pd.DataFrame) -> pd.DataFrame:
        """
        Engineer features related to point cloud data
        
        Args:
            annotations_df: DataFrame containing annotations
            
        Returns:
            DataFrame with point cloud features
        """
        df = annotations_df.copy()
        
        # Point density features
        df['lidar_point_density'] = df['num_lidar_pts'] / (df['volume'] + 1e-8)
        df['radar_point_density'] = df['num_radar_pts'] / (df['volume'] + 1e-8)
        
        # Point ratio features
        total_points = df['num_lidar_pts'] + df['num_radar_pts'] + 1e-8
        df['lidar_point_ratio'] = df['num_lidar_pts'] / total_points
        df['radar_point_ratio'] = df['num_radar_pts'] / total_points
        
        # Point categories
        df['lidar_point_category'] = pd.cut(
            df['num_lidar_pts'],
            bins=[0, 10, 50, 200, 1000, np.inf],
            labels=['Very Sparse', 'Sparse', 'Medium', 'Dense', 'Very Dense']
        )
        
        df['radar_point_category'] = pd.cut(
            df['num_radar_pts'],
            bins=[0, 1, 5, 20, 50, np.inf],
            labels=['None', 'Few', 'Some', 'Many', 'Abundant']
        )
        
        # Detection quality score (combination of distance and point count)
        df['detection_quality_score'] = (
            np.log1p(df['num_lidar_pts']) * 
            np.exp(-df['distance_from_ego'] / 50)  # Decay with distance
        )
        
        return df
    
    def engineer_temporal_features(self, sensor_df: pd.DataFrame) -> pd.DataFrame:
        """
        Engineer temporal features from sensor data
        
        Args:
            sensor_df: DataFrame containing sensor data
            
        Returns:
            DataFrame with temporal features
        """
        df = sensor_df.copy()
        
        if 'timestamp' in df.columns:
            # Convert timestamp to datetime
            df['timestamp_dt'] = pd.to_datetime(df['timestamp'], unit='us')
            
            # Extract time components
            df['hour'] = df['timestamp_dt'].dt.hour
            df['day_of_week'] = df['timestamp_dt'].dt.dayofweek
            df['month'] = df['timestamp_dt'].dt.month
            
            # Time of day categories
            df['time_of_day'] = pd.cut(
                df['hour'],
                bins=[0, 6, 12, 18, 24],
                labels=['Night', 'Morning', 'Afternoon', 'Evening'],
                right=False
            )
            
            # Calculate time differences within groups
            df_sorted = df.sort_values(['sensor_channel', 'timestamp'])
            df_sorted['time_diff'] = df_sorted.groupby('sensor_channel')['timestamp'].diff()
            df_sorted['time_diff_seconds'] = df_sorted['time_diff'] / 1e6  # Convert to seconds
            
            return df_sorted
        
        return df
    
    def engineer_scene_features(self, scenes_df: pd.DataFrame) -> pd.DataFrame:
        """
        Engineer scene-level features
        
        Args:
            scenes_df: DataFrame containing scene metadata
            
        Returns:
            DataFrame with scene features
        """
        df = scenes_df.copy()
        
        # Scene duration categories
        if 'nbr_samples' in df.columns:
            df['scene_length_category'] = pd.cut(
                df['nbr_samples'],
                bins=[0, 20, 40, 60, 80, np.inf],
                labels=['Very Short', 'Short', 'Medium', 'Long', 'Very Long']
            )
        
        # Location encoding
        if 'log_location' in df.columns:
            le = LabelEncoder()
            df['location_encoded'] = le.fit_transform(df['log_location'])
            self.label_encoders['location'] = le
        
        # Time features from date
        if 'log_date_captured' in df.columns:
            df['log_date_dt'] = pd.to_datetime(df['log_date_captured'])
            df['log_year'] = df['log_date_dt'].dt.year
            df['log_month'] = df['log_date_dt'].dt.month
            df['log_day'] = df['log_date_dt'].dt.day
            df['log_weekday'] = df['log_date_dt'].dt.dayofweek
            
            # Season classification
            df['season'] = df['log_month'].apply(self._classify_season)
        
        return df
    
    def engineer_interaction_features(self, annotations_df: pd.DataFrame) -> pd.DataFrame:
        """
        Engineer features based on object interactions within samples
        
        Args:
            annotations_df: DataFrame containing annotations
            
        Returns:
            DataFrame with interaction features
        """
        df = annotations_df.copy()
        
        # Group by sample to analyze object interactions
        sample_groups = df.groupby('sample_token')
        
        interaction_features = []
        
        for sample_token, group in sample_groups:
            group_features = group.copy()
            
            # Number of objects in the same sample
            group_features['objects_in_sample'] = len(group)
            
            # Category diversity in sample
            unique_categories = group['category_name'].nunique()
            group_features['category_diversity'] = unique_categories
            
            # Distance to nearest object of same category
            for idx, row in group_features.iterrows():
                same_category = group[group['category_name'] == row['category_name']]
                if len(same_category) > 1:
                    # Calculate distances to other objects of same category
                    current_pos = np.array([[row['translation_x'], row['translation_y'], row['translation_z']]])
                    other_positions = same_category[same_category.index != idx][
                        ['translation_x', 'translation_y', 'translation_z']
                    ].values
                    
                    if len(other_positions) > 0:
                        distances = cdist(current_pos, other_positions)[0]
                        group_features.loc[idx, 'nearest_same_category_distance'] = distances.min()
                    else:
                        group_features.loc[idx, 'nearest_same_category_distance'] = np.inf
                else:
                    group_features.loc[idx, 'nearest_same_category_distance'] = np.inf
            
            interaction_features.append(group_features)
        
        return pd.concat(interaction_features, ignore_index=True)
    
    def create_aggregated_features(self, annotations_df: pd.DataFrame, 
                                 group_by: str = 'category_name') -> pd.DataFrame:
        """
        Create aggregated features by grouping
        
        Args:
            annotations_df: DataFrame containing annotations
            group_by: Column to group by
            
        Returns:
            DataFrame with aggregated features
        """
        if group_by not in annotations_df.columns:
            return pd.DataFrame()
        
        # Numerical columns to aggregate
        numerical_cols = annotations_df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Define aggregation functions
        agg_functions = {
            'count': 'count',
            'mean': 'mean',
            'std': 'std',
            'min': 'min',
            'max': 'max',
            'median': 'median',
            'q75': lambda x: x.quantile(0.75),
            'q25': lambda x: x.quantile(0.25)
        }
        
        aggregated_features = {}
        
        for col in numerical_cols:
            if col != group_by:
                for agg_name, agg_func in agg_functions.items():
                    agg_col_name = f"{col}_{agg_name}"
                    aggregated_features[agg_col_name] = annotations_df.groupby(group_by)[col].agg(agg_func)
        
        return pd.DataFrame(aggregated_features).reset_index()
    
    def normalize_features(self, df: pd.DataFrame, 
                          columns: List[str] = None) -> pd.DataFrame:
        """
        Normalize numerical features
        
        Args:
            df: Input DataFrame
            columns: Specific columns to normalize (if None, normalize all numerical)
            
        Returns:
            DataFrame with normalized features
        """
        df_normalized = df.copy()
        
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        for col in columns:
            if col in df.columns:
                scaler = StandardScaler()
                df_normalized[f"{col}_normalized"] = scaler.fit_transform(df[[col]])
                self.scalers[col] = scaler
        
        return df_normalized
    
    def _classify_season(self, month: int) -> str:
        """
        Classify month into season
        
        Args:
            month: Month number (1-12)
            
        Returns:
            Season name
        """
        if month in [12, 1, 2]:
            return 'Winter'
        elif month in [3, 4, 5]:
            return 'Spring'
        elif month in [6, 7, 8]:
            return 'Summer'
        elif month in [9, 10, 11]:
            return 'Fall'
        else:
            return 'Unknown'


class FeatureEngineer:
    """
    General feature engineer with basic functionality
    """
    
    def __init__(self):
        self.encoders = {}
        self.scalers = {}
    
    def create_polynomial_features(self, df: pd.DataFrame, 
                                  columns: List[str], 
                                  degree: int = 2) -> pd.DataFrame:
        """
        Create polynomial features
        
        Args:
            df: Input DataFrame
            columns: Columns to create polynomial features for
            degree: Polynomial degree
            
        Returns:
            DataFrame with polynomial features
        """
        df_poly = df.copy()
        
        for col in columns:
            if col in df.columns:
                for d in range(2, degree + 1):
                    df_poly[f"{col}_poly_{d}"] = df[col] ** d
        
        return df_poly
    
    def create_interaction_features(self, df: pd.DataFrame, 
                                   columns: List[str]) -> pd.DataFrame:
        """
        Create interaction features between columns
        
        Args:
            df: Input DataFrame
            columns: Columns to create interactions for
            
        Returns:
            DataFrame with interaction features
        """
        df_interact = df.copy()
        
        for i in range(len(columns)):
            for j in range(i + 1, len(columns)):
                col1, col2 = columns[i], columns[j]
                if col1 in df.columns and col2 in df.columns:
                    df_interact[f"{col1}_{col2}_interact"] = df[col1] * df[col2]
        
        return df_interact
    
    def encode_categorical_features(self, df: pd.DataFrame, 
                                   columns: List[str] = None) -> pd.DataFrame:
        """
        Encode categorical features
        
        Args:
            df: Input DataFrame
            columns: Columns to encode (if None, encode all categorical)
            
        Returns:
            DataFrame with encoded features
        """
        df_encoded = df.copy()
        
        if columns is None:
            columns = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        for col in columns:
            if col in df.columns:
                le = LabelEncoder()
                df_encoded[f"{col}_encoded"] = le.fit_transform(df[col].astype(str))
                self.encoders[col] = le
        
        return df_encoded