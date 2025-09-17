"""
Data Cleaning for nuScenes EDA
This module provides comprehensive data cleaning functionality for nuScenes dataset
tailored for the 22 analysis types in this EDA system.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from nuscenes.nuscenes import NuScenes
import warnings
warnings.filterwarnings('ignore')


class NuScenesDataCleaner:
    """
    Data cleaner specifically designed for nuScenes dataset
    """
    
    def __init__(self, dataroot: str = None, version: str = "v1.0-mini"):
        """
        Initialize data cleaner with nuScenes dataset access
        
        Args:
            dataroot: Path to nuScenes dataset
            version: Dataset version
        """
        self.cleaning_report = {}
        self.dataroot = dataroot
        self.version = version
        self.nusc = None
        if dataroot:
            try:
                self.nusc = NuScenes(version=version, dataroot=dataroot, verbose=False)
            except Exception as e:
                print(f"Warning: Could not initialize nuScenes: {e}")
    
    def validate_nuscenes_integrity(self) -> Dict[str, Any]:
        """
        Validate nuScenes dataset integrity for EDA analyses
        
        Returns:
            Dictionary containing validation results
        """
        if not self.nusc:
            return {"error": "nuScenes not initialized"}
        
        validation = {
            "scenes": len(self.nusc.scene),
            "samples": len(self.nusc.sample),
            "sample_annotations": len(self.nusc.sample_annotation),
            "sample_data": len(self.nusc.sample_data),
            "sensors": len(self.nusc.sensor),
            "calibrated_sensors": len(self.nusc.calibrated_sensor),
            "ego_poses": len(self.nusc.ego_pose),
            "categories": len(self.nusc.category),
            "attributes": len(self.nusc.attribute),
            "visibility": len(self.nusc.visibility),
            "instances": len(self.nusc.instance),
            "maps": len(self.nusc.map),
            "logs": len(self.nusc.log)
        }
        
        # Check for missing essential data
        missing_data = []
        if validation["scenes"] == 0:
            missing_data.append("scenes")
        if validation["sample_annotations"] == 0:
            missing_data.append("sample_annotations")
        if validation["categories"] == 0:
            missing_data.append("categories")
            
        validation["missing_essential"] = missing_data
        validation["integrity_score"] = (13 - len(missing_data)) / 13 * 100
        
        return validation
    
    def clean_pedestrian_data(self) -> Dict[str, Any]:
        """
        Clean and validate pedestrian-related data for analyses 1-6
        
        Returns:
            Cleaned pedestrian data summary
        """
        if not self.nusc:
            return {}
        
        pedestrian_categories = [
            'human.pedestrian.adult',
            'human.pedestrian.child',
            'human.pedestrian.construction_worker',
            'human.pedestrian.police_officer'
        ]
        
        pedestrian_data = []
        invalid_entries = 0
        
        for ann in self.nusc.sample_annotation:
            if ann['category_name'] in pedestrian_categories:
                # Validate essential fields
                if (ann['translation'] and len(ann['translation']) == 3 and
                    ann['size'] and len(ann['size']) == 3 and
                    ann['rotation'] and len(ann['rotation']) == 4):
                    
                    pedestrian_data.append({
                        'token': ann['token'],
                        'category': ann['category_name'],
                        'translation': ann['translation'],
                        'size': ann['size'],
                        'rotation': ann['rotation'],
                        'visibility_token': ann['visibility_token'],
                        'attribute_tokens': ann['attribute_tokens'],
                        'num_lidar_pts': ann['num_lidar_pts'],
                        'num_radar_pts': ann['num_radar_pts']
                    })
                else:
                    invalid_entries += 1
        
        return {
            'valid_pedestrians': len(pedestrian_data),
            'invalid_entries': invalid_entries,
            'categories_found': list(set([p['category'] for p in pedestrian_data])),
            'data_quality': len(pedestrian_data) / (len(pedestrian_data) + invalid_entries) * 100 if pedestrian_data else 0
        }
    
    def clean_vehicle_data(self) -> Dict[str, Any]:
        """
        Clean and validate vehicle-related data for analyses 7-9
        
        Returns:
            Cleaned vehicle data summary
        """
        if not self.nusc:
            return {}
        
        vehicle_categories = [
            'vehicle.car', 'vehicle.truck', 'vehicle.bus.bendy', 'vehicle.bus.rigid',
            'vehicle.motorcycle', 'vehicle.bicycle', 'vehicle.emergency.ambulance',
            'vehicle.emergency.police', 'vehicle.construction', 'vehicle.trailer'
        ]
        
        vehicle_data = []
        invalid_entries = 0
        
        for ann in self.nusc.sample_annotation:
            if ann['category_name'] in vehicle_categories:
                # Validate essential fields
                if (ann['translation'] and len(ann['translation']) == 3 and
                    ann['size'] and len(ann['size']) == 3):
                    
                    vehicle_data.append({
                        'token': ann['token'],
                        'category': ann['category_name'],
                        'translation': ann['translation'],
                        'size': ann['size'],
                        'rotation': ann['rotation'],
                        'num_lidar_pts': ann['num_lidar_pts'],
                        'num_radar_pts': ann['num_radar_pts']
                    })
                else:
                    invalid_entries += 1
        
        return {
            'valid_vehicles': len(vehicle_data),
            'invalid_entries': invalid_entries,
            'categories_found': list(set([v['category'] for v in vehicle_data])),
            'data_quality': len(vehicle_data) / (len(vehicle_data) + invalid_entries) * 100 if vehicle_data else 0
        }
    
    def clean_environmental_data(self) -> Dict[str, Any]:
        """
        Clean and validate environmental data for analyses 10-13
        
        Returns:
            Cleaned environmental data summary
        """
        if not self.nusc:
            return {}
        
        # Scene-level environmental data
        scene_data = []
        log_data = {}
        
        for scene in self.nusc.scene:
            if scene['name'] and scene['log_token']:
                log = self.nusc.get('log', scene['log_token'])
                scene_data.append({
                    'scene_token': scene['token'],
                    'scene_name': scene['name'],
                    'location': log.get('location', 'Unknown'),
                    'date_captured': log.get('date_captured', None),
                    'vehicle': log.get('vehicle', 'Unknown')
                })
                
                if scene['log_token'] not in log_data:
                    log_data[scene['log_token']] = log
        
        # Weather and time analysis from scene names
        weather_conditions = {'clear': 0, 'rain': 0, 'night': 0, 'unknown': 0}
        time_conditions = {'day': 0, 'night': 0, 'unknown': 0}
        
        for scene in scene_data:
            name = scene['scene_name'].lower()
            
            # Weather detection
            if 'rain' in name:
                weather_conditions['rain'] += 1
            elif 'night' in name:
                weather_conditions['night'] += 1
            elif any(term in name for term in ['clear', 'sunny', 'day']):
                weather_conditions['clear'] += 1
            else:
                weather_conditions['unknown'] += 1
            
            # Time detection  
            if 'night' in name:
                time_conditions['night'] += 1
            elif any(term in name for term in ['day', 'morning', 'afternoon']):
                time_conditions['day'] += 1
            else:
                time_conditions['unknown'] += 1
        
        return {
            'valid_scenes': len(scene_data),
            'unique_locations': len(set([s['location'] for s in scene_data])),
            'weather_distribution': weather_conditions,
            'time_distribution': time_conditions,
            'date_range': {
                'earliest': min([s['date_captured'] for s in scene_data if s['date_captured']]) if scene_data else None,
                'latest': max([s['date_captured'] for s in scene_data if s['date_captured']]) if scene_data else None
            }
        }
    
    def clean_road_infrastructure_data(self) -> Dict[str, Any]:
        """
        Clean and validate road infrastructure data for analyses 14-17
        
        Returns:
            Cleaned road infrastructure summary
        """
        if not self.nusc:
            return {}
        
        # Map-related data
        map_data = []
        for map_record in self.nusc.map:
            if map_record['filename'] and map_record['category']:
                map_data.append({
                    'token': map_record['token'],
                    'category': map_record['category'],
                    'filename': map_record['filename']
                })
        
        # Static object categories for road infrastructure
        infrastructure_categories = [
            'static_object.bicycle_rack',
            'movable_object.barrier',
            'movable_object.debris',
            'movable_object.pushable_pullable',
            'movable_object.trafficcone',
            'static_object.traffic_light'
        ]
        
        infrastructure_data = []
        for ann in self.nusc.sample_annotation:
            if ann['category_name'] in infrastructure_categories:
                if ann['translation'] and ann['size']:
                    infrastructure_data.append({
                        'category': ann['category_name'],
                        'translation': ann['translation'],
                        'size': ann['size']
                    })
        
        return {
            'valid_maps': len(map_data),
            'map_categories': list(set([m['category'] for m in map_data])),
            'infrastructure_objects': len(infrastructure_data),
            'infrastructure_categories': list(set([i['category'] for i in infrastructure_data]))
        }
    
    def clean_ego_vehicle_data(self) -> Dict[str, Any]:
        """
        Clean and validate ego vehicle data for analyses 18-20
        
        Returns:
            Cleaned ego vehicle data summary
        """
        if not self.nusc:
            return {}
        
        ego_poses = []
        invalid_poses = 0
        
        for pose in self.nusc.ego_pose:
            if (pose['translation'] and len(pose['translation']) == 3 and
                pose['rotation'] and len(pose['rotation']) == 4):
                ego_poses.append({
                    'token': pose['token'],
                    'translation': pose['translation'],
                    'rotation': pose['rotation'],
                    'timestamp': pose['timestamp']
                })
            else:
                invalid_poses += 1
        
        # Analyze ego vehicle motion patterns
        if len(ego_poses) > 1:
            speeds = []
            for i in range(1, len(ego_poses)):
                prev_pose = ego_poses[i-1]
                curr_pose = ego_poses[i]
                
                # Calculate displacement
                dx = curr_pose['translation'][0] - prev_pose['translation'][0]
                dy = curr_pose['translation'][1] - prev_pose['translation'][1]
                distance = np.sqrt(dx**2 + dy**2)
                
                # Calculate time difference (timestamps in microseconds)
                dt = (curr_pose['timestamp'] - prev_pose['timestamp']) / 1e6
                
                if dt > 0:
                    speed = distance / dt
                    speeds.append(speed)
            
            speed_stats = {
                'mean_speed': np.mean(speeds) if speeds else 0,
                'max_speed': np.max(speeds) if speeds else 0,
                'min_speed': np.min(speeds) if speeds else 0,
                'std_speed': np.std(speeds) if speeds else 0
            }
        else:
            speed_stats = {'mean_speed': 0, 'max_speed': 0, 'min_speed': 0, 'std_speed': 0}
        
        return {
            'valid_ego_poses': len(ego_poses),
            'invalid_poses': invalid_poses,
            'data_quality': len(ego_poses) / (len(ego_poses) + invalid_poses) * 100 if ego_poses else 0,
            'speed_statistics': speed_stats,
            'temporal_span': {
                'start': min([p['timestamp'] for p in ego_poses]) if ego_poses else 0,
                'end': max([p['timestamp'] for p in ego_poses]) if ego_poses else 0
            }
        }
    
    def clean_annotations(self, annotations_df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean annotation data
        
        Args:
            annotations_df: DataFrame containing annotations
            
        Returns:
            Cleaned DataFrame
        """
        df = annotations_df.copy()
        initial_shape = df.shape
        
        cleaning_steps = []
        
        # Remove annotations with invalid coordinates
        invalid_coords = (
            (df['translation_x'].isna()) |
            (df['translation_y'].isna()) |
            (df['translation_z'].isna())
        )
        invalid_count = invalid_coords.sum()
        if invalid_count > 0:
            df = df[~invalid_coords]
            cleaning_steps.append(f"Removed {invalid_count} annotations with invalid coordinates")
        
        # Remove annotations with invalid sizes
        invalid_sizes = (
            (df['size_width'] <= 0) |
            (df['size_length'] <= 0) |
            (df['size_height'] <= 0) |
            df['size_width'].isna() |
            df['size_length'].isna() |
            df['size_height'].isna()
        )
        invalid_size_count = invalid_sizes.sum()
        if invalid_size_count > 0:
            df = df[~invalid_sizes]
            cleaning_steps.append(f"Removed {invalid_size_count} annotations with invalid sizes")
        
        # Handle outliers in point counts
        # Remove annotations with extremely high point counts (likely errors)
        lidar_outliers = df['num_lidar_pts'] > df['num_lidar_pts'].quantile(0.999)
        radar_outliers = df['num_radar_pts'] > df['num_radar_pts'].quantile(0.999)
        
        outlier_count = (lidar_outliers | radar_outliers).sum()
        if outlier_count > 0:
            df = df[~(lidar_outliers | radar_outliers)]
            cleaning_steps.append(f"Removed {outlier_count} annotations with outlier point counts")
        
        # Handle extreme distances
        df['distance_from_ego'] = np.sqrt(
            df['translation_x']**2 + 
            df['translation_y']**2 + 
            df['translation_z']**2
        )
        
        # Remove objects beyond reasonable sensor range (e.g., >200m)
        distance_outliers = df['distance_from_ego'] > 200
        distance_outlier_count = distance_outliers.sum()
        if distance_outlier_count > 0:
            df = df[~distance_outliers]
            cleaning_steps.append(f"Removed {distance_outlier_count} annotations beyond 200m range")
        
        # Remove the temporary distance column
        df = df.drop('distance_from_ego', axis=1)
        
        # Handle missing category names
        missing_category = df['category_name'].isna()
        missing_category_count = missing_category.sum()
        if missing_category_count > 0:
            df = df[~missing_category]
            cleaning_steps.append(f"Removed {missing_category_count} annotations with missing category")
        
        final_shape = df.shape
        
        self.cleaning_report['annotations'] = {
            'initial_shape': initial_shape,
            'final_shape': final_shape,
            'removed_rows': initial_shape[0] - final_shape[0],
            'cleaning_steps': cleaning_steps
        }
        
        return df
    
    def clean_sensor_data(self, sensor_df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean sensor data
        
        Args:
            sensor_df: DataFrame containing sensor data
            
        Returns:
            Cleaned DataFrame
        """
        df = sensor_df.copy()
        initial_shape = df.shape
        
        cleaning_steps = []
        
        # Remove entries with missing essential information
        missing_essential = (
            df['sensor_channel'].isna() |
            df['sensor_modality'].isna() |
            df['filename'].isna()
        )
        missing_count = missing_essential.sum()
        if missing_count > 0:
            df = df[~missing_essential]
            cleaning_steps.append(f"Removed {missing_count} entries with missing essential sensor info")
        
        # Handle invalid timestamps
        if 'timestamp' in df.columns:
            invalid_timestamps = (
                (df['timestamp'] <= 0) |
                df['timestamp'].isna()
            )
            invalid_ts_count = invalid_timestamps.sum()
            if invalid_ts_count > 0:
                df = df[~invalid_timestamps]
                cleaning_steps.append(f"Removed {invalid_ts_count} entries with invalid timestamps")
        
        # Handle invalid image dimensions for camera data
        camera_data = df[df['sensor_modality'] == 'camera']
        if not camera_data.empty:
            invalid_dimensions = (
                (camera_data['width'] <= 0) |
                (camera_data['height'] <= 0) |
                camera_data['width'].isna() |
                camera_data['height'].isna()
            )
            invalid_dim_count = invalid_dimensions.sum()
            if invalid_dim_count > 0:
                # Remove invalid camera entries
                invalid_indices = camera_data[invalid_dimensions].index
                df = df.drop(invalid_indices)
                cleaning_steps.append(f"Removed {invalid_dim_count} camera entries with invalid dimensions")
        
        # Clean ego pose data
        ego_pose_columns = ['ego_translation_x', 'ego_translation_y', 'ego_translation_z',
                           'ego_rotation_w', 'ego_rotation_x', 'ego_rotation_y', 'ego_rotation_z']
        
        available_ego_cols = [col for col in ego_pose_columns if col in df.columns]
        if available_ego_cols:
            invalid_ego_pose = df[available_ego_cols].isna().any(axis=1)
            invalid_ego_count = invalid_ego_pose.sum()
            if invalid_ego_count > 0:
                df = df[~invalid_ego_pose]
                cleaning_steps.append(f"Removed {invalid_ego_count} entries with invalid ego pose")
        
        final_shape = df.shape
        
        self.cleaning_report['sensor_data'] = {
            'initial_shape': initial_shape,
            'final_shape': final_shape,
            'removed_rows': initial_shape[0] - final_shape[0],
            'cleaning_steps': cleaning_steps
        }
        
        return df
    
    def clean_scenes(self, scenes_df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean scene metadata
        
        Args:
            scenes_df: DataFrame containing scene metadata
            
        Returns:
            Cleaned DataFrame
        """
        df = scenes_df.copy()
        initial_shape = df.shape
        
        cleaning_steps = []
        
        # Remove scenes with missing essential information
        missing_essential = (
            df['token'].isna() |
            df['name'].isna() |
            df['log_token'].isna()
        )
        missing_count = missing_essential.sum()
        if missing_count > 0:
            df = df[~missing_essential]
            cleaning_steps.append(f"Removed {missing_count} scenes with missing essential info")
        
        # Handle invalid sample counts
        if 'nbr_samples' in df.columns:
            invalid_samples = (
                (df['nbr_samples'] <= 0) |
                df['nbr_samples'].isna()
            )
            invalid_sample_count = invalid_samples.sum()
            if invalid_sample_count > 0:
                df = df[~invalid_samples]
                cleaning_steps.append(f"Removed {invalid_sample_count} scenes with invalid sample counts")
        
        # Clean date information
        if 'log_date_captured' in df.columns:
            # Convert to datetime and handle invalid dates
            df['log_date_captured'] = pd.to_datetime(df['log_date_captured'], errors='coerce')
            invalid_dates = df['log_date_captured'].isna()
            invalid_date_count = invalid_dates.sum()
            if invalid_date_count > 0:
                df = df[~invalid_dates]
                cleaning_steps.append(f"Removed {invalid_date_count} scenes with invalid dates")
        
        final_shape = df.shape
        
        self.cleaning_report['scenes'] = {
            'initial_shape': initial_shape,
            'final_shape': final_shape,
            'removed_rows': initial_shape[0] - final_shape[0],
            'cleaning_steps': cleaning_steps
        }
        
        return df
    
    def handle_duplicates(self, df: pd.DataFrame, 
                         subset_columns: List[str] = None) -> pd.DataFrame:
        """
        Handle duplicate records
        
        Args:
            df: Input DataFrame
            subset_columns: Columns to check for duplicates
            
        Returns:
            DataFrame without duplicates
        """
        initial_count = len(df)
        
        if subset_columns:
            df_clean = df.drop_duplicates(subset=subset_columns, keep='first')
        else:
            df_clean = df.drop_duplicates(keep='first')
        
        duplicates_removed = initial_count - len(df_clean)
        
        if duplicates_removed > 0:
            print(f"Removed {duplicates_removed} duplicate records")
        
        return df_clean
    
    def validate_data_consistency(self, annotations_df: pd.DataFrame,
                                sensor_df: pd.DataFrame,
                                scenes_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate consistency across different data tables
        
        Args:
            annotations_df: DataFrame containing annotations
            sensor_df: DataFrame containing sensor data
            scenes_df: DataFrame containing scenes
            
        Returns:
            Dictionary containing validation results
        """
        validation_report = {}
        
        # Check if sample tokens in annotations exist in sensor data
        if not annotations_df.empty and not sensor_df.empty:
            ann_samples = set(annotations_df['sample_token'].unique())
            sensor_samples = set(sensor_df['sample_token'].unique())
            
            missing_in_sensor = ann_samples - sensor_samples
            missing_in_annotations = sensor_samples - ann_samples
            
            validation_report['sample_token_consistency'] = {
                'annotations_unique_samples': len(ann_samples),
                'sensor_unique_samples': len(sensor_samples),
                'missing_in_sensor_data': len(missing_in_sensor),
                'missing_in_annotations': len(missing_in_annotations)
            }
        
        # Check scene consistency
        if not scenes_df.empty:
            if not sensor_df.empty:
                # Check if scenes referenced in sensor data exist
                scene_tokens_in_sensor = set(sensor_df.get('scene_token', pd.Series()).dropna().unique())
                scene_tokens_available = set(scenes_df['token'].unique())
                missing_scenes = scene_tokens_in_sensor - scene_tokens_available
                
                validation_report['scene_consistency'] = {
                    'referenced_scenes': len(scene_tokens_in_sensor),
                    'available_scenes': len(scene_tokens_available),
                    'missing_scenes': len(missing_scenes)
                }
        
        return validation_report
    
    def clean_multimodal_synchronization_data(self) -> Dict[str, Any]:
        """
        Clean and validate multimodal sensor synchronization data for analysis 16
        
        Returns:
            Sensor synchronization data quality report
        """
        if not self.nusc:
            return {}
        
        sensor_data = {}
        timestamp_mismatches = 0
        total_samples = 0
        
        for sample in self.nusc.sample:
            total_samples += 1
            sample_sensors = {}
            
            # Get all sensor data for this sample
            for key, token in sample['data'].items():
                if token:
                    sample_data = self.nusc.get('sample_data', token)
                    sample_sensors[key] = {
                        'timestamp': sample_data['timestamp'],
                        'is_key_frame': sample_data['is_key_frame'],
                        'filename': sample_data['filename']
                    }
            
            # Check timestamp synchronization
            timestamps = [data['timestamp'] for data in sample_sensors.values()]
            if timestamps:
                timestamp_diff = max(timestamps) - min(timestamps)
                if timestamp_diff > 50000:  # 50ms threshold
                    timestamp_mismatches += 1
            
            sensor_data[sample['token']] = sample_sensors
        
        # Analyze sensor coverage
        sensor_coverage = {}
        for sample_token, sensors in sensor_data.items():
            for sensor_name in sensors.keys():
                if sensor_name not in sensor_coverage:
                    sensor_coverage[sensor_name] = 0
                sensor_coverage[sensor_name] += 1
        
        return {
            'total_samples': total_samples,
            'timestamp_mismatches': timestamp_mismatches,
            'synchronization_quality': (total_samples - timestamp_mismatches) / total_samples * 100 if total_samples > 0 else 0,
            'sensor_coverage': sensor_coverage,
            'sensors_per_sample': len(sensor_coverage)
        }
    
    def clean_rare_class_data(self) -> Dict[str, Any]:
        """
        Clean and validate rare class occurrence data for analysis 22
        
        Returns:
            Rare class analysis results
        """
        if not self.nusc:
            return {}
        
        category_counts = {}
        total_annotations = 0
        
        for ann in self.nusc.sample_annotation:
            total_annotations += 1
            category = ann['category_name']
            if category not in category_counts:
                category_counts[category] = 0
            category_counts[category] += 1
        
        # Define rare classes (less than 1% of total annotations)
        rare_threshold = total_annotations * 0.01
        rare_classes = {cat: count for cat, count in category_counts.items() if count < rare_threshold}
        common_classes = {cat: count for cat, count in category_counts.items() if count >= rare_threshold}
        
        return {
            'total_annotations': total_annotations,
            'total_categories': len(category_counts),
            'rare_classes': rare_classes,
            'common_classes': common_classes,
            'rare_class_count': len(rare_classes),
            'rare_class_percentage': len(rare_classes) / len(category_counts) * 100 if category_counts else 0
        }
    
    def get_comprehensive_cleaning_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive cleaning report for all analysis types
        
        Returns:
            Complete cleaning and validation report
        """
        report = {
            'dataset_integrity': self.validate_nuscenes_integrity(),
            'pedestrian_data_quality': self.clean_pedestrian_data(),
            'vehicle_data_quality': self.clean_vehicle_data(),
            'environmental_data_quality': self.clean_environmental_data(),
            'road_infrastructure_quality': self.clean_road_infrastructure_data(),
            'ego_vehicle_data_quality': self.clean_ego_vehicle_data(),
            'multimodal_sync_quality': self.clean_multimodal_synchronization_data(),
            'rare_class_analysis': self.clean_rare_class_data()
        }
        
        # Overall quality score
        quality_scores = []
        for analysis_type, data in report.items():
            if isinstance(data, dict) and 'data_quality' in data:
                quality_scores.append(data['data_quality'])
            elif isinstance(data, dict) and 'integrity_score' in data:
                quality_scores.append(data['integrity_score'])
            elif isinstance(data, dict) and 'synchronization_quality' in data:
                quality_scores.append(data['synchronization_quality'])
        
        report['overall_quality_score'] = np.mean(quality_scores) if quality_scores else 0
        report['analysis_readiness'] = report['overall_quality_score'] > 80
        
        return report
    
    def get_cleaning_report(self) -> Dict[str, Any]:
        """
        Get comprehensive cleaning report
        
        Returns:
            Dictionary containing cleaning report
        """
        return self.cleaning_report
    
    def reset_report(self):
        """Reset the cleaning report"""
        self.cleaning_report = {}


class DataCleaner:
    """
    General data cleaner with basic functionality
    """
    
    def __init__(self):
        pass
    
    def remove_missing_values(self, df: pd.DataFrame, 
                            threshold: float = 0.5) -> pd.DataFrame:
        """
        Remove columns with too many missing values
        
        Args:
            df: Input DataFrame
            threshold: Threshold for missing value ratio (0.0 to 1.0)
            
        Returns:
            DataFrame with columns removed
        """
        missing_ratio = df.isnull().sum() / len(df)
        columns_to_keep = missing_ratio[missing_ratio <= threshold].index
        
        removed_columns = set(df.columns) - set(columns_to_keep)
        if removed_columns:
            print(f"Removed columns with >{threshold*100}% missing values: {removed_columns}")
        
        return df[columns_to_keep]
    
    def fill_missing_values(self, df: pd.DataFrame, 
                          strategy: str = 'mean') -> pd.DataFrame:
        """
        Fill missing values using specified strategy
        
        Args:
            df: Input DataFrame
            strategy: Strategy to fill missing values ('mean', 'median', 'mode', 'ffill', 'bfill')
            
        Returns:
            DataFrame with filled missing values
        """
        df_filled = df.copy()
        
        numerical_cols = df_filled.select_dtypes(include=[np.number]).columns
        categorical_cols = df_filled.select_dtypes(include=['object', 'category']).columns
        
        if strategy == 'mean':
            df_filled[numerical_cols] = df_filled[numerical_cols].fillna(df_filled[numerical_cols].mean())
        elif strategy == 'median':
            df_filled[numerical_cols] = df_filled[numerical_cols].fillna(df_filled[numerical_cols].median())
        elif strategy == 'mode':
            for col in numerical_cols:
                df_filled[col] = df_filled[col].fillna(df_filled[col].mode()[0] if not df_filled[col].mode().empty else 0)
        elif strategy in ['ffill', 'bfill']:
            df_filled = df_filled.fillna(method=strategy)
        
        # For categorical columns, use mode
        for col in categorical_cols:
            if not df_filled[col].mode().empty:
                df_filled[col] = df_filled[col].fillna(df_filled[col].mode()[0])
        
        return df_filled
    
    def remove_outliers(self, df: pd.DataFrame, 
                       columns: List[str] = None,
                       method: str = 'iqr',
                       factor: float = 1.5) -> pd.DataFrame:
        """
        Remove outliers from specified columns
        
        Args:
            df: Input DataFrame
            columns: Columns to check for outliers (if None, check all numerical)
            method: Method to detect outliers ('iqr' or 'zscore')
            factor: Factor for outlier detection
            
        Returns:
            DataFrame without outliers
        """
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        df_clean = df.copy()
        
        for col in columns:
            if col in df_clean.columns:
                if method == 'iqr':
                    Q1 = df_clean[col].quantile(0.25)
                    Q3 = df_clean[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - factor * IQR
                    upper_bound = Q3 + factor * IQR
                    
                    outliers = (df_clean[col] < lower_bound) | (df_clean[col] > upper_bound)
                    df_clean = df_clean[~outliers]
                    
                elif method == 'zscore':
                    z_scores = np.abs((df_clean[col] - df_clean[col].mean()) / df_clean[col].std())
                    outliers = z_scores > factor
                    df_clean = df_clean[~outliers]
        
        removed_count = len(df) - len(df_clean)
        if removed_count > 0:
            print(f"Removed {removed_count} outlier records")
        
        return df_clean
    
    def standardize_column_names(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Standardize column names (lowercase, replace spaces with underscores)
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with standardized column names
        """
        df_clean = df.copy()
        df_clean.columns = df_clean.columns.str.lower().str.replace(' ', '_').str.replace('-', '_')
        return df_clean