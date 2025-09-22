from src.data_loader import (
    load_weather_conditions,
    load_road_details,
    load_road_type_distribution,
    load_road_obstacles,
    load_environment_distribution,
    load_time_of_day_distribution,
    load_geographical_locations,
    load_rare_class_occurrences,
    load_vehicle_class_data,
    load_object_behaviour_data,
    load_pedestrian_density_road_types,
    load_pedestrian_cyclist_ratio,
    load_pedestrian_behaviour_data,
    load_pedestrian_road_crossing,
    load_pedestrian_visibility_status,
    load_multimodal_synchronization_data,
    load_road_furniture_data,
    load_traffic_density_weather_data,
    load_ego_vehicle_motion_data,
    load_ego_vehicle_events_data,
    load_vehicle_position_ego_data,
    load_pedestrian_path_ego_data,
    load_traffic_density_data,
    load_drivable_area_percentage_data,
)

from plots.Weather import plot_weather_distribution
from plots.RoadCurvature import plot_road_details_distribution
from plots.RoadType import plot_road_type_distribution
from plots.RoadObstacles import plot_road_obstacles_distribution
from plots.EnvironmentDistribution import plot_environment_distribution
from plots.TimeOfDay import plot_time_of_day_distribution
from plots.GeographicalLocations import plot_geographical_locations
from plots.RareClassOccurrences import plot_rare_class_occurrences
from plots.VehicleClass import plot_vehicle_class
from plots.ObjectBehaviour import plot_object_behaviour
from plots.PedestrianDensityRoadTypes import plot_pedestrian_density_road_types
from plots.PedestrianCyclistRatio import plot_pedestrian_cyclist_ratio
from plots.PedestrianBehaviour import plot_pedestrian_behaviour
from plots.PedestrianRoadCrossing import plot_pedestrian_road_crossing
from plots.PedestrianVisibilityStatus import plot_pedestrian_visibility_status
from plots.MultiModalSynchronization import plot_multimodal_synchronization
from plots.RoadFurniture import plot_road_furniture
from plots.TrafficDensityWeather import plot_traffic_density_weather
from plots.EgoVehicleMotion import plot_ego_vehicle_motion
from plots.EgoVehicleEvents import plot_ego_vehicle_events
from plots.VehiclePositionEgo import plot_vehicle_position_ego
from plots.PedestrianPathEgo import plot_pedestrian_path_ego
from plots.TrafficDensity import plot_traffic_density_across_frames
from plots.DrivableAreaPercentage import plot_drivable_area_percentage

def main():
    dataroot = "Data/Raw/nuscenes/v1.0-mini"
    version = "v1.0-mini"

    # --- Menu of analyses ---
    analysis_map = {
        "1": ("Weather Conditions", load_weather_conditions, plot_weather_distribution),
        "2": ("Road Details (Curvature)", load_road_details, plot_road_details_distribution),
        "3": ("Road Type Distribution", load_road_type_distribution, plot_road_type_distribution),
        "4": ("Road Obstacles", load_road_obstacles, plot_road_obstacles_distribution),
        "5": ("Environment Distribution", load_environment_distribution, plot_environment_distribution),
        "6": ("Time of Day Distribution", load_time_of_day_distribution, plot_time_of_day_distribution),
        "7": ("Geographical Locations", load_geographical_locations, plot_geographical_locations),
        "8": ("Rare Class Occurrences", load_rare_class_occurrences, plot_rare_class_occurrences),
        "9": ("Vehicle Class Distribution", load_vehicle_class_data, plot_vehicle_class),
        "10": ("Object Behaviour Distribution", load_object_behaviour_data, plot_object_behaviour),
        "11": ("Pedestrian Density across Road Types", load_pedestrian_density_road_types, plot_pedestrian_density_road_types),
        "12": ("Pedestrian/Cyclist Ratio", load_pedestrian_cyclist_ratio, plot_pedestrian_cyclist_ratio),
        "13": ("Pedestrian Behaviour", load_pedestrian_behaviour_data, plot_pedestrian_behaviour),
        "14": ("Pedestrian Road Crossing", load_pedestrian_road_crossing, plot_pedestrian_road_crossing),
        "15": ("Pedestrian Visibility Status", load_pedestrian_visibility_status, plot_pedestrian_visibility_status),
        "16": ("Multi-Modal Synchronization Analysis", load_multimodal_synchronization_data, plot_multimodal_synchronization),
        "17": ("Road Furniture Analysis", load_road_furniture_data, plot_road_furniture),
        "18": ("Traffic Density vs Weather Conditions", load_traffic_density_weather_data, plot_traffic_density_weather),
        "19": ("Ego Vehicle Motion Analysis", load_ego_vehicle_motion_data, plot_ego_vehicle_motion),
        "20": ("Ego Vehicle Events Analysis", load_ego_vehicle_events_data, plot_ego_vehicle_events),
        "21": ("Vehicle Position w.r.t. Ego Vehicle Analysis", load_vehicle_position_ego_data, plot_vehicle_position_ego),
        "22": ("Pedestrian Path w.r.t. Ego Vehicle Analysis", load_pedestrian_path_ego_data, plot_pedestrian_path_ego),
        "23": ("Traffic Density across Frames", load_traffic_density_data, plot_traffic_density_across_frames),
        "24": ("Drivable Area Percentage Analysis", load_drivable_area_percentage_data, plot_drivable_area_percentage),
    }

    print("\n📊 Available Analyses:")
    for num, (name, _, _) in analysis_map.items():
        print(f"{num}. {name}")

    selected = input("\nEnter the numbers of analyses you want to run (comma-separated): ").replace(" ", "").split(",")
    selected = [s for s in selected if s in analysis_map]

    if not selected:
        print("⚠️ No valid selection. Exiting.")
        return

    for s in selected:
        name, loader_func, plot_func = analysis_map[s]
        print(f"\n📊 Running {name}...")
        data = loader_func(dataroot, version)
        plot_func(data)

if __name__ == "__main__":
    main()
