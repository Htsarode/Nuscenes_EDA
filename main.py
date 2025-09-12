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
    load_pedestrian_visibility_status
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
