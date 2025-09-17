import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import Dict
import pandas as pd
import os

def plot_vehicle_position_ego(data: Dict[str, int], title: str = "Vehicle Position w.r.t. Ego Vehicle Analysis", output_dir: str = "figures/exploratory"):
    """
    Plot vehicle position relative to ego vehicle analysis data with multiple chart options.
    
    Args:
        data: Dictionary containing position types and vehicle counts
        title: Title for the plot
        output_dir: Directory to save the plots
    """
    if not data:
        print("❌ No data available for plotting.")
        return
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Define all expected labels (will show all on x-axis even if count is 0)
    all_labels = ["Front", "Left", "Right", "Behind"]
    
    # Ensure all labels are present in data
    complete_data = {}
    for label in all_labels:
        complete_data[label] = data.get(label, 0)
    
    # Chart selection menu
    print("\n🎨 Select the type of chart to display:")
    print("1. Bar Chart")
    print("2. Pie Chart") 
    print("3. Donut Chart")
    print("4. Heat Map")
    print("5. Radar Chart")
    print("6. Histogram")
    print("7. Stacked Bar Chart")
    print("8. Scatter Plot")
    print("9. Density Plot")
    
    try:
        choice = int(input("Enter your choice (1-9): "))
    except ValueError:
        print("❌ Invalid input. Defaulting to Bar Chart.")
        choice = 1
    
    # Prepare data
    labels = list(complete_data.keys())
    values = list(complete_data.values())
    
    # Position-themed colors (representing spatial directions)
    position_colors = {
        "Front": '#FF6B6B',      # Red (Alert/Forward)
        "Left": '#4ECDC4',       # Teal (Left Side)
        "Right": '#45B7D1',      # Blue (Right Side)
        "Behind": '#96CEB4'      # Green (Behind/Safe)
    }
    colors = [position_colors[label] for label in labels]
    
    plt.figure(figsize=(14, 8))
    plt.style.use('seaborn-v0_8')
    
    if choice == 1:  # Bar Chart
        bars = plt.bar(labels, values, color=colors, alpha=0.8, edgecolor='black', linewidth=1.2)
        plt.xlabel("Relative Position", fontsize=14, fontweight='bold')
        plt.ylabel("Vehicle Count", fontsize=14, fontweight='bold')
        plt.title(f"{title} - Bar Chart", fontsize=16, fontweight='bold', pad=20)
        plt.xticks(rotation=0, ha='center')
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(values)*0.01,
                    f'{value}', ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    elif choice == 2:  # Pie Chart
        # Filter out zero values for pie chart
        non_zero_labels = [label for label, val in zip(labels, values) if val > 0]
        non_zero_values = [val for val in values if val > 0]
        non_zero_colors = [position_colors[label] for label in non_zero_labels]
        
        if non_zero_values:
            plt.pie(non_zero_values, labels=non_zero_labels, colors=non_zero_colors, 
                   autopct='%1.1f%%', startangle=90, explode=[0.05]*len(non_zero_values))
        else:
            plt.text(0.5, 0.5, 'No data to display', transform=plt.gca().transAxes,
                    ha='center', va='center', fontsize=16)
        plt.title(f"{title} - Pie Chart", fontsize=16, fontweight='bold', pad=20)
        plt.axis('equal')
    
    elif choice == 3:  # Donut Chart
        non_zero_labels = [label for label, val in zip(labels, values) if val > 0]
        non_zero_values = [val for val in values if val > 0]
        non_zero_colors = [position_colors[label] for label in non_zero_labels]
        
        if non_zero_values:
            plt.pie(non_zero_values, labels=non_zero_labels, colors=non_zero_colors,
                   autopct='%1.1f%%', startangle=90, pctdistance=0.85,
                   explode=[0.05]*len(non_zero_values))
            # Create donut by adding a white circle in center
            centre_circle = plt.Circle((0,0), 0.70, fc='white')
            plt.gca().add_artist(centre_circle)
        else:
            plt.text(0.5, 0.5, 'No data to display', transform=plt.gca().transAxes,
                    ha='center', va='center', fontsize=16)
        plt.title(f"{title} - Donut Chart", fontsize=16, fontweight='bold', pad=20)
        plt.axis('equal')
    
    elif choice == 4:  # Heat Map
        # Create a heatmap with position types
        heat_data = np.array(values).reshape(1, -1)
        
        sns.heatmap(heat_data, annot=True, fmt='d', cmap='RdYlBu_r', 
                   xticklabels=labels, yticklabels=['Vehicle Count'],
                   cbar_kws={'label': 'Count'})
        plt.xlabel("Relative Position", fontsize=14, fontweight='bold')
        plt.ylabel("", fontsize=14, fontweight='bold')
        plt.title(f"{title} - Heat Map", fontsize=16, fontweight='bold', pad=20)
        plt.xticks(rotation=0, ha='center')
    
    elif choice == 5:  # Radar Chart
        # Set up radar chart
        angles = np.linspace(0, 2*np.pi, len(labels), endpoint=False).tolist()
        values_radar = values + [values[0]]  # Close the circle
        angles += angles[:1]  # Close the circle
        
        ax = plt.subplot(111, projection='polar')
        ax.plot(angles, values_radar, 'o-', linewidth=2, color='#FF6B6B')
        ax.fill(angles, values_radar, alpha=0.25, color='#FF6B6B')
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels, fontsize=10)
        ax.set_ylabel("Vehicle Count", fontsize=12, fontweight='bold')
        plt.title(f"{title} - Radar Chart", fontsize=16, fontweight='bold', pad=30)
    
    elif choice == 6:  # Histogram
        # Create histogram-style visualization
        plt.hist(range(len(labels)), weights=values, bins=len(labels), 
                color=colors, alpha=0.7, edgecolor='black')
        plt.xlabel("Relative Position", fontsize=14, fontweight='bold')
        plt.ylabel("Vehicle Count", fontsize=14, fontweight='bold')
        plt.title(f"{title} - Histogram", fontsize=16, fontweight='bold', pad=20)
        plt.xticks(range(len(labels)), labels, rotation=0, ha='center')
        
        # Add value labels
        for i, value in enumerate(values):
            plt.text(i, value + max(values)*0.01, f'{value}', 
                    ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    elif choice == 7:  # Stacked Bar Chart
        bottom = 0
        for i, (label, value) in enumerate(zip(labels, values)):
            plt.bar(["Vehicle Position Distribution"], [value], bottom=bottom, 
                   color=colors[i], label=label, edgecolor='black')
            if value > 0:  # Only show text for non-zero values
                plt.text(0, bottom + value/2, f"{label}: {value}", 
                        ha="center", va="center", fontweight="bold", fontsize=10, rotation=0)
            bottom += value
        plt.xlabel("Analysis Type", fontsize=14, fontweight='bold')
        plt.ylabel("Vehicle Count", fontsize=14, fontweight='bold')
        plt.title(f"{title} - Stacked Bar Chart", fontsize=16, fontweight='bold', pad=20)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    elif choice == 8:  # Scatter Plot
        x_pos = np.arange(len(labels))
        plt.scatter(x_pos, values, c=colors, s=200, alpha=0.7, edgecolors='black')
        for i, (x, y) in enumerate(zip(x_pos, values)):
            plt.annotate(f"{y}", (x, y), textcoords="offset points", 
                        xytext=(0,10), ha='center', fontweight='bold')
        plt.xticks(x_pos, labels, rotation=0, ha='center')
        plt.xlabel("Relative Position", fontsize=14, fontweight='bold')
        plt.ylabel("Vehicle Count", fontsize=14, fontweight='bold')
        plt.title(f"{title} - Scatter Plot", fontsize=16, fontweight='bold', pad=20)
    
    elif choice == 9:  # Density Plot
        # Create density-like visualization
        df = pd.DataFrame({
            'Position': [label for label, count in zip(labels, values) for _ in range(count)],
            'Count': [1] * sum(values)
        })
        
        if not df.empty:
            sns.kdeplot(data=df, x='Position', weights='Count', 
                       fill=True, alpha=0.6, color='#FF6B6B')
            plt.xticks(rotation=0, ha='center')
        else:
            plt.text(0.5, 0.5, 'No data to display', transform=plt.gca().transAxes,
                    ha='center', va='center', fontsize=16)
        
        plt.xlabel("Relative Position", fontsize=14, fontweight='bold')
        plt.ylabel("Density", fontsize=14, fontweight='bold')
        plt.title(f"{title} - Density Plot", fontsize=16, fontweight='bold', pad=20)
    
    else:
        print("❌ Invalid choice. Please select 1-9.")
        return
    
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Determine chart type for filename
    chart_types = {
        1: "bar_chart",
        2: "pie_chart", 
        3: "donut_chart",
        4: "heatmap",
        5: "radar_chart",
        6: "histogram",
        7: "stacked_bar_chart",
        8: "scatter_plot",
        9: "density_plot"
    }
    
    chart_name = chart_types.get(choice, "chart")
    filename = f"vehicle_position_ego_{chart_name}.png"
    filepath = os.path.join(output_dir, filename)
    
    # Save the plot
    plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"📊 Plot saved as: {filepath}")
    
    # Print statistics
    total = sum(values)
    print(f"\n📊 Vehicle Position w.r.t. Ego Vehicle Analysis Statistics:")
    print(f"📈 Total vehicles detected around ego: {total}")
    
    if total > 0:
        for label, count in zip(labels, values):
            percentage = (count / total) * 100
            print(f"🔸 {label}: {count} vehicles ({percentage:.1f}%)")
            
        # Additional insights
        print(f"\n💡 Key Insights:")
        max_position = labels[values.index(max(values))]
        print(f"🔍 Most common vehicle position: {max_position} ({max(values)} vehicles)")
        
        # Traffic pattern analysis
        front_behind = complete_data["Front"] + complete_data["Behind"]
        left_right = complete_data["Left"] + complete_data["Right"]
        
        print(f"🚗 Longitudinal traffic (front/behind): {front_behind} vehicles ({front_behind/total*100:.1f}%)")
        print(f"🛣️ Lateral traffic (left/right): {left_right} vehicles ({left_right/total*100:.1f}%)")
        
        # Safety and driving context analysis
        front_vehicles = complete_data["Front"]
        behind_vehicles = complete_data["Behind"]
        
        if front_vehicles > behind_vehicles:
            print(f"🔍 Forward-heavy traffic pattern (more vehicles ahead)")
        elif behind_vehicles > front_vehicles:
            print(f"🔍 Rear-heavy traffic pattern (more vehicles behind)")
        else:
            print(f"🔍 Balanced front-rear distribution")
            
        # Lane change opportunity analysis
        left_vehicles = complete_data["Left"] 
        right_vehicles = complete_data["Right"]
        
        if left_vehicles < right_vehicles:
            print(f"🔄 Left lane appears less congested (potential for left lane changes)")
        elif right_vehicles < left_vehicles:
            print(f"🔄 Right lane appears less congested (potential for right lane changes)")
        else:
            print(f"🔄 Equal lateral distribution (limited lane change opportunities)")
            
        # Traffic density assessment
        if total > 80:
            print(f"🚦 High traffic density scenario")
        elif total > 40:
            print(f"🚗 Moderate traffic density scenario")
        else:
            print(f"🛣️ Low traffic density scenario")
            
        # Spatial distribution insights
        if complete_data["Front"] > sum([complete_data[pos] for pos in ["Left", "Right", "Behind"]]):
            print(f"📍 Front-dominant spatial distribution (following behavior)")
        
        # Safety implications
        total_surrounding = complete_data["Left"] + complete_data["Right"] + complete_data["Behind"]
        if total_surrounding > complete_data["Front"]:
            print(f"⚠️ Ego vehicle appears to be leading traffic (high responsibility)")
        else:
            print(f"🛡️ Ego vehicle in mixed traffic environment")
            
    else:
        print("⚠️ No vehicle position data found in the dataset")
    
    plt.show()

if __name__ == "__main__":
    # This plotting function should only be called from main.py with real nuScenes data
    # No sample/test data will be used
    print("❌ This file should be run through main.py to use real nuScenes mini dataset data.")
    print("🔍 Run 'python main.py' and select the Vehicle Position w.r.t. Ego Vehicle Analysis option.")
