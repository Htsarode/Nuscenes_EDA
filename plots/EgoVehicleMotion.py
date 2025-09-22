import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import Dict
import pandas as pd
import os

def add_custom_legend(ax, chart_type, y_axis_motive, x_axis_meaning):
    """
    Add a custom legend explaining the meaning of x and y axes.
    Only shows y-axis motive for bar, histogram, and stacked bar charts.
    
    Args:
        ax: The matplotlib axes object
        chart_type: Type of chart (bar, pie, donut, heatmap, radar, histogram, stackedbar, scatter, density)
        y_axis_motive: Description of what the y-axis represents
        x_axis_meaning: Description of what the x-axis represents
    """
    legend_text = f"📊 X-Axis: {x_axis_meaning}"
    
    # Only show y-axis motive for specific chart types
    if chart_type in ["bar", "histogram", "stackedbar"]:
        legend_text += f"\n📈 Y-Axis: {y_axis_motive}"
    
    ax.text(0.02, 0.98, legend_text, transform=ax.transAxes, fontsize=9, 
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

def plot_ego_vehicle_motion(data: Dict[str, int], title: str = "Ego Vehicle Motion Analysis", output_dir: str = "figures/exploratory"):
    """
    Plot ego vehicle motion analysis data with multiple chart options.
    
    Args:
        data: Dictionary containing ego motion states and frame counts
        title: Title for the plot
        output_dir: Directory to save the plots
    """
    if not data:
        print("❌ No data available for plotting.")
        return
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Define all expected labels (will show all on x-axis even if count is 0)
    all_labels = ["Stop at red light", "Stop at ped crossing", "moving"]
    
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
    
    # Motion-themed colors
    motion_colors = {
        "Stop at red light": '#FF4444',      # Red
        "Stop at ped crossing": '#FFA500',   # Orange  
        "moving": '#32CD32'                  # Lime Green
    }
    colors = [motion_colors[label] for label in labels]
    
    plt.figure(figsize=(14, 8))
    plt.style.use('seaborn-v0_8')
    
    if choice == 1:  # Bar Chart
        bars = plt.bar(labels, values, color=colors, alpha=0.8, edgecolor='black', linewidth=1.2)
        plt.xlabel("Ego Motion State", fontsize=14, fontweight='bold')
        plt.ylabel("Frame Count", fontsize=14, fontweight='bold')
        plt.title(f"{title} - Bar Chart", fontsize=16, fontweight='bold', pad=20)
        plt.xticks(rotation=15, ha='right')
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(values)*0.01,
                    f'{value}', ha='center', va='bottom', fontsize=12, fontweight='bold')
        
        add_custom_legend(plt.gca(), "bar", 
                        "Number of ego vehicle motion frames detected",
                        "Ego vehicle motion states/maneuvers")
    
    elif choice == 2:  # Pie Chart
        # Filter out zero values for pie chart
        non_zero_labels = [label for label, val in zip(labels, values) if val > 0]
        non_zero_values = [val for val in values if val > 0]
        non_zero_colors = [motion_colors[label] for label in non_zero_labels]
        
        if non_zero_values:
            plt.pie(non_zero_values, labels=non_zero_labels, colors=non_zero_colors, 
                   autopct='%1.1f%%', startangle=90, explode=[0.05]*len(non_zero_values))
        else:
            plt.text(0.5, 0.5, 'No data to display', transform=plt.gca().transAxes,
                    ha='center', va='center', fontsize=16)
        plt.title(f"{title} - Pie Chart", fontsize=16, fontweight='bold', pad=20)
        plt.axis('equal')
        
        add_custom_legend(plt.gca(), "pie", 
                        "Number of ego vehicle motion frames detected",
                        "Ego vehicle motion states/maneuvers")
    
    elif choice == 3:  # Donut Chart
        non_zero_labels = [label for label, val in zip(labels, values) if val > 0]
        non_zero_values = [val for val in values if val > 0]
        non_zero_colors = [motion_colors[label] for label in non_zero_labels]
        
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
        
        add_custom_legend(plt.gca(), "donut", 
                        "Number of ego vehicle motion frames detected",
                        "Ego vehicle motion states/maneuvers")
    
    elif choice == 4:  # Heat Map
        # Create a heatmap with motion states
        heat_data = np.array(values).reshape(1, -1)
        
        sns.heatmap(heat_data, annot=True, fmt='d', cmap='RdYlGn', 
                   xticklabels=labels, yticklabels=['Frame Count'],
                   cbar_kws={'label': 'Count'})
        plt.xlabel("Ego Motion State", fontsize=14, fontweight='bold')
        plt.ylabel("", fontsize=14, fontweight='bold')
        plt.title(f"{title} - Heat Map", fontsize=16, fontweight='bold', pad=20)
        plt.xticks(rotation=15, ha='right')
        
        add_custom_legend(plt.gca(), "heatmap", 
                        "Number of ego vehicle motion frames detected",
                        "Ego vehicle motion states/maneuvers")
    
    elif choice == 5:  # Radar Chart
        # Set up radar chart
        angles = np.linspace(0, 2*np.pi, len(labels), endpoint=False).tolist()
        values_radar = values + [values[0]]  # Close the circle
        angles += angles[:1]  # Close the circle
        
        ax = plt.subplot(111, projection='polar')
        ax.plot(angles, values_radar, 'o-', linewidth=2, color='#32CD32')
        ax.fill(angles, values_radar, alpha=0.25, color='#32CD32')
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels, fontsize=10)
        ax.set_ylabel("Frame Count", fontsize=12, fontweight='bold')
        plt.title(f"{title} - Radar Chart", fontsize=16, fontweight='bold', pad=30)
        
        add_custom_legend(ax, "radar", 
                        "Number of ego vehicle motion frames detected",
                        "Ego vehicle motion states/maneuvers")
    
    elif choice == 6:  # Histogram
        # Create histogram-style visualization
        plt.hist(range(len(labels)), weights=values, bins=len(labels), 
                color=colors, alpha=0.7, edgecolor='black')
        plt.xlabel("Ego Motion State", fontsize=14, fontweight='bold')
        plt.ylabel("Frame Count", fontsize=14, fontweight='bold')
        plt.title(f"{title} - Histogram", fontsize=16, fontweight='bold', pad=20)
        plt.xticks(range(len(labels)), labels, rotation=15, ha='right')
        
        # Add value labels
        for i, value in enumerate(values):
            plt.text(i, value + max(values)*0.01, f'{value}', 
                    ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    elif choice == 7:  # Stacked Bar Chart
        bottom = 0
        for i, (label, value) in enumerate(zip(labels, values)):
            plt.bar(["Ego Vehicle Motion States"], [value], bottom=bottom, 
                   color=colors[i], label=label, edgecolor='black')
            if value > 0:  # Only show text for non-zero values
                plt.text(0, bottom + value/2, f"{label}: {value}", 
                        ha="center", va="center", fontweight="bold", fontsize=10, rotation=0)
            bottom += value
        plt.xlabel("Analysis Type", fontsize=14, fontweight='bold')
        plt.ylabel("Frame Count", fontsize=14, fontweight='bold')
        plt.title(f"{title} - Stacked Bar Chart", fontsize=16, fontweight='bold', pad=20)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    elif choice == 8:  # Scatter Plot
        x_pos = np.arange(len(labels))
        plt.scatter(x_pos, values, c=colors, s=200, alpha=0.7, edgecolors='black')
        for i, (x, y) in enumerate(zip(x_pos, values)):
            plt.annotate(f"{y}", (x, y), textcoords="offset points", 
                        xytext=(0,10), ha='center', fontweight='bold')
        plt.xticks(x_pos, labels, rotation=15, ha='right')
        plt.xlabel("Ego Motion State", fontsize=14, fontweight='bold')
        plt.ylabel("Frame Count", fontsize=14, fontweight='bold')
        plt.title(f"{title} - Scatter Plot", fontsize=16, fontweight='bold', pad=20)
    
    elif choice == 9:  # Density Plot
        # Create density-like visualization
        df = pd.DataFrame({
            'Ego Motion State': [label for label, count in zip(labels, values) for _ in range(count)],
            'Count': [1] * sum(values)
        })
        
        if not df.empty:
            sns.kdeplot(data=df, x='Ego Motion State', weights='Count', 
                       fill=True, alpha=0.6, color='#32CD32')
            plt.xticks(rotation=15, ha='right')
        else:
            plt.text(0.5, 0.5, 'No data to display', transform=plt.gca().transAxes,
                    ha='center', va='center', fontsize=16)
        
        plt.xlabel("Ego Motion State", fontsize=14, fontweight='bold')
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
    filename = f"ego_vehicle_motion_{chart_name}.png"
    filepath = os.path.join(output_dir, filename)
    
    # Save the plot
    plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"📊 Plot saved as: {filepath}")
    
    # Print statistics
    total = sum(values)
    print(f"\n📊 Ego Vehicle Motion Analysis Statistics:")
    print(f"📈 Total frames analyzed: {total}")
    
    if total > 0:
        for label, count in zip(labels, values):
            percentage = (count / total) * 100
            print(f"🔸 {label}: {count} frames ({percentage:.1f}%)")
            
        # Additional insights
        print(f"\n💡 Key Insights:")
        max_motion = labels[values.index(max(values))]
        print(f"🔍 Most common motion state: {max_motion} ({max(values)} frames)")
        
        # Motion categorization
        stopping_states = ["Stop at red light", "Stop at ped crossing"]
        moving_states = ["moving"]
        
        stopping_count = sum(complete_data[state] for state in stopping_states)
        moving_count = sum(complete_data[state] for state in moving_states)
        
        print(f"🛑 Stopping scenarios: {stopping_count} frames ({stopping_count/total*100:.1f}%)")
        print(f"🚗 Moving scenarios: {moving_count} frames ({moving_count/total*100:.1f}%)")
        
        if moving_count > stopping_count:
            print(f"📈 Ego vehicle spends more time moving than stopped")
        elif stopping_count > moving_count:
            print(f"📉 Ego vehicle has frequent stopping scenarios")
        else:
            print(f"⚖️ Balanced distribution between moving and stopping")
            
        # Safety insights
        safety_stops = complete_data["Stop at ped crossing"]
        traffic_stops = complete_data["Stop at red light"]
        
        if safety_stops > 0:
            safety_ratio = (safety_stops / total) * 100
            print(f"🚶 Pedestrian safety stops: {safety_ratio:.1f}% of total frames")
        
        if traffic_stops > 0:
            traffic_ratio = (traffic_stops / total) * 100
            print(f"🚦 Traffic light compliance: {traffic_ratio:.1f}% of total frames")
    else:
        print("⚠️ No ego vehicle motion data found in the dataset")
    
    plt.show()

if __name__ == "__main__":
    # This plotting function should only be called from main.py with real nuScenes data
    # No sample/test data will be used
    print("❌ This file should be run through main.py to use real nuScenes mini dataset data.")
    print("🔍 Run 'python main.py' and select the Ego Vehicle Motion Analysis option.")
