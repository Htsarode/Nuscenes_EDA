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

def plot_multimodal_synchronization(data: Dict[str, int], title: str = "Multi-Modal Synchronization Analysis", output_dir: str = "figures/exploratory"):
    """
    Plot multi-modal synchronization data with multiple chart options.
    
    Args:
        data: Dictionary containing sensor type synchronized frame counts
        title: Title for the plot
        output_dir: Directory to save the plots
    """
    if not data:
        print("❌ No data available for plotting.")
        return
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Define all expected labels (will show all on x-axis even if count is 0)
    all_labels = ["Lidar", "Radar", "Camera"]
    
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
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # Blue, Orange, Green
    
    plt.figure(figsize=(12, 8))
    plt.style.use('seaborn-v0_8')
    
    if choice == 1:  # Bar Chart
        bars = plt.bar(labels, values, color=colors, alpha=0.8, edgecolor='black', linewidth=1.2)
        plt.xlabel("Sensor Type", fontsize=14, fontweight='bold')
        plt.ylabel("Number of Synchronized Frames", fontsize=14, fontweight='bold')
        plt.title(f"{title} - Bar Chart", fontsize=16, fontweight='bold', pad=20)
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(values)*0.01,
                    f'{value}', ha='center', va='bottom', fontsize=12, fontweight='bold')
        add_custom_legend(plt.gca(), "bar", "Number of synchronized frame instances", "Multi-modal sensor types")
    
    elif choice == 2:  # Pie Chart
        # Filter out zero values for pie chart
        non_zero_labels = [label for label, val in zip(labels, values) if val > 0]
        non_zero_values = [val for val in values if val > 0]
        non_zero_colors = [colors[i] for i, val in enumerate(values) if val > 0]
        
        if non_zero_values:
            plt.pie(non_zero_values, labels=non_zero_labels, colors=non_zero_colors, 
                   autopct='%1.1f%%', startangle=90, explode=[0.05]*len(non_zero_values))
        else:
            plt.text(0.5, 0.5, 'No data to display', transform=plt.gca().transAxes,
                    ha='center', va='center', fontsize=16)
        plt.title(f"{title} - Pie Chart", fontsize=16, fontweight='bold', pad=20)
        plt.axis('equal')
        add_custom_legend(plt.gca(), "pie", "Number of synchronized frame instances", "Multi-modal sensor types")
    
    elif choice == 3:  # Donut Chart
        non_zero_labels = [label for label, val in zip(labels, values) if val > 0]
        non_zero_values = [val for val in values if val > 0]
        non_zero_colors = [colors[i] for i, val in enumerate(values) if val > 0]
        
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
        add_custom_legend(plt.gca(), "donut", "Number of synchronized frame instances", "Multi-modal sensor types")
    
    elif choice == 4:  # Heat Map
        # Create a heatmap with sensor types
        heat_data = np.array(values).reshape(1, -1)
        sns.heatmap(heat_data, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=labels, yticklabels=['Synchronized Frames'],
                   cbar_kws={'label': 'Frame Count'})
        plt.xlabel("Sensor Type", fontsize=14, fontweight='bold')
        plt.ylabel("", fontsize=14, fontweight='bold')
        plt.title(f"{title} - Heat Map", fontsize=16, fontweight='bold', pad=20)
        add_custom_legend(plt.gca(), "heatmap", "Number of synchronized frame instances", "Multi-modal sensor types")
    
    elif choice == 5:  # Radar Chart
        # Set up radar chart
        angles = np.linspace(0, 2*np.pi, len(labels), endpoint=False).tolist()
        values_radar = values + [values[0]]  # Close the circle
        angles += angles[:1]  # Close the circle
        
        ax = plt.subplot(111, projection='polar')
        ax.plot(angles, values_radar, 'o-', linewidth=2, color=colors[0])
        ax.fill(angles, values_radar, alpha=0.25, color=colors[0])
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels)
        ax.set_ylabel("Frame Count", fontsize=12, fontweight='bold')
        plt.title(f"{title} - Radar Chart", fontsize=16, fontweight='bold', pad=30)
        add_custom_legend(ax, "radar", "Number of synchronized frame instances", "Multi-modal sensor types")
    
    elif choice == 6:  # Histogram
        # Create histogram-style visualization
        plt.hist(range(len(labels)), weights=values, bins=len(labels), 
                color=colors, alpha=0.7, edgecolor='black')
        plt.xlabel("Sensor Type", fontsize=14, fontweight='bold')
        plt.ylabel("Number of Synchronized Frames", fontsize=14, fontweight='bold')
        plt.title(f"{title} - Histogram", fontsize=16, fontweight='bold', pad=20)
        plt.xticks(range(len(labels)), labels)
        
        # Add value labels
        for i, value in enumerate(values):
            plt.text(i, value + max(values)*0.01, f'{value}', 
                    ha='center', va='bottom', fontsize=12, fontweight='bold')
        add_custom_legend(plt.gca(), "histogram", "Number of synchronized frame instances", "Multi-modal sensor types")
    
    elif choice == 7:  # Stacked Bar Chart
        bottom = 0
        for i, (label, value) in enumerate(zip(labels, values)):
            plt.bar(["Multi-Modal Synchronization"], [value], bottom=bottom, 
                   color=colors[i], label=label, edgecolor='black')
            plt.text(0, bottom + value/2, f"{label}: {value}", 
                    ha="center", va="center", fontweight="bold", fontsize=10)
            bottom += value
        plt.xlabel("Analysis Type", fontsize=14, fontweight='bold')
        plt.ylabel("Number of Synchronized Frames", fontsize=14, fontweight='bold')
        plt.title(f"{title} - Stacked Bar Chart", fontsize=16, fontweight='bold', pad=20)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        add_custom_legend(plt.gca(), "stackedbar", "Number of synchronized frame instances", "Multi-modal sensor categories")
    
    elif choice == 8:  # Scatter Plot
        x_pos = np.arange(len(labels))
        plt.scatter(x_pos, values, c=colors, s=200, alpha=0.7, edgecolors='black')
        for i, (x, y) in enumerate(zip(x_pos, values)):
            plt.annotate(f"{labels[i]}\n({y})", (x, y), textcoords="offset points", 
                        xytext=(0,10), ha='center', fontweight='bold')
        plt.xticks(x_pos, labels)
        plt.xlabel("Sensor Type", fontsize=14, fontweight='bold')
        plt.ylabel("Number of Synchronized Frames", fontsize=14, fontweight='bold')
        plt.title(f"{title} - Scatter Plot", fontsize=16, fontweight='bold', pad=20)
        add_custom_legend(plt.gca(), "scatter", "Number of synchronized frame instances", "Multi-modal sensor types")
    
    elif choice == 9:  # Density Plot
        # Create density-like visualization
        df = pd.DataFrame({
            'Sensor Type': [label for label, count in zip(labels, values) for _ in range(count)],
            'Count': [1] * sum(values)
        })
        
        if not df.empty:
            sns.kdeplot(data=df, x='Sensor Type', weights='Count', 
                       fill=True, alpha=0.6, color='#1f77b4')
            add_custom_legend(plt.gca(), "density", "Number of synchronized frame instances", "Sensor synchronization distribution")
        else:
            plt.text(0.5, 0.5, 'No data to display', transform=plt.gca().transAxes,
                    ha='center', va='center', fontsize=16)
        
        plt.xlabel("Sensor Type", fontsize=14, fontweight='bold')
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
    filename = f"multimodal_synchronization_{chart_name}.png"
    filepath = os.path.join(output_dir, filename)
    
    # Save the plot
    plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"📊 Plot saved as: {filepath}")
    
    # Print statistics
    total = sum(values)
    print(f"\n📊 Multi-Modal Synchronization Statistics:")
    print(f"📈 Total synchronized frames: {total}")
    
    if total > 0:
        for label, count in zip(labels, values):
            percentage = (count / total) * 100
            print(f"🔸 {label}: {count} frames ({percentage:.1f}%)")
            
        # Additional insights
        print(f"\n💡 Key Insights:")
        max_sensor = labels[values.index(max(values))]
        print(f"🔍 Most synchronized sensor: {max_sensor} ({max(values)} frames)")
        
        # Calculate synchronization ratio
        non_zero_sensors = sum(1 for v in values if v > 0)
        sync_ratio = (non_zero_sensors / len(labels)) * 100
        print(f"⚡ Sensor synchronization coverage: {sync_ratio:.1f}% ({non_zero_sensors}/{len(labels)} sensor types)")
    else:
        print("⚠️ No multi-modal synchronization data found in the dataset")
    
    plt.show()

if __name__ == "__main__":
    # This plotting function should only be called from main.py with real nuScenes data
    # No sample/test data will be used
    print("❌ This file should be run through main.py to use real nuScenes mini dataset data.")
    print("🔍 Run 'python main.py' and select the Multi-Modal Synchronization Analysis option.")
