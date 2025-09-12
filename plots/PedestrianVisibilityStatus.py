import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import Dict
import pandas as pd
import os

def plot_pedestrian_visibility_status(data: Dict[str, int], title: str = "Pedestrian Visibility Status Analysis", output_dir: str = "figures/exploratory"):
    """
    Plot pedestrian visibility status data with multiple chart options.
    
    Args:
        data: Dictionary containing visibility status counts
        title: Title for the plot
        output_dir: Directory to save the plots
    """
    if not data:
        print("❌ No data available for plotting.")
        return
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Define all expected labels (will show all on x-axis even if count is 0)
    all_labels = ["Fully Visible", "Occluded", "Truncated"]
    
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
    colors = ['#2E8B57', '#FF6B6B', '#4ECDC4']  # Green, Red, Teal
    
    plt.figure(figsize=(12, 8))
    plt.style.use('seaborn-v0_8')
    
    if choice == 1:  # Bar Chart
        bars = plt.bar(labels, values, color=colors, alpha=0.8, edgecolor='black', linewidth=1.2)
        plt.xlabel("Visibility Status", fontsize=14, fontweight='bold')
        plt.ylabel("Pedestrian Count", fontsize=14, fontweight='bold')
        plt.title(f"{title} - Bar Chart", fontsize=16, fontweight='bold', pad=20)
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(values)*0.01,
                    f'{value}', ha='center', va='bottom', fontsize=12, fontweight='bold')
    
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
    
    elif choice == 4:  # Heat Map
        # Create a heatmap matrix
        data_matrix = np.array(values).reshape(1, -1)
        sns.heatmap(data_matrix, annot=True, fmt='d', cmap='YlOrRd',
                   xticklabels=labels, yticklabels=['Count'],
                   cbar_kws={'label': 'Pedestrian Count'})
        plt.title(f"{title} - Heat Map", fontsize=16, fontweight='bold', pad=20)
        plt.xlabel("Visibility Status", fontsize=14, fontweight='bold')
    
    elif choice == 5:  # Radar Chart
        angles = np.linspace(0, 2*np.pi, len(labels), endpoint=False).tolist()
        values_radar = values + [values[0]]  # Complete the circle
        angles += [angles[0]]
        
        ax = plt.subplot(111, projection='polar')
        ax.plot(angles, values_radar, 'o-', linewidth=2, color='#2E8B57')
        ax.fill(angles, values_radar, color='#2E8B57', alpha=0.25)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels)
        ax.set_ylim(0, max(values) * 1.1 if max(values) > 0 else 1)
        plt.title(f"{title} - Radar Chart", fontsize=16, fontweight='bold', pad=20)
    
    elif choice == 6:  # Histogram
        # Create histogram-style plot
        plt.hist([labels[i] for i in range(len(labels)) for _ in range(values[i])], 
                bins=len(labels), color=colors, alpha=0.7, edgecolor='black')
        plt.xlabel("Visibility Status", fontsize=14, fontweight='bold')
        plt.ylabel("Pedestrian Count", fontsize=14, fontweight='bold')
        plt.title(f"{title} - Histogram", fontsize=16, fontweight='bold', pad=20)
        plt.xticks(range(len(labels)), labels)
    
    elif choice == 7:  # Stacked Bar Chart
        plt.bar(['Pedestrian Visibility'], [sum(values)], color='lightgray', alpha=0.3)
        bottom = 0
        for i, (label, value) in enumerate(zip(labels, values)):
            if value > 0:
                plt.bar(['Pedestrian Visibility'], [value], bottom=bottom, 
                       label=label, color=colors[i], alpha=0.8)
                bottom += value
        plt.xlabel("Analysis Category", fontsize=14, fontweight='bold')
        plt.ylabel("Pedestrian Count", fontsize=14, fontweight='bold')
        plt.title(f"{title} - Stacked Bar Chart", fontsize=16, fontweight='bold', pad=20)
        plt.legend()
    
    elif choice == 8:  # Scatter Plot
        x_pos = range(len(labels))
        plt.scatter(x_pos, values, s=[v*10 + 100 for v in values], 
                   c=colors, alpha=0.7, edgecolors='black', linewidth=2)
        plt.xlabel("Visibility Status", fontsize=14, fontweight='bold')
        plt.ylabel("Pedestrian Count", fontsize=14, fontweight='bold')
        plt.title(f"{title} - Scatter Plot", fontsize=16, fontweight='bold', pad=20)
        plt.xticks(x_pos, labels)
        
        # Add value labels
        for i, value in enumerate(values):
            plt.annotate(f'{value}', (i, value), textcoords="offset points", 
                        xytext=(0,10), ha='center', fontweight='bold')
    
    elif choice == 9:  # Density Plot
        # Create density-like visualization
        df = pd.DataFrame({
            'Visibility Status': [label for label, count in zip(labels, values) for _ in range(count)],
            'Count': [1] * sum(values)
        })
        
        if not df.empty:
            sns.kdeplot(data=df, x='Visibility Status', weights='Count', 
                       fill=True, alpha=0.6, color='#2E8B57')
        else:
            plt.text(0.5, 0.5, 'No data to display', transform=plt.gca().transAxes,
                    ha='center', va='center', fontsize=16)
        
        plt.xlabel("Visibility Status", fontsize=14, fontweight='bold')
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
    filename = f"pedestrian_visibility_status_{chart_name}.png"
    filepath = os.path.join(output_dir, filename)
    
    # Save the plot
    plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"📊 Plot saved as: {filepath}")
    
    # Print statistics
    total = sum(values)
    print(f"\n📊 Pedestrian Visibility Status Statistics:")
    print(f"📈 Total pedestrians analyzed: {total}")
    
    if total > 0:
        for label, count in zip(labels, values):
            percentage = (count / total) * 100
            print(f"🔸 {label}: {count} ({percentage:.1f}%)")
            
        # Additional insights
        print(f"\n💡 Key Insights:")
        max_category = labels[values.index(max(values))]
        print(f"🔍 Most common visibility status: {max_category} ({max(values)} pedestrians)")
        
        if values[0] > 0:  # Fully Visible
            visibility_ratio = (values[0] / total) * 100
            print(f"👁️ Visibility ratio: {visibility_ratio:.1f}% of pedestrians are fully visible")
    else:
        print("⚠️ No pedestrian visibility data found in the dataset")
    
    plt.show()

if __name__ == "__main__":
    # This plotting function should only be called from main.py with real nuScenes data
    # No sample/test data will be used
    print("❌ This file should be run through main.py to use real nuScenes mini dataset data.")
