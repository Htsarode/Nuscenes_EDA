import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os
from math import pi
import plotly.express as px
import plotly.graph_objects as go

def add_custom_legend(ax, chart_type, y_axis_motive=None, x_axis_meaning=None):
    """Add custom legend explaining axis meanings"""
    legend_text = []
    
    if chart_type in ['bar', 'histogram', 'stackedbar'] and y_axis_motive:
        legend_text.append(f"Y-axis: {y_axis_motive}")
    
    if x_axis_meaning:
        legend_text.append(f"X-axis: {x_axis_meaning}")
    
    if legend_text:
        legend_content = "\n".join(legend_text)
        ax.text(0.02, 0.98, legend_content, transform=ax.transAxes,
                fontsize=10, verticalalignment='top', 
                bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.8),
                zorder=1000)

def plot_pedestrian_distance(distance_data, output_dir="figures/exploratory"):
    """
    Plot pedestrian distance distribution with multiple chart options.
    
    Args:
        distance_data: Dictionary containing pedestrian distance data
        output_dir: Directory to save the plots
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Ensure all categories are present
    expected_labels = ["Far", "Medium", "Near"]
    cleaned = {label: distance_data.get(label, 0) for label in expected_labels}
    
    labels = list(cleaned.keys())
    values = list(cleaned.values())
    total = sum(values)
    
    while True:
        print("\nSelect chart type:")
        print("1. Bar Chart")
        print("2. Pie Chart")
        print("3. Donut Chart")
        print("4. Heat Map")
        print("5. Radar Chart")
        print("6. Histogram")
        print("7. Stacked Bar Chart")
        print("8. Scatter Plot")
        print("9. Density Plot")
        print("0. Exit")
        
        choice = input("Enter your choice (0-9): ")
        
        if choice == "0":
            break
            
        plt.figure(figsize=(10, 6))
        
        if choice == "1":  # Bar Chart
            ax = plt.gca()
            plt.bar(labels, values)
            plt.title("Pedestrian Distance Distribution")
            plt.xlabel("Distance Category")
            plt.ylabel("Number of Pedestrians")
            add_custom_legend(ax, 'bar', "Count of pedestrians", "Distance categories")
            
        elif choice == "2":  # Pie Chart
            plt.pie(values, labels=labels, autopct='%1.1f%%')
            plt.title("Pedestrian Distance Distribution")
            
        elif choice == "3":  # Donut Chart
            plt.pie(values, labels=labels, autopct='%1.1f%%', pctdistance=0.85)
            centre_circle = plt.Circle((0,0), 0.70, fc='white')
            fig = plt.gcf()
            fig.gca().add_artist(centre_circle)
            plt.title("Pedestrian Distance Distribution")
            
        elif choice == "4":  # Heat Map
            data_matrix = np.array(values).reshape(1, -1)
            ax = sns.heatmap(data_matrix, annot=True, xticklabels=labels, yticklabels=['Count'])
            plt.title("Pedestrian Distance Heatmap")
            
        elif choice == "5":  # Radar Chart
            angles = [n / float(len(labels)) * 2 * pi for n in range(len(labels))]
            angles += angles[:1]
            
            ax = plt.subplot(111, polar=True)
            values_plot = values + [values[0]]
            ax.plot(angles, values_plot)
            ax.fill(angles, values_plot, alpha=0.25)
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(labels)
            plt.title("Pedestrian Distance Radar Chart")
            
        elif choice == "6":  # Histogram
            plt.hist(np.repeat(range(len(labels)), values), bins=len(labels))
            plt.xticks(range(len(labels)), labels)
            plt.title("Pedestrian Distance Histogram")
            plt.xlabel("Distance Category")
            plt.ylabel("Frequency")
            
        elif choice == "7":  # Stacked Bar Chart
            bottom = 0
            for i, value in enumerate(values):
                plt.bar(0, value, bottom=bottom, label=labels[i])
                bottom += value
            plt.title("Pedestrian Distance Stacked Bar Chart")
            plt.legend()
            plt.xticks([])
            plt.ylabel("Number of Pedestrians")
            
        elif choice == "8":  # Scatter Plot
            plt.scatter(labels, values)
            plt.title("Pedestrian Distance Scatter Plot")
            plt.xlabel("Distance Category")
            plt.ylabel("Number of Pedestrians")
            
        elif choice == "9":  # Density Plot
            # Create artificial distribution for visualization
            data = []
            for i, count in enumerate(values):
                data.extend([i] * count)
            sns.kdeplot(data=data)
            plt.xticks(range(len(labels)), labels)
            plt.title("Pedestrian Distance Density Plot")
            plt.xlabel("Distance Category")
            plt.ylabel("Density")
            
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'pedestrian_distance_{choice}.png'))
        plt.close()
        
    return distance_data
