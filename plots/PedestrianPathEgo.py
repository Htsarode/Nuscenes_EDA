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

def plot_pedestrian_path_ego(data: Dict[str, int], title: str = "Pedestrian Path w.r.t. Ego Vehicle Analysis", output_dir: str = "figures/exploratory"):
    """
    Plot pedestrian path relative to ego vehicle analysis data with multiple chart options.
    
    Args:
        data: Dictionary containing path categories and frame counts
        title: Title for the plot
        output_dir: Directory to save the plots
    """
    if not data:
        print("❌ No data available for plotting.")
        return
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Define all expected labels (will show all on x-axis even if count is 0)
    all_labels = ["In Path", "Out of Path"]
    
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
    
    # Path-themed colors (representing pedestrian safety context)
    path_colors = {
        "In Path": '#E74C3C',        # Red (Danger/Alert - pedestrian in vehicle path)
        "Out of Path": '#27AE60'     # Green (Safe - pedestrian out of vehicle path)
    }
    colors = [path_colors[label] for label in labels]
    
    plt.figure(figsize=(14, 8))
    plt.style.use('seaborn-v0_8')
    
    if choice == 1:  # Bar Chart
        bars = plt.bar(labels, values, color=colors, alpha=0.8, edgecolor='black', linewidth=1.2)
        plt.xlabel("Pedestrian Path Category", fontsize=14, fontweight='bold')
        plt.ylabel("Frame Count", fontsize=14, fontweight='bold')
        plt.title(f"{title} - Bar Chart", fontsize=16, fontweight='bold', pad=20)
        plt.xticks(rotation=0, ha='center')
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(values)*0.01,
                    f'{value}', ha='center', va='bottom', fontsize=12, fontweight='bold')
        add_custom_legend(plt.gca(), "bar", "Number of pedestrian instances observed", "Pedestrian path relative to ego vehicle")
    
    elif choice == 2:  # Pie Chart
        # Filter out zero values for pie chart
        non_zero_labels = [label for label, val in zip(labels, values) if val > 0]
        non_zero_values = [val for val in values if val > 0]
        non_zero_colors = [path_colors[label] for label in non_zero_labels]
        
        if non_zero_values:
            plt.pie(non_zero_values, labels=non_zero_labels, colors=non_zero_colors, 
                   autopct='%1.1f%%', startangle=90, explode=[0.05]*len(non_zero_values))
        else:
            plt.text(0.5, 0.5, 'No data to display', transform=plt.gca().transAxes,
                    ha='center', va='center', fontsize=16)
        plt.title(f"{title} - Pie Chart", fontsize=16, fontweight='bold', pad=20)
        plt.axis('equal')
        add_custom_legend(plt.gca(), "pie", "Number of pedestrian instances observed", "Pedestrian path relative to ego vehicle")
    
    elif choice == 3:  # Donut Chart
        non_zero_labels = [label for label, val in zip(labels, values) if val > 0]
        non_zero_values = [val for val in values if val > 0]
        non_zero_colors = [path_colors[label] for label in non_zero_labels]
        
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
        add_custom_legend(plt.gca(), "donut", "Number of pedestrian instances observed", "Pedestrian path relative to ego vehicle")
    
    elif choice == 4:  # Heat Map
        # Create a heatmap with path categories
        heat_data = np.array(values).reshape(1, -1)
        
        sns.heatmap(heat_data, annot=True, fmt='d', cmap='RdYlGn_r', 
                   xticklabels=labels, yticklabels=['Frame Count'],
                   cbar_kws={'label': 'Count'})
        plt.xlabel("Pedestrian Path Category", fontsize=14, fontweight='bold')
        plt.ylabel("", fontsize=14, fontweight='bold')
        plt.title(f"{title} - Heat Map", fontsize=16, fontweight='bold', pad=20)
        plt.xticks(rotation=0, ha='center')
        add_custom_legend(plt.gca(), "heatmap", "Number of pedestrian instances observed", "Pedestrian path relative to ego vehicle")
    
    elif choice == 5:  # Radar Chart
        # Set up radar chart
        angles = np.linspace(0, 2*np.pi, len(labels), endpoint=False).tolist()
        values_radar = values + [values[0]]  # Close the circle
        angles += angles[:1]  # Close the circle
        
        ax = plt.subplot(111, projection='polar')
        ax.plot(angles, values_radar, 'o-', linewidth=2, color='#E74C3C')
        ax.fill(angles, values_radar, alpha=0.25, color='#E74C3C')
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels, fontsize=10)
        ax.set_ylabel("Frame Count", fontsize=12, fontweight='bold')
        plt.title(f"{title} - Radar Chart", fontsize=16, fontweight='bold', pad=30)
        add_custom_legend(ax, "radar", "Number of pedestrian instances observed", "Pedestrian path relative to ego vehicle")
    
    elif choice == 6:  # Histogram
        # Create histogram-style visualization
        plt.hist(range(len(labels)), weights=values, bins=len(labels), 
                color=colors, alpha=0.7, edgecolor='black')
        plt.xlabel("Pedestrian Path Category", fontsize=14, fontweight='bold')
        plt.ylabel("Frame Count", fontsize=14, fontweight='bold')
        plt.title(f"{title} - Histogram", fontsize=16, fontweight='bold', pad=20)
        plt.xticks(range(len(labels)), labels, rotation=0, ha='center')
        
        # Add value labels
        for i, value in enumerate(values):
            plt.text(i, value + max(values)*0.01, f'{value}', 
                    ha='center', va='bottom', fontsize=12, fontweight='bold')
        add_custom_legend(plt.gca(), "histogram", "Number of pedestrian instances observed", "Pedestrian path relative to ego vehicle")
    
    elif choice == 7:  # Stacked Bar Chart
        bottom = 0
        for i, (label, value) in enumerate(zip(labels, values)):
            plt.bar(["Pedestrian Path Analysis"], [value], bottom=bottom, 
                   color=colors[i], label=label, edgecolor='black')
            if value > 0:  # Only show text for non-zero values
                plt.text(0, bottom + value/2, f"{label}: {value}", 
                        ha="center", va="center", fontweight="bold", fontsize=10, rotation=0)
            bottom += value
        plt.xlabel("Analysis Type", fontsize=14, fontweight='bold')
        plt.ylabel("Frame Count", fontsize=14, fontweight='bold')
        plt.title(f"{title} - Stacked Bar Chart", fontsize=16, fontweight='bold', pad=20)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        add_custom_legend(plt.gca(), "stackedbar", "Number of pedestrian instances observed", "Pedestrian path categories")
    
    elif choice == 8:  # Scatter Plot
        x_pos = np.arange(len(labels))
        plt.scatter(x_pos, values, c=colors, s=200, alpha=0.7, edgecolors='black')
        for i, (x, y) in enumerate(zip(x_pos, values)):
            plt.annotate(f"{y}", (x, y), textcoords="offset points", 
                        xytext=(0,10), ha='center', fontweight='bold')
        plt.xticks(x_pos, labels, rotation=0, ha='center')
        plt.xlabel("Pedestrian Path Category", fontsize=14, fontweight='bold')
        plt.ylabel("Frame Count", fontsize=14, fontweight='bold')
        plt.title(f"{title} - Scatter Plot", fontsize=16, fontweight='bold', pad=20)
        add_custom_legend(plt.gca(), "scatter", "Number of pedestrian instances observed", "Pedestrian path relative to ego vehicle")
    
    elif choice == 9:  # Density Plot
        # Create density-like visualization
        df = pd.DataFrame({
            'Path Category': [label for label, count in zip(labels, values) for _ in range(count)],
            'Count': [1] * sum(values)
        })
        
        if not df.empty:
            sns.kdeplot(data=df, x='Path Category', weights='Count', 
                       fill=True, alpha=0.6, color='#E74C3C')
            plt.xticks(rotation=0, ha='center')
            add_custom_legend(plt.gca(), "density", "Number of pedestrian instances observed", "Pedestrian path distribution")
        else:
            plt.text(0.5, 0.5, 'No data to display', transform=plt.gca().transAxes,
                    ha='center', va='center', fontsize=16)
        
        plt.xlabel("Pedestrian Path Category", fontsize=14, fontweight='bold')
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
    filename = f"pedestrian_path_ego_{chart_name}.png"
    filepath = os.path.join(output_dir, filename)
    
    # Save the plot
    plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"📊 Plot saved as: {filepath}")
    
    # Print statistics
    total = sum(values)
    print(f"\n📊 Pedestrian Path w.r.t. Ego Vehicle Analysis Statistics:")
    print(f"📈 Total pedestrian path instances analyzed: {total}")
    
    if total > 0:
        for label, count in zip(labels, values):
            percentage = (count / total) * 100
            print(f"🔸 {label}: {count} instances ({percentage:.1f}%)")
            
        # Additional insights
        print(f"\n💡 Key Insights:")
        
        in_path = complete_data["In Path"]
        out_of_path = complete_data["Out of Path"]
        
        # Safety assessment
        if in_path > 0:
            risk_ratio = (in_path / total) * 100
            print(f"⚠️ Pedestrian collision risk: {risk_ratio:.1f}% of pedestrians are in ego vehicle path")
            
            if risk_ratio > 50:
                print(f"🚨 HIGH RISK: Majority of pedestrians are in vehicle path")
            elif risk_ratio > 20:
                print(f"⚠️ MODERATE RISK: Significant pedestrians in vehicle path")
            else:
                print(f"✅ LOW RISK: Few pedestrians in vehicle path")
        else:
            print(f"✅ SAFE: No pedestrians detected in ego vehicle path")
            
        # Driving behavior implications
        if out_of_path > in_path:
            safety_margin = ((out_of_path - in_path) / total) * 100
            print(f"🛡️ Safety margin: {safety_margin:.1f}% more pedestrians are safely out of path")
            
        # Traffic scenario analysis
        if total > 50:
            print(f"🚶‍♂️ High pedestrian density environment")
        elif total > 20:
            print(f"🚶‍♀️ Moderate pedestrian density environment")
        else:
            print(f"🚶 Low pedestrian density environment")
            
        # Path analysis insights
        if in_path > 0:
            in_path_ratio = (in_path / total) * 100
            if in_path_ratio > 75:
                print(f"🔍 Urban crossing scenario detected (high in-path pedestrian ratio)")
            elif in_path_ratio > 40:
                print(f"🔍 Mixed traffic scenario (balanced pedestrian distribution)")
            else:
                print(f"🔍 Sidewalk/peripheral scenario (low in-path pedestrian ratio)")
                
        # Autonomous driving implications
        if in_path > 0:
            print(f"🤖 AV System Alert: Path monitoring and braking preparation required")
            print(f"🚦 Traffic Management: Consider speed reduction and pedestrian yield protocols")
        else:
            print(f"🤖 AV System Status: Clear path detected, standard operation mode")
            
        # Pedestrian behavior context
        total_pedestrians = total
        if total_pedestrians > 0:
            print(f"\n🔍 Pedestrian Behavior Context:")
            print(f"📍 Total pedestrian interactions analyzed: {total_pedestrians}")
            
            if in_path > 0:
                print(f"🚸 Active crossing behavior detected")
                print(f"⚠️ Driver attention and caution required")
            
            if out_of_path > 0:
                print(f"🚶 Sidewalk/safe area pedestrian activity")
                
    else:
        print("⚠️ No pedestrian path data found in the dataset")
        print("📍 This could indicate:")
        print("   • No pedestrians detected in the scenes")
        print("   • Pedestrians outside detection range")
        print("   • Dataset focus on highway/rural scenarios")
    
    plt.show()

if __name__ == "__main__":
    # This plotting function should only be called from main.py with real nuScenes data
    # No sample/test data will be used
    print("❌ This file should be run through main.py to use real nuScenes mini dataset data.")
    print("🔍 Run 'python main.py' and select the Pedestrian Path w.r.t. Ego Vehicle Analysis option.")
