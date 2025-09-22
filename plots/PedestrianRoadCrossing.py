import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os
from math import pi
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

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

def plot_pedestrian_road_crossing(crossing_data, output_dir="figures/exploratory"):
    """
    Plot pedestrian road crossing distribution with multiple chart options.
    
    Args:
        crossing_data: Dictionary with crossing type counts
        output_dir: Directory to save the plots
    """
    os.makedirs(output_dir, exist_ok=True)

    # --- Ensure all expected labels are present ---
    expected_labels = ["Jaywalking", "Crosswalk"]
    cleaned = {label: crossing_data.get(label, 0) for label in expected_labels}

    labels = list(cleaned.keys())
    values = list(cleaned.values())
    total = sum(values)

    # --- Chart options ---
    chart_map = {
        "1": "bar",
        "2": "pie", 
        "3": "donut",
        "4": "heatmap",
        "5": "radar",
        "6": "histogram",
        "7": "stackedbar",
        "8": "scatter",
        "9": "density"
    }

    print("\nAvailable charts:")
    for num, name in chart_map.items():
        print(f"{num}. {name.title()}")
    chosen = input("Enter chart numbers (comma-separated): ").replace(" ", "").split(",")
    selected_charts = [chart_map[c] for c in chosen if c in chart_map]

    if not selected_charts:
        print("⚠️ No valid chart selected. Exiting.")
        return

    colors = list(plt.cm.Set2(np.linspace(0, 1, len(labels))))

    # --- Statistical Summary ---
    mean_val = np.mean(values)
    std_val = np.std(values)
    max_val = max(values)
    min_val = min(values)
    most_common = labels[values.index(max_val)]
    least_common = labels[values.index(min_val)]

    stats_text = f"""Pedestrian Road Crossing Summary

Total Crossings: {total}
Crossing Types: {len(labels)}

Most Common: {most_common} ({max_val})
Least Common: {least_common} ({min_val})

Mean: {mean_val:.1f}
Std Dev: {std_val:.1f}
Range: {max_val - min_val}

Distribution:"""
    for label, value in zip(labels, values):
        pct = (value / total * 100) if total > 0 else 0
        stats_text += f"\n• {label}: {value} ({pct:.1f}%)"

    def add_stats(ax):
        ax.axis("off")
        ax.text(0.05, 0.95, stats_text, transform=ax.transAxes,
                fontsize=11, verticalalignment="top", fontfamily="monospace",
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightcyan", alpha=0.8))
        ax.set_title("Statistical Summary", fontsize=14, fontweight="bold")

    # --- Generate chosen charts ---
    for chart in selected_charts:
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))
        fig.suptitle(f"{chart.capitalize()} Chart with Statistical Summary - Pedestrian Road Crossing",
                     fontsize=16, fontweight="bold", y=0.98)
        ax_chart = axes[0]

        if chart == "bar":
            bars = ax_chart.bar(labels, values, color=colors, edgecolor="black", linewidth=1.2)
            for bar, val in zip(bars, values):
                ax_chart.text(bar.get_x()+bar.get_width()/2, val, f"{val}",
                              ha="center", va="bottom", fontweight="bold", fontsize=10)
            ax_chart.set_title("Bar Chart - Pedestrian Crossing Type Counts", fontsize=14, fontweight="bold")
            ax_chart.set_xlabel("Pedestrian Crossing Type", fontsize=12, fontweight="bold")
            ax_chart.set_ylabel("Occurrence Count", fontsize=12, fontweight="bold")
            ax_chart.grid(True, alpha=0.3)
            add_custom_legend(ax_chart, "bar", "Number of pedestrian crossing instances observed", "Pedestrian crossing types")

        elif chart == "pie":
            if total > 0:
                wedges, texts, autotexts = ax_chart.pie(values, labels=labels, autopct='%1.1f%%', 
                                                        colors=colors, startangle=90, 
                                                        explode=[0.05] * len(labels))
                for autotext in autotexts:
                    autotext.set_color('white')
                    autotext.set_fontweight('bold')
                    autotext.set_fontsize(11)
            else:
                ax_chart.text(0.5, 0.5, "No Data Available", ha="center", va="center", 
                              fontsize=14, fontweight="bold", transform=ax_chart.transAxes)
            ax_chart.set_title("Pie Chart - Pedestrian Crossing Type Distribution", fontsize=14, fontweight="bold")
            add_custom_legend(ax_chart, "pie", "Number of pedestrian crossing instances observed", "Pedestrian crossing types")

        elif chart == "donut":
            if total > 0:
                wedges, texts, autotexts = ax_chart.pie(values, labels=labels, autopct='%1.1f%%', 
                                                        colors=colors, startangle=90, 
                                                        pctdistance=0.85, 
                                                        explode=[0.02] * len(labels))
                centre_circle = plt.Circle((0, 0), 0.70, fc='white')
                ax_chart.add_artist(centre_circle)
                ax_chart.text(0, 0, f"Total\n{total}", ha="center", va="center", 
                              fontsize=12, fontweight="bold")
                for autotext in autotexts:
                    autotext.set_color('black')
                    autotext.set_fontweight('bold')
            else:
                ax_chart.text(0.5, 0.5, "No Data Available", ha="center", va="center", 
                              fontsize=14, fontweight="bold", transform=ax_chart.transAxes)
            ax_chart.set_title("Donut Chart - Pedestrian Crossing Type Distribution", fontsize=14, fontweight="bold")
            add_custom_legend(ax_chart, "donut", "Number of pedestrian crossing instances observed", "Pedestrian crossing types")

        elif chart == "heatmap":
            heatmap_data = np.array(values).reshape(1, -1)
            sns.heatmap(heatmap_data, annot=True, fmt='d', cmap='YlOrRd', 
                        xticklabels=labels, yticklabels=["Count"], ax=ax_chart, cbar=True)
            ax_chart.set_title("Heatmap - Pedestrian Crossing Type Distribution", fontsize=14, fontweight="bold")
            ax_chart.set_xlabel("Pedestrian Crossing Type", fontsize=12, fontweight="bold")
            add_custom_legend(ax_chart, "heatmap", "Number of pedestrian crossing instances observed", "Pedestrian crossing types")

        elif chart == "radar":
            if total > 0:
                angles = np.linspace(0, 2 * pi, len(labels), endpoint=False).tolist()
                values_normalized = [(v / max(values)) * 100 if max(values) > 0 else 0 for v in values]
                values_normalized += values_normalized[:1]  # Complete the circle
                angles += angles[:1]  # Complete the circle
                
                ax_chart.plot(angles, values_normalized, 'o-', linewidth=2, color='red')
                ax_chart.fill(angles, values_normalized, alpha=0.25, color='red')
                ax_chart.set_xticks(angles[:-1])
                ax_chart.set_xticklabels(labels)
                ax_chart.set_ylim(0, 100)
                ax_chart.set_title("Radar Chart - Pedestrian Crossing Type Distribution", fontsize=14, fontweight="bold")
                ax_chart.grid(True)
            else:
                ax_chart.text(0.5, 0.5, "No Data Available", ha="center", va="center", 
                              fontsize=14, fontweight="bold", transform=ax_chart.transAxes)
            add_custom_legend(ax_chart, "radar", "Number of pedestrian crossing instances observed", "Pedestrian crossing types")

        elif chart == "histogram":
            if total > 0:
                bins = np.arange(len(labels) + 1) - 0.5
                ax_chart.hist([labels[i] for i in range(len(labels)) for _ in range(values[i])], 
                              bins=len(labels), color=colors, edgecolor='black', alpha=0.7)
                ax_chart.set_xticks(range(len(labels)))
                ax_chart.set_xticklabels(labels, rotation=45)
            else:
                ax_chart.bar(labels, values, color=colors)
                ax_chart.text(0.5, 0.5, "No Data Available", ha="center", va="center", 
                              fontsize=14, fontweight="bold", transform=ax_chart.transAxes)
            ax_chart.set_title("Histogram - Pedestrian Crossing Type Distribution", fontsize=14, fontweight="bold")
            ax_chart.set_xlabel("Pedestrian Crossing Type", fontsize=12, fontweight="bold")
            ax_chart.set_ylabel("Frequency", fontsize=12, fontweight="bold")
            add_custom_legend(ax_chart, "histogram", "Number of pedestrian crossing instances observed", "Pedestrian crossing types")

        elif chart == "stackedbar":
            ax_chart.bar(["Pedestrian Crossing"], [sum(values)], color='lightblue', 
                         label='Total', edgecolor='black')
            bottom = 0
            for i, (label, value) in enumerate(zip(labels, values)):
                if value > 0:
                    ax_chart.bar(["Pedestrian Crossing"], [value], bottom=bottom, 
                                 color=colors[i], label=label, edgecolor='black')
                    ax_chart.text(0, bottom + value/2, f"{label}: {value}", 
                                  ha="center", va="center", fontweight="bold", fontsize=10)
                bottom += value
            ax_chart.set_title("Stacked Bar Chart - Pedestrian Crossing Type Distribution", fontsize=14, fontweight="bold")
            ax_chart.set_ylabel("Occurrence Count", fontsize=12, fontweight="bold")
            ax_chart.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            add_custom_legend(ax_chart, "stackedbar", "Number of pedestrian crossing instances observed", "Pedestrian crossing types")

        elif chart == "scatter":
            x_pos = np.arange(len(labels))
            ax_chart.scatter(x_pos, values, c=colors, s=200, alpha=0.7, edgecolors='black')
            for i, (x, y) in enumerate(zip(x_pos, values)):
                ax_chart.annotate(f"{labels[i]}\n({y})", (x, y), textcoords="offset points", 
                                  xytext=(0,10), ha='center', fontweight='bold')
            ax_chart.set_xticks(x_pos)
            ax_chart.set_xticklabels(labels)
            ax_chart.set_title("Scatter Plot - Pedestrian Crossing Type Distribution", fontsize=14, fontweight="bold")
            ax_chart.set_xlabel("Pedestrian Crossing Type", fontsize=12, fontweight="bold")
            ax_chart.set_ylabel("Occurrence Count", fontsize=12, fontweight="bold")
            ax_chart.grid(True, alpha=0.3)
            add_custom_legend(ax_chart, "scatter", "Number of pedestrian crossing instances observed", "Pedestrian crossing types")

        elif chart == "density":
            if total > 0:
                # Create data points for density plot
                data_points = []
                for i, (label, value) in enumerate(zip(labels, values)):
                    data_points.extend([i] * value)
                
                if data_points:
                    ax_chart.hist(data_points, bins=len(labels), density=True, alpha=0.7, 
                                  color='skyblue', edgecolor='black')
                    ax_chart.set_xticks(range(len(labels)))
                    ax_chart.set_xticklabels(labels)
                else:
                    ax_chart.text(0.5, 0.5, "No Data Available", ha="center", va="center", 
                                  fontsize=14, fontweight="bold", transform=ax_chart.transAxes)
            else:
                ax_chart.text(0.5, 0.5, "No Data Available", ha="center", va="center", 
                              fontsize=14, fontweight="bold", transform=ax_chart.transAxes)
            ax_chart.set_title("Density Plot - Pedestrian Crossing Type Distribution", fontsize=14, fontweight="bold")
            ax_chart.set_xlabel("Pedestrian Crossing Type", fontsize=12, fontweight="bold")
            ax_chart.set_ylabel("Density", fontsize=12, fontweight="bold")
            add_custom_legend(ax_chart, "density", "Number of pedestrian crossing instances observed", "Pedestrian crossing types")

        # Add statistical summary to the second subplot
        add_stats(axes[1])

        plt.tight_layout()
        filename = f"pedestrian_road_crossing_{chart}_plot.png"
        filepath = os.path.join(output_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"✅ {chart.capitalize()} chart saved as {filepath}")
        plt.show()

    print(f"\n📊 All selected charts have been generated and saved to {output_dir}")
    print("📈 Statistical summary includes:")
    print("  • Total crossing instances and distribution percentages")
    print("  • Most/least common crossing types")
    print("  • Statistical measures (mean, std dev, range)")
