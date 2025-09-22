import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os
from math import pi

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

def plot_pedestrian_density_road_types(pedestrian_data, output_dir="figures/exploratory"):
    """
    Plot pedestrian density across road types with multiple chart options.
    
    Args:
        pedestrian_data: Dictionary with pedestrian counts for different road types
        output_dir: Directory to save the plots
    """
    os.makedirs(output_dir, exist_ok=True)

    # --- Ensure 5 fixed labels are always present ---
    expected_labels = ["Narrow", "Highway", "OneWay", "OffRoad", "City Road"]
    cleaned = {label: pedestrian_data.get(label, 0) for label in expected_labels}

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

    stats_text = f"""Pedestrian Density across Road Types Summary

Total Frames: {total}
Road Types: {len(labels)}

Highest Density: {most_common} ({max_val})
Lowest Density: {least_common} ({min_val})

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
        fig.suptitle(f"{chart.capitalize()} Chart with Statistical Summary - Pedestrian Density across Road Types",
                     fontsize=16, fontweight="bold", y=0.98)
        ax_chart = axes[0]

        if chart == "bar":
            bars = ax_chart.bar(labels, values, color=colors, edgecolor="black", linewidth=1.2)
            for bar, val in zip(bars, values):
                ax_chart.text(bar.get_x()+bar.get_width()/2, val, f"{val}",
                              ha="center", va="bottom", fontweight="bold", fontsize=10)
            ax_chart.set_title("Bar Chart - Pedestrian Density by Road Type", fontsize=14, fontweight="bold")
            ax_chart.set_xlabel("Road Type Categories", fontsize=12, fontweight="bold")
            ax_chart.set_ylabel("Frame Count", fontsize=12, fontweight="bold")
            ax_chart.grid(True, alpha=0.3)
            # Rotate x-axis labels for better readability
            ax_chart.tick_params(axis='x', rotation=45)
            add_custom_legend(ax_chart, "bar", "Number of pedestrian instances observed", "Road type categories")

        elif chart == "pie":
            ax_chart.pie(values, labels=labels, autopct="%1.1f%%",
                         startangle=90, colors=colors, explode=[0.05]*len(labels))
            ax_chart.set_title("Pie Chart - Pedestrian Distribution by Road Type", fontsize=14, fontweight="bold")
            add_custom_legend(ax_chart, "pie", "Number of pedestrian instances observed", "Road type categories")

        elif chart == "donut":
            wedges, texts, autotexts = ax_chart.pie(values, labels=labels,
                                                    startangle=90, colors=colors,
                                                    pctdistance=0.85, explode=[0.02]*len(labels),
                                                    autopct=lambda p: f"{int(round(p/100.*total))}")
            centre_circle = plt.Circle((0, 0), 0.70, fc="white")
            ax_chart.add_artist(centre_circle)
            ax_chart.text(0, 0, f"Total\n{total}", ha="center", va="center",
                          fontsize=12, fontweight="bold", color="darkblue")
            ax_chart.set_title("Donut Chart - Pedestrian Density by Road Type", fontsize=14, fontweight="bold")
            add_custom_legend(ax_chart, "donut", "Number of pedestrian instances observed", "Road type categories")

        elif chart == "heatmap":
            sns.heatmap(np.array([values]), annot=True, fmt="d", cmap="YlGnBu",
                        xticklabels=labels, yticklabels=["Frame Count"], ax=ax_chart,
                        cbar_kws={'label': 'Count'})
            ax_chart.set_title("Heat Map - Pedestrian Density Intensity", fontsize=14, fontweight="bold")
            ax_chart.tick_params(axis='x', rotation=45)
            add_custom_legend(ax_chart, "heatmap", "Number of pedestrian instances observed", "Road type categories")

        elif chart == "radar":
            fig.clf()
            fig, axes = plt.subplots(1, 2, figsize=(16, 7))
            ax_chart = plt.subplot(1, 2, 1, projection="polar")
            num = len(labels)
            angles = [n/float(num) * 2*np.pi for n in range(num)]
            angles += angles[:1]
            radar_values = values + values[:1]
            ax_chart.plot(angles, radar_values, "o-", linewidth=2, color='darkgreen', markersize=8)
            ax_chart.fill(angles, radar_values, alpha=0.25, color='lightgreen')
            ax_chart.set_xticks(angles[:-1])
            ax_chart.set_xticklabels(labels)
            ax_chart.set_title("Radar Chart - Pedestrian Density by Road Type", fontsize=14, fontweight="bold")
            add_custom_legend(ax_chart, "radar", "Number of pedestrian instances observed", "Road type categories")

        elif chart == "histogram":
            ax_chart.hist(values, bins=max(6, len(set(values))), color="lightblue", edgecolor="black", alpha=0.7)
            ax_chart.set_title("Histogram - Frame Counts per Road Type", fontsize=14, fontweight="bold")
            ax_chart.set_xlabel("Frame Counts", fontsize=12, fontweight="bold")
            ax_chart.set_ylabel("Frequency", fontsize=12, fontweight="bold")
            ax_chart.grid(True, alpha=0.3)
            add_custom_legend(ax_chart, "histogram", "Frequency of occurrence", "Frame count ranges")

        elif chart == "stackedbar":
            ax_chart.bar(labels, values, color="teal", label="Total Count", alpha=0.8)
            ax_chart.bar(labels, [v/2 for v in values], color="coral", label="50% Mark", alpha=0.6)
            ax_chart.set_title("Stacked Bar Chart - Pedestrian Density by Road Type", fontsize=14, fontweight="bold")
            ax_chart.set_xlabel("Road Type Categories", fontsize=12, fontweight="bold")
            ax_chart.set_ylabel("Frame Count", fontsize=12, fontweight="bold")
            ax_chart.legend()
            ax_chart.grid(True, alpha=0.3)
            ax_chart.tick_params(axis='x', rotation=45)
            add_custom_legend(ax_chart, "stackedbar", "Number of pedestrian instances observed", "Road type categories")

        elif chart == "scatter":
            x_positions = range(len(labels))
            ax_chart.scatter(x_positions, values, color="darkred", s=150, alpha=0.7, edgecolors='black')
            ax_chart.set_xticks(x_positions)
            ax_chart.set_xticklabels(labels, rotation=45)
            ax_chart.set_title("Scatter Plot - Pedestrian Density by Road Type", fontsize=14, fontweight="bold")
            ax_chart.set_xlabel("Road Type Categories", fontsize=12, fontweight="bold")
            ax_chart.set_ylabel("Frame Count", fontsize=12, fontweight="bold")
            ax_chart.grid(True, alpha=0.3)
            add_custom_legend(ax_chart, "scatter", "Number of pedestrian instances observed", "Road type categories")

        elif chart == "density":
            if len(set(values)) > 1:  # Only plot if there's variation in data
                sns.kdeplot(values, fill=True, color="purple", ax=ax_chart, alpha=0.7)
                ax_chart.set_title("Density Plot - Frame Count Distribution", fontsize=14, fontweight="bold")
                ax_chart.set_xlabel("Frame Count", fontsize=12, fontweight="bold")
                ax_chart.set_ylabel("Density", fontsize=12, fontweight="bold")
                add_custom_legend(ax_chart, "density", "Number of pedestrian instances observed", "Frame count distribution")
            else:
                ax_chart.text(0.5, 0.5, "Insufficient variation\nfor density plot", 
                             ha="center", va="center", transform=ax_chart.transAxes,
                             fontsize=12, bbox=dict(boxstyle="round", facecolor="yellow", alpha=0.5))
                ax_chart.set_title("Density Plot - Frame Count Distribution", fontsize=14, fontweight="bold")

        add_stats(axes[1])

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        outpath = os.path.join(output_dir, f"pedestrian_density_road_types_{chart}_with_stats.png")
        plt.savefig(outpath, dpi=300, bbox_inches="tight", facecolor="white")
        plt.close()
        print(f"✅ Saved: {outpath}")

    print(f"\n🎉 All selected charts saved to: {output_dir}")

if __name__ == "__main__":
    print("🚶 Pedestrian Density across Road Types EDA Plotting Tool")
    print("=" * 60)
    print("This module is designed to be imported and used with real nuScenes data.")
    print("Use the main.py file or test scripts to run the complete analysis.")
