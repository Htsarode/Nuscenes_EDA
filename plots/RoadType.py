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

def plot_road_type_distribution(road_types, output_dir="figures/exploratory"):
    os.makedirs(output_dir, exist_ok=True)

    # Expected road types (force all labels present)
    expected_labels = ["Narrow", "Highway", "OneWay", "OffRoad", "City Road", "Parking lot"]
    cleaned = {label: road_types.get(label, 0) for label in expected_labels}

    labels = list(cleaned.keys())
    values = list(cleaned.values())
    total_frames = sum(values)

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
        print(f"{num}. {name}")
    chosen = input("Enter chart numbers (comma-separated): ").replace(" ", "").split(",")
    selected_charts = [chart_map[c] for c in chosen if c in chart_map]

    if not selected_charts:
        print("⚠️ No valid chart selected. Exiting.")
        return

    colors = list(plt.cm.Set2(np.linspace(0, 1, len(labels))))

    # --- Statistical Summary ---
    mean_frames = np.mean(values)
    std_frames = np.std(values)
    max_frames = max(values)
    min_frames = min(values)
    most_common = labels[values.index(max_frames)]
    least_common = labels[values.index(min_frames)]

    stats_text = f"""Statistical Summary

Total Frames: {total_frames}
Road Types: {len(labels)}

Most Common: {most_common} ({max_frames} frames)
Least Common: {least_common} ({min_frames} frames)

Mean: {mean_frames:.1f} frames
Std Dev: {std_frames:.1f} frames
Range: {max_frames - min_frames} frames

Distribution:"""
    for label, value in zip(labels, values):
        percentage = (value / total_frames * 100) if total_frames > 0 else 0
        stats_text += f"\n• {label}: {value} ({percentage:.1f}%)"

    def add_stats(ax):
        ax.axis("off")
        ax.text(0.05, 0.95, stats_text, transform=ax.transAxes,
                fontsize=11, verticalalignment="top", fontfamily="monospace",
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))
        ax.set_title("Statistical Summary", fontsize=14, fontweight="bold")

    # --- Generate chosen charts ---
    for chart in selected_charts:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle(f"{chart.capitalize()} Chart with Statistical Summary - Road Types",
                     fontsize=16, fontweight="bold", y=0.98)
        ax_chart = axes[0]

        if chart == "bar":
            bars = ax_chart.bar(labels, values, color=colors, edgecolor="black")
            ax_chart.set_title("Bar Chart - Road Types")
            ax_chart.set_ylabel("Frame Count")
            add_custom_legend(ax_chart, "bar", 
                            "Number of road frames/instances observed",
                            "Road types and infrastructure categories")
            for bar in bars:
                h = bar.get_height()
                ax_chart.text(bar.get_x()+bar.get_width()/2, h, f"{int(h)}",
                              ha="center", va="bottom", fontweight="bold")

        elif chart == "pie":
            ax_chart.pie(values, labels=labels, autopct="%1.1f%%",
                         startangle=90, colors=colors, explode=[0.05]*len(labels))
            ax_chart.set_title("Pie Chart - Percentage Distribution")
            add_custom_legend(ax_chart, "pie", 
                            "Number of road frames/instances observed",
                            "Road types and infrastructure categories")

        elif chart == "donut":
            wedges, texts, autotexts = ax_chart.pie(values, labels=labels,
                                                    startangle=90, colors=colors,
                                                    pctdistance=0.85, explode=[0.02]*len(labels),
                                                    autopct=lambda p: f"{int(round(p/100.*total_frames))}")
            centre_circle = plt.Circle((0, 0), 0.70, fc="white")
            ax_chart.add_artist(centre_circle)
            ax_chart.text(0, 0, f"Total\nFrames\n{total_frames}", ha="center", va="center",
                          fontsize=12, fontweight="bold", color="darkblue")
            ax_chart.set_title("Donut Chart - Frame Counts")

        elif chart == "heatmap":
            sns.heatmap(np.array([values]), annot=True, fmt="d", cmap="Oranges",
                        xticklabels=labels, yticklabels=["Frames"], ax=ax_chart)
            ax_chart.set_title("Heat Map - Road Type Intensity")

        elif chart == "radar":
            fig.clf()
            fig, axes = plt.subplots(1, 2, figsize=(14, 6))
            ax_chart = plt.subplot(1, 2, 1, projection="polar")
            num_types = len(labels)
            angles = [n/float(num_types) * 2*np.pi for n in range(num_types)]
            angles += angles[:1]
            radar_values = values + values[:1]
            ax_chart.plot(angles, radar_values, "o-", linewidth=2)
            ax_chart.fill(angles, radar_values, alpha=0.25)
            ax_chart.set_xticks(angles[:-1])
            ax_chart.set_xticklabels(labels)
            ax_chart.set_title("Radar Chart - Road Type Distribution")

        elif chart == "histogram":
            ax_chart.hist(values, bins=6, color="skyblue", edgecolor="black")
            ax_chart.set_title("Histogram - Frame Counts")

        elif chart == "stackedbar":
            ax_chart.bar(labels, values, color="steelblue", label="Frames")
            ax_chart.bar(labels, [v/2 for v in values], color="orange", label="Half Frames")
            ax_chart.set_title("Stacked Bar Chart")
            ax_chart.legend()

        elif chart == "scatter":
            ax_chart.scatter(labels, values, color="purple", s=100)
            ax_chart.set_title("Scatter Plot - Road Types")

        elif chart == "density":
            sns.kdeplot(values, fill=True, color="green", ax=ax_chart)
            ax_chart.set_title("Density Plot - Frame Distribution")

        add_stats(axes[1])

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        outpath = os.path.join(output_dir, f"roadtype_{chart}_with_stats.png")
        plt.savefig(outpath, dpi=300, bbox_inches="tight", facecolor="white")
        plt.close()
        print(f"✅ Saved: {outpath}")
