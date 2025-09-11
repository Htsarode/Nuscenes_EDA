import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os
from math import pi
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def plot_vehicle_class(vehicle_data, output_dir="figures/exploratory"):
    """
    Plot vehicle class distribution with multiple chart options.
    
    Args:
        vehicle_data: Dictionary with vehicle class counts
        output_dir: Directory to save the plots
    """
    os.makedirs(output_dir, exist_ok=True)

    # --- Ensure 5 fixed labels are always present ---
    expected_labels = ["Car", "Bus", "Truck", "Van", "Trailer"]
    cleaned = {label: vehicle_data.get(label, 0) for label in expected_labels}

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

    colors = list(plt.cm.Set3(np.linspace(0, 1, len(labels))))

    # --- Statistical Summary ---
    mean_val = np.mean(values)
    std_val = np.std(values)
    max_val = max(values)
    min_val = min(values)
    most_common = labels[values.index(max_val)]
    least_common = labels[values.index(min_val)]

    stats_text = f"""Vehicle Class Distribution Summary

Total Vehicles: {total}
Classes: {len(labels)}

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
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgreen", alpha=0.8))
        ax.set_title("Statistical Summary", fontsize=14, fontweight="bold")

    # --- Generate chosen charts ---
    for chart in selected_charts:
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))
        fig.suptitle(f"{chart.capitalize()} Chart with Statistical Summary - Vehicle Class Distribution",
                     fontsize=16, fontweight="bold", y=0.98)
        ax_chart = axes[0]

        if chart == "bar":
            bars = ax_chart.bar(labels, values, color=colors, edgecolor="black", linewidth=1.2)
            for bar, val in zip(bars, values):
                ax_chart.text(bar.get_x()+bar.get_width()/2, val, f"{val}",
                              ha="center", va="bottom", fontweight="bold", fontsize=10)
            ax_chart.set_title("Bar Chart - Vehicle Class Counts", fontsize=14, fontweight="bold")
            ax_chart.set_xlabel("Vehicle Class", fontsize=12, fontweight="bold")
            ax_chart.set_ylabel("Vehicle Count", fontsize=12, fontweight="bold")
            ax_chart.grid(True, alpha=0.3)

        elif chart == "pie":
            ax_chart.pie(values, labels=labels, autopct="%1.1f%%",
                         startangle=90, colors=colors, explode=[0.05]*len(labels))
            ax_chart.set_title("Pie Chart - Vehicle Class Percentage Distribution", fontsize=14, fontweight="bold")

        elif chart == "donut":
            wedges, texts, autotexts = ax_chart.pie(values, labels=labels,
                                                    startangle=90, colors=colors,
                                                    pctdistance=0.85, explode=[0.02]*len(labels),
                                                    autopct=lambda p: f"{int(round(p/100.*total))}")
            centre_circle = plt.Circle((0, 0), 0.70, fc="white")
            ax_chart.add_artist(centre_circle)
            ax_chart.text(0, 0, f"Total\n{total}", ha="center", va="center",
                          fontsize=12, fontweight="bold", color="darkblue")
            ax_chart.set_title("Donut Chart - Vehicle Class Counts", fontsize=14, fontweight="bold")

        elif chart == "heatmap":
            sns.heatmap(np.array([values]), annot=True, fmt="d", cmap="YlOrRd",
                        xticklabels=labels, yticklabels=["Vehicle Count"], ax=ax_chart,
                        cbar_kws={'label': 'Count'})
            ax_chart.set_title("Heat Map - Vehicle Class Intensity", fontsize=14, fontweight="bold")

        elif chart == "radar":
            fig.clf()
            fig, axes = plt.subplots(1, 2, figsize=(16, 7))
            ax_chart = plt.subplot(1, 2, 1, projection="polar")
            num = len(labels)
            angles = [n/float(num) * 2*np.pi for n in range(num)]
            angles += angles[:1]
            radar_values = values + values[:1]
            ax_chart.plot(angles, radar_values, "o-", linewidth=2, color='darkblue', markersize=8)
            ax_chart.fill(angles, radar_values, alpha=0.25, color='lightblue')
            ax_chart.set_xticks(angles[:-1])
            ax_chart.set_xticklabels(labels)
            ax_chart.set_title("Radar Chart - Vehicle Class Distribution", fontsize=14, fontweight="bold")

        elif chart == "histogram":
            ax_chart.hist(values, bins=max(6, len(set(values))), color="skyblue", edgecolor="black", alpha=0.7)
            ax_chart.set_title("Histogram - Vehicle Counts per Class", fontsize=14, fontweight="bold")
            ax_chart.set_xlabel("Vehicle Counts", fontsize=12, fontweight="bold")
            ax_chart.set_ylabel("Frequency", fontsize=12, fontweight="bold")
            ax_chart.grid(True, alpha=0.3)

        elif chart == "stackedbar":
            ax_chart.bar(labels, values, color="steelblue", label="Total Count", alpha=0.8)
            ax_chart.bar(labels, [v/2 for v in values], color="orange", label="50% Mark", alpha=0.6)
            ax_chart.set_title("Stacked Bar Chart - Vehicle Classes", fontsize=14, fontweight="bold")
            ax_chart.set_xlabel("Vehicle Class", fontsize=12, fontweight="bold")
            ax_chart.set_ylabel("Vehicle Count", fontsize=12, fontweight="bold")
            ax_chart.legend()
            ax_chart.grid(True, alpha=0.3)

        elif chart == "scatter":
            x_positions = range(len(labels))
            ax_chart.scatter(x_positions, values, color="purple", s=150, alpha=0.7, edgecolors='black')
            ax_chart.set_xticks(x_positions)
            ax_chart.set_xticklabels(labels)
            ax_chart.set_title("Scatter Plot - Vehicle Classes", fontsize=14, fontweight="bold")
            ax_chart.set_xlabel("Vehicle Class", fontsize=12, fontweight="bold")
            ax_chart.set_ylabel("Vehicle Count", fontsize=12, fontweight="bold")
            ax_chart.grid(True, alpha=0.3)

        elif chart == "density":
            if len(set(values)) > 1:  # Only plot if there's variation in data
                sns.kdeplot(values, fill=True, color="green", ax=ax_chart, alpha=0.7)
                ax_chart.set_title("Density Plot - Vehicle Count Distribution", fontsize=14, fontweight="bold")
                ax_chart.set_xlabel("Vehicle Count", fontsize=12, fontweight="bold")
                ax_chart.set_ylabel("Density", fontsize=12, fontweight="bold")
            else:
                ax_chart.text(0.5, 0.5, "Insufficient variation\nfor density plot", 
                             ha="center", va="center", transform=ax_chart.transAxes,
                             fontsize=12, bbox=dict(boxstyle="round", facecolor="yellow", alpha=0.5))
                ax_chart.set_title("Density Plot - Vehicle Count Distribution", fontsize=14, fontweight="bold")

        add_stats(axes[1])

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        outpath = os.path.join(output_dir, f"vehicle_class_{chart}_with_stats.png")
        plt.savefig(outpath, dpi=300, bbox_inches="tight", facecolor="white")
        plt.close()
        print(f"✅ Saved: {outpath}")

    print(f"\n🎉 All selected charts saved to: {output_dir}")

if __name__ == "__main__":
    print("🚗 Vehicle Class EDA Plotting Tool")
    print("=" * 40)
    print("This module is designed to be imported and used with real nuScenes data.")
    print("Use the main.py file or test scripts to run the complete analysis.")
