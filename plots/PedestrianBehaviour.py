import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os
from math import pi

def add_custom_legend(ax, chart_type, y_axis_motive=None, x_axis_meaning=None):
    """Add custom legend explaining axis meanings"""
    legend_text = []
    
    # Add y-axis motive only for bar, histogram, and stacked bar charts
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

def plot_pedestrian_behaviour(behaviour_data, output_dir="figures/exploratory"):
    """
    Plot pedestrian behaviour (Standing, Walking, Running) with multiple chart options.
    Args:
        behaviour_data: Dictionary with pedestrian behaviour counts
        output_dir: Directory to save the plots
    """
    os.makedirs(output_dir, exist_ok=True)

    expected_labels = ["Standing", "Walking", "Running"]
    cleaned = {label: behaviour_data.get(label, 0) for label in expected_labels}
    labels = list(cleaned.keys())
    values = list(cleaned.values())
    total = sum(values)

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

    colors = list(plt.cm.Pastel2(np.linspace(0, 1, len(labels))))

    mean_val = np.mean(values)
    std_val = np.std(values)
    max_val = max(values)
    min_val = min(values)
    most_common = labels[values.index(max_val)]
    least_common = labels[values.index(min_val)]

    stats_text = f"""Pedestrian Behaviour Summary\n\nTotal Pedestrians: {total}\nActivity Types: {len(labels)}\n\nMost Common: {most_common} ({max_val})\nLeast Common: {least_common} ({min_val})\n\nMean: {mean_val:.1f}\nStd Dev: {std_val:.1f}\nRange: {max_val - min_val}\n\nDistribution:"""
    for label, value in zip(labels, values):
        pct = (value / total * 100) if total > 0 else 0
        stats_text += f"\n• {label}: {value} ({pct:.1f}%)"

    def add_stats(ax):
        ax.axis("off")
        ax.text(0.05, 0.95, stats_text, transform=ax.transAxes,
                fontsize=11, verticalalignment="top", fontfamily="monospace",
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.8))
        ax.set_title("Statistical Summary", fontsize=14, fontweight="bold")

    for chart in selected_charts:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle(f"{chart.capitalize()} Chart with Statistical Summary - Pedestrian Behaviour",
                     fontsize=16, fontweight="bold", y=0.98)
        ax_chart = axes[0]

        if chart == "bar":
            bars = ax_chart.bar(labels, values, color=colors, edgecolor="black", linewidth=1.2)
            for bar, val in zip(bars, values):
                ax_chart.text(bar.get_x()+bar.get_width()/2, val, f"{val}",
                              ha="center", va="bottom", fontweight="bold", fontsize=10)
            ax_chart.set_title("Bar Chart - Pedestrian Behaviour", fontsize=14, fontweight="bold")
            ax_chart.set_xlabel("Activity Type", fontsize=12, fontweight="bold")
            ax_chart.set_ylabel("Pedestrian Count", fontsize=12, fontweight="bold")
            ax_chart.grid(True, alpha=0.3)
            add_custom_legend(ax_chart, "bar", 
                            "Number of pedestrian instances observed",
                            "Pedestrian behavior types/categories")

        elif chart == "pie":
            ax_chart.pie(values, labels=labels, autopct="%1.1f%%",
                         startangle=90, colors=colors, explode=[0.05]*len(labels))
            ax_chart.set_title("Pie Chart - Pedestrian Behaviour", fontsize=14, fontweight="bold")
            add_custom_legend(ax_chart, "pie", 
                            "Number of pedestrian instances observed",
                            "Pedestrian behavior types/categories")

        elif chart == "donut":
            wedges, texts, autotexts = ax_chart.pie(values, labels=labels,
                                                    startangle=90, colors=colors,
                                                    pctdistance=0.85, explode=[0.02]*len(labels),
                                                    autopct=lambda p: f"{int(round(p/100.*total))}")
            centre_circle = plt.Circle((0, 0), 0.70, fc="white")
            ax_chart.add_artist(centre_circle)
            ax_chart.text(0, 0, f"Total\n{total}", ha="center", va="center",
                          fontsize=12, fontweight="bold", color="darkblue")
            ax_chart.set_title("Donut Chart - Pedestrian Behaviour", fontsize=14, fontweight="bold")
            add_custom_legend(ax_chart, "donut", 
                            "Number of pedestrian instances observed",
                            "Pedestrian behavior types/categories")

        elif chart == "heatmap":
            sns.heatmap(np.array([values]), annot=True, fmt="d", cmap="YlOrRd",
                        xticklabels=labels, yticklabels=["Pedestrian Count"], ax=ax_chart,
                        cbar_kws={'label': 'Count'})
            ax_chart.set_title("Heat Map - Pedestrian Behaviour Intensity", fontsize=14, fontweight="bold")
            add_custom_legend(ax_chart, "heatmap", 
                            "Number of pedestrian instances observed",
                            "Pedestrian behavior types/categories")

        elif chart == "radar":
            fig.clf()
            fig, axes = plt.subplots(1, 2, figsize=(14, 6))
            ax_chart = plt.subplot(1, 2, 1, projection="polar")
            num = len(labels)
            angles = [n/float(num) * 2*np.pi for n in range(num)]
            angles += angles[:1]
            radar_values = values + values[:1]
            ax_chart.plot(angles, radar_values, "o-", linewidth=2, color='darkorange', markersize=8)
            ax_chart.fill(angles, radar_values, alpha=0.25, color='gold')
            ax_chart.set_xticks(angles[:-1])
            ax_chart.set_xticklabels(labels)
            ax_chart.set_title("Radar Chart - Pedestrian Behaviour", fontsize=14, fontweight="bold")
            add_custom_legend(ax_chart, "radar", 
                            "Number of pedestrian instances observed",
                            "Pedestrian behavior types/categories")

        elif chart == "histogram":
            ax_chart.hist(values, bins=max(3, len(set(values))), color="lightblue", edgecolor="black", alpha=0.7)
            ax_chart.set_title("Histogram - Pedestrian Behaviour", fontsize=14, fontweight="bold")
            ax_chart.set_xlabel("Pedestrian Count", fontsize=12, fontweight="bold")
            ax_chart.set_ylabel("Frequency", fontsize=12, fontweight="bold")
            ax_chart.grid(True, alpha=0.3)
            add_custom_legend(ax_chart, "histogram", 
                            "Number of pedestrian instances observed",
                            "Pedestrian behavior types/categories")

        elif chart == "stackedbar":
            ax_chart.bar(labels, values, color="orange", label="Total Count", alpha=0.8)
            ax_chart.bar(labels, [v/2 for v in values], color="lightgreen", label="50% Mark", alpha=0.6)
            ax_chart.set_title("Stacked Bar Chart - Pedestrian Behaviour", fontsize=14, fontweight="bold")
            ax_chart.set_xlabel("Activity Type", fontsize=12, fontweight="bold")
            ax_chart.set_ylabel("Pedestrian Count", fontsize=12, fontweight="bold")
            ax_chart.legend()
            ax_chart.grid(True, alpha=0.3)
            add_custom_legend(ax_chart, "stackedbar", 
                            "Number of pedestrian instances observed",
                            "Pedestrian behavior types/categories")

        elif chart == "scatter":
            x_positions = range(len(labels))
            ax_chart.scatter(x_positions, values, color="purple", s=150, alpha=0.7, edgecolors='black')
            ax_chart.set_xticks(x_positions)
            ax_chart.set_xticklabels(labels)
            ax_chart.set_title("Scatter Plot - Pedestrian Behaviour", fontsize=14, fontweight="bold")
            ax_chart.set_xlabel("Activity Type", fontsize=12, fontweight="bold")
            ax_chart.set_ylabel("Pedestrian Count", fontsize=12, fontweight="bold")
            ax_chart.grid(True, alpha=0.3)
            add_custom_legend(ax_chart, "scatter", 
                            "Number of pedestrian instances observed",
                            "Pedestrian behavior types/categories")

        elif chart == "density":
            if len(set(values)) > 1:
                sns.kdeplot(values, fill=True, color="green", ax=ax_chart, alpha=0.7)
                ax_chart.set_title("Density Plot - Pedestrian Behaviour", fontsize=14, fontweight="bold")
                ax_chart.set_xlabel("Pedestrian Count", fontsize=12, fontweight="bold")
                ax_chart.set_ylabel("Density", fontsize=12, fontweight="bold")
            else:
                ax_chart.text(0.5, 0.5, "Insufficient variation\nfor density plot", 
                             ha="center", va="center", transform=ax_chart.transAxes,
                             fontsize=12, bbox=dict(boxstyle="round", facecolor="yellow", alpha=0.5))
                ax_chart.set_title("Density Plot - Pedestrian Behaviour", fontsize=14, fontweight="bold")
            add_custom_legend(ax_chart, "density", 
                            "Number of pedestrian instances observed",
                            "Pedestrian behavior types/categories")

        add_stats(axes[1])

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        outpath = os.path.join(output_dir, f"pedestrian_behaviour_{chart}_with_stats.png")
        plt.savefig(outpath, dpi=300, bbox_inches="tight", facecolor="white")
        plt.close()
        print(f"✅ Saved: {outpath}")

    print(f"\n🎉 All selected charts saved to: {output_dir}")
