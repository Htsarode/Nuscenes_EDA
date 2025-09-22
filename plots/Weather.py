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
 
def plot_weather_distribution(weather_conditions, output_dir="figures/exploratory"):
    os.makedirs(output_dir, exist_ok=True)
 
    # --- Clean and normalize weather data ---
    normalized = {k.capitalize(): v for k, v in weather_conditions.items()}
 
    # Now include Unknown as a plotted category
    expected_weathers = ["Sunny", "Rainy", "Snow", "Clear", "Foggy", "Overcast", "Sleet", "Unknown"]
    cleaned = {w: normalized.get(w, 0) for w in expected_weathers}
 
    labels = expected_weathers
    values = [cleaned[w] for w in expected_weathers]
 
    total_scenes = sum(values)  # now includes Unknown in total
 
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
 
    # Colors (make Unknown grey)
    colors = list(plt.cm.Set3(np.linspace(0, 1, len(labels)-1))) + ["lightgrey"]
 
    # --- Statistical Summary Text ---
    mean_scenes = np.mean(values)
    std_scenes = np.std(values)
    max_scenes = max(values)
    min_scenes = min(values)
    most_common = labels[values.index(max_scenes)]
    least_common = labels[values.index(min_scenes)]
 
    stats_text = f"""Statistical Summary
 
Total Scenes: {total_scenes}
Weather Conditions: {len(labels)}
 
Most Common: {most_common} ({max_scenes} scenes)
Least Common: {least_common} ({min_scenes} scenes)
 
Mean: {mean_scenes:.1f} scenes
Std Dev: {std_scenes:.1f} scenes
Range: {max_scenes - min_scenes} scenes
 
Weather Distribution:"""
    for label, value in zip(labels, values):
        percentage = (value / total_scenes * 100) if total_scenes > 0 else 0
        stats_text += f"\n• {label}: {value} ({percentage:.1f}%)"
 
    def add_stats(ax):
        ax.axis('off')
        ax.text(0.05, 0.95, stats_text, transform=ax.transAxes,
                fontsize=11, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.8))
        ax.set_title("Statistical Summary", fontsize=14, fontweight="bold")
 
    # --- Generate chosen charts ---
    for chart in selected_charts:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle(f"{chart.capitalize()} Chart with Statistical Summary - nuScenes Dataset",
                     fontsize=16, fontweight="bold", y=0.98)
        ax_chart = axes[0]
 
        if chart == "bar":
            bars = ax_chart.bar(labels, values, color=colors, edgecolor="black", linewidth=1.2)
            for bar in bars:
                h = bar.get_height()
                ax_chart.text(bar.get_x()+bar.get_width()/2, h, f"{int(h)}",
                              ha='center', va='bottom', fontweight='bold')
            ax_chart.set_title("Bar Chart - Scene Counts")
            ax_chart.set_ylabel("Number of Scenes")
            ax_chart.grid(axis="y", linestyle="--", alpha=0.7)
            add_custom_legend(ax_chart, "bar", "Number of scenes/frames observed", "Weather conditions in dataset")

        elif chart == "pie":
            wedges, texts, autotexts = ax_chart.pie(values, labels=labels,
                                                    autopct="%1.1f%%", startangle=90,
                                                    colors=colors, explode=[0.05]*len(labels))
            ax_chart.set_title("Pie Chart - Percentage Distribution")
            for autotext in autotexts:
                autotext.set_color('white')
                autotext.set_fontweight('bold')
            add_custom_legend(ax_chart, "pie", None, "Weather conditions distribution")

        elif chart == "donut":
            wedges, texts, autotexts = ax_chart.pie(values, labels=labels,
                                                    startangle=90, colors=colors,
                                                    pctdistance=0.85, explode=[0.02]*len(labels),
                                                    autopct=lambda p: f"{int(round(p/100.*total_scenes))}")
            centre_circle = plt.Circle((0, 0), 0.70, fc="white")
            ax_chart.add_artist(centre_circle)
            ax_chart.text(0, 0, f'Total\nScenes\n{total_scenes}', ha='center', va='center',
                          fontsize=12, fontweight='bold', color='darkblue')
            ax_chart.set_title("Donut Chart - Scene Counts")
            add_custom_legend(ax_chart, "donut", None, "Weather conditions with scene counts")

        elif chart == "heatmap":
            data = np.array(values).reshape(1, -1)
            sns.heatmap(data, annot=True, fmt="d", cmap="YlOrRd",
                        xticklabels=labels, yticklabels=["Scene Count"],
                        ax=ax_chart, cbar_kws={'label': 'Number of Scenes'})
            ax_chart.set_title("Heat Map - Weather Intensity")
            ax_chart.tick_params(axis='x', rotation=45)
            add_custom_legend(ax_chart, "heatmap", None, "Weather conditions intensity")

        elif chart == "radar":
            fig.clf()
            fig, axes = plt.subplots(1, 2, figsize=(14, 6))
            ax_chart = plt.subplot(1, 2, 1, projection='polar')
            num_conditions = len(labels)
            angles = [n / float(num_conditions) * 2 * pi for n in range(num_conditions)]
            angles += angles[:1]
            radar_values = values + values[:1]
            ax_chart.plot(angles, radar_values, 'o-', linewidth=3, color='darkred')
            ax_chart.fill(angles, radar_values, alpha=0.25, color='red')
            ax_chart.set_xticks(angles[:-1])
            ax_chart.set_xticklabels(labels, fontsize=11, fontweight='bold')
            ax_chart.set_title("Radar Chart - Weather Distribution")
            add_custom_legend(ax_chart, "radar", None, "Weather conditions radar plot")

        elif chart == "histogram":
            ax_chart.hist(values, bins=5, color='skyblue', edgecolor='black')
            ax_chart.set_title("Histogram - Frequency of Scene Counts")
            ax_chart.set_xlabel("Scene Counts")
            ax_chart.set_ylabel("Frequency")
            add_custom_legend(ax_chart, "histogram", "Frequency of occurrences", "Scene count ranges")

        elif chart == "stackedbar":
            ax_chart.bar(labels, values, color='steelblue', label='Scenes')
            ax_chart.bar(labels, [v/2 for v in values], color='orange', label='Half Scenes')
            ax_chart.set_title("Stacked Bar Chart - Scene Counts vs Half")
            ax_chart.legend()
            add_custom_legend(ax_chart, "stackedbar", "Number of scenes", "Weather conditions comparison")

        elif chart == "scatter":
            ax_chart.scatter(labels, values, color='purple', s=100)
            ax_chart.set_title("Scatter Plot - Weather vs Scene Counts")
            ax_chart.set_ylabel("Number of Scenes")
            add_custom_legend(ax_chart, "scatter", None, "Weather conditions vs scene counts")

        elif chart == "density":
            sns.kdeplot(values, fill=True, color='green', ax=ax_chart)
            ax_chart.set_title("Density Plot - Scene Count Distribution")
            ax_chart.set_xlabel("Scene Counts")
            add_custom_legend(ax_chart, "density", None, "Scene count density distribution")
 
        add_stats(axes[1])
 
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        outpath = os.path.join(output_dir, f"{chart}_with_stats.png")
        plt.savefig(outpath, dpi=300, bbox_inches="tight", facecolor='white')
        plt.close()
        print(f"✅ Saved: {outpath}")