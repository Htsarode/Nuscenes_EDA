import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os
from math import pi

def plot_time_of_day_distribution(time_data, output_dir="figures/exploratory"):
    os.makedirs(output_dir, exist_ok=True)

    labels = ["Morning", "Noon", "Evening", "Night"]
    values = [time_data.get(t, 0) for t in labels]
    total_scenes = sum(values)

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

    # Colors
    colors = list(plt.cm.Set2(np.linspace(0, 1, len(labels))))

    # --- Statistical Summary Text ---
    mean_scenes = np.mean(values)
    std_scenes = np.std(values)
    max_scenes = max(values)
    min_scenes = min(values)
    most_common = labels[values.index(max_scenes)]
    least_common = labels[values.index(min_scenes)]

    stats_text = f"""Statistical Summary

Total Scenes: {total_scenes}
Time Periods: {len(labels)}

Most Common: {most_common} ({max_scenes} scenes)
Least Common: {least_common} ({min_scenes} scenes)

Mean: {mean_scenes:.1f} scenes
Std Dev: {std_scenes:.1f} scenes
Range: {max_scenes - min_scenes} scenes

Time of Day Distribution:"""
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
        fig.suptitle(f"{chart.capitalize()} Chart with Statistical Summary - Time of Day",
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

        elif chart == "pie":
            wedges, texts, autotexts = ax_chart.pie(values, labels=labels,
                                                    autopct="%1.1f%%", startangle=90,
                                                    colors=colors, explode=[0.05]*len(labels))
            ax_chart.set_title("Pie Chart - Percentage Distribution")
            for autotext in autotexts:
                autotext.set_color('white')
                autotext.set_fontweight('bold')

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

        elif chart == "heatmap":
            data = np.array(values).reshape(1, -1)
            sns.heatmap(data, annot=True, fmt="d", cmap="YlOrRd",
                        xticklabels=labels, yticklabels=["Scene Count"],
                        ax=ax_chart, cbar_kws={'label': 'Number of Scenes'})
            ax_chart.set_title("Heat Map - Time of Day")
            ax_chart.tick_params(axis='x', rotation=45)

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
            ax_chart.set_title("Radar Chart - Time of Day Distribution")

        elif chart == "histogram":
            ax_chart.hist(values, bins=5, color='skyblue', edgecolor='black')
            ax_chart.set_title("Histogram - Frequency of Scene Counts")
            ax_chart.set_xlabel("Scene Counts")
            ax_chart.set_ylabel("Frequency")

        elif chart == "stackedbar":
            ax_chart.bar(labels, values, color='steelblue', label='Scenes')
            ax_chart.bar(labels, [v/2 for v in values], color='orange', label='Half Scenes')
            ax_chart.set_title("Stacked Bar Chart - Scenes vs Half")
            ax_chart.legend()

        elif chart == "scatter":
            ax_chart.scatter(labels, values, color='purple', s=100)
            ax_chart.set_title("Scatter Plot - Time of Day vs Scene Counts")
            ax_chart.set_ylabel("Number of Scenes")

        elif chart == "density":
            sns.kdeplot(values, fill=True, color='green', ax=ax_chart)
            ax_chart.set_title("Density Plot - Scene Count Distribution")
            ax_chart.set_xlabel("Scene Counts")

        add_stats(axes[1])

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        outpath = os.path.join(output_dir, f"timeofday_{chart}_with_stats.png")
        plt.savefig(outpath, dpi=300, bbox_inches="tight", facecolor='white')
        plt.close()
        print(f"✅ Saved: {outpath}")


if __name__ == "__main__":
    # ----- USER INPUT -----
    print("Enter number of scenes for each time of day:")
    morning = int(input("🌅 Morning: "))
    noon = int(input("☀️ Noon: "))
    evening = int(input("🌆 Evening: "))
    night = int(input("🌙 Night: "))

    time_data = {
        "Morning": morning,
        "Noon": noon,
        "Evening": evening,
        "Night": night
    }

    plot_time_of_day_distribution(time_data)
