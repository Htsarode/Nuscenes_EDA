import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os

def plot_environment_distribution(environment_data, output_dir="figures/exploratory"):
    os.makedirs(output_dir, exist_ok=True)

    # Force only 5 fixed labels
    expected_labels = ["Urban", "Rural", "Desert", "Offroad", "Forest"]
    cleaned = {label: environment_data.get(label, 0) for label in expected_labels}

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
        print(f"{num}. {name}")
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

    stats_text = f"""Environment Statistical Summary

Total Scenes: {total}
Environment Types: {len(labels)}

Most Common: {most_common} ({max_val} scenes)
Least Common: {least_common} ({min_val} scenes)

Mean: {mean_val:.1f} scenes
Std Dev: {std_val:.1f} scenes
Range: {max_val - min_val} scenes

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
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle(f"{chart.capitalize()} Chart with Statistical Summary - Environment",
                     fontsize=16, fontweight="bold", y=0.98)
        ax_chart = axes[0]

        if chart == "bar":
            bars = ax_chart.bar(labels, values, color=colors, edgecolor="black")
            ax_chart.set_title("Bar Chart - Environment Types")
            ax_chart.set_ylabel("Scene Count")
            for bar in bars:
                h = bar.get_height()
                ax_chart.text(bar.get_x()+bar.get_width()/2, h, f"{int(h)}",
                              ha="center", va="bottom", fontweight="bold")

        elif chart == "pie":
            ax_chart.pie(values, labels=labels, autopct="%1.1f%%",
                         startangle=90, colors=colors, explode=[0.05]*len(labels))
            ax_chart.set_title("Pie Chart - Percentage Distribution")

        elif chart == "donut":
            wedges, texts, autotexts = ax_chart.pie(values, labels=labels,
                                                    startangle=90, colors=colors,
                                                    pctdistance=0.85, explode=[0.02]*len(labels),
                                                    autopct=lambda p: f"{int(round(p/100.*total))}")
            centre_circle = plt.Circle((0, 0), 0.70, fc="white")
            ax_chart.add_artist(centre_circle)
            ax_chart.text(0, 0, f"Total\nScenes\n{total}", ha="center", va="center",
                          fontsize=12, fontweight="bold", color="darkgreen")
            ax_chart.set_title("Donut Chart - Scene Counts")

        elif chart == "heatmap":
            sns.heatmap(np.array([values]), annot=True, fmt="d", cmap="Greens",
                        xticklabels=labels, yticklabels=["Scenes"], ax=ax_chart)
            ax_chart.set_title("Heat Map - Environment Intensity")

        elif chart == "radar":
            fig.clf()
            fig, axes = plt.subplots(1, 2, figsize=(14, 6))
            ax_chart = plt.subplot(1, 2, 1, projection="polar")
            num = len(labels)
            angles = [n/float(num) * 2*np.pi for n in range(num)]
            angles += angles[:1]
            radar_values = values + values[:1]
            ax_chart.plot(angles, radar_values, "o-", linewidth=2)
            ax_chart.fill(angles, radar_values, alpha=0.25)
            ax_chart.set_xticks(angles[:-1])
            ax_chart.set_xticklabels(labels)
            ax_chart.set_title("Radar Chart - Environment Distribution")

        elif chart == "histogram":
            ax_chart.hist(values, bins=5, color="skyblue", edgecolor="black")
            ax_chart.set_title("Histogram - Scene Counts")

        elif chart == "stackedbar":
            ax_chart.bar(labels, values, color="seagreen", label="Scenes")
            ax_chart.bar(labels, [v/2 for v in values], color="orange", label="Half Scenes")
            ax_chart.set_title("Stacked Bar Chart")
            ax_chart.legend()

        elif chart == "scatter":
            ax_chart.scatter(labels, values, color="purple", s=100)
            ax_chart.set_title("Scatter Plot - Environments")

        elif chart == "density":
            sns.kdeplot(values, fill=True, color="green", ax=ax_chart)
            ax_chart.set_title("Density Plot - Scene Distribution")

        add_stats(axes[1])

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        outpath = os.path.join(output_dir, f"environment_{chart}_with_stats.png")
        plt.savefig(outpath, dpi=300, bbox_inches="tight", facecolor="white")
        plt.close()
        print(f"✅ Saved: {outpath}")
