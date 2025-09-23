import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os

def add_custom_legend(ax, chart_type, y_axis_desc, x_axis_desc):
    """Add custom legend with chart information and axis descriptions"""
    legend_text = f"📊 {chart_type.title()} Chart\n📈 Y-axis: {y_axis_desc}\n📈 X-axis: {x_axis_desc}"
    
    # Add legend for all chart types
    legend_box = ax.text(0.02, 0.98, legend_text, transform=ax.transAxes, fontsize=9,
                        verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3", 
                        facecolor='lightyellow', alpha=0.8))

def plot_acceleration(acceleration_data, output_dir="figures/exploratory"):
    """
    Plot acceleration bin (Low, Medium, High) frame counts with user-selected chart type.
    Args:
        acceleration_data: dict with keys 'Low Acceleration', 'Medium Acceleration', 'High Acceleration'
        output_dir: directory to save plots
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Define all expected labels (will show all on x-axis even if count is 0)
    labels = ["Low Acceleration", "Medium Acceleration", "High Acceleration"]
    values = [acceleration_data.get(l, 0) for l in labels]
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
    
    colors = list(plt.cm.Set2(np.linspace(0, 1, len(labels))))
    stats_text = f"""Acceleration Bin Frame Count Summary\n\nTotal Frames: {total}\n"""
    
    for label, value in zip(labels, values):
        pct = (value / total * 100) if total > 0 else 0
        stats_text += f"\n• {label}: {value} ({pct:.1f}%)"
    
    def add_stats(ax):
        ax.axis("off")
        ax.text(0.05, 0.95, stats_text, transform=ax.transAxes,
                fontsize=11, verticalalignment="top", fontfamily="monospace",
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgreen", alpha=0.8))
        ax.set_title("Statistical Summary", fontsize=14, fontweight="bold")
    
    for chart in selected_charts:
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))
        fig.suptitle(f"{chart.capitalize()} Chart with Statistical Summary - Acceleration Distribution",
                     fontsize=16, fontweight="bold", y=0.98)
        ax_chart = axes[0]
        
        if chart == "bar":
            bars = ax_chart.bar(labels, values, color=colors, edgecolor="black", linewidth=1.2)
            for bar, val in zip(bars, values):
                ax_chart.text(bar.get_x()+bar.get_width()/2, val, f"{val}",
                              ha="center", va="bottom", fontweight="bold", fontsize=10)
            ax_chart.set_title("Bar Chart - Acceleration Bin Frame Counts", fontsize=14, fontweight="bold")
            ax_chart.set_xlabel("Acceleration Bin", fontsize=12, fontweight="bold")
            ax_chart.set_ylabel("Frame Count", fontsize=12, fontweight="bold")
            ax_chart.grid(True, alpha=0.3)
            add_custom_legend(ax_chart, "bar", "Number of frames in each acceleration bin", "Acceleration categories")
        
        elif chart == "pie":
            ax_chart.pie(values, labels=labels, autopct="%1.1f%%",
                         startangle=90, colors=colors, explode=[0.05]*len(labels))
            ax_chart.set_title("Pie Chart - Acceleration Bin Distribution", fontsize=14, fontweight="bold")
            add_custom_legend(ax_chart, "pie", "Frame count percentage in each acceleration bin", "Acceleration categories (Low/Medium/High)")
        
        elif chart == "donut":
            wedges, texts, autotexts = ax_chart.pie(values, labels=labels,
                                                    startangle=90, colors=colors,
                                                    pctdistance=0.85, explode=[0.02]*len(labels),
                                                    autopct=lambda p: f"{int(round(p/100.*total))}")
            centre_circle = plt.Circle((0, 0), 0.70, fc="white")
            ax_chart.add_artist(centre_circle)
            ax_chart.text(0, 0, f"Total\n{total}", ha="center", va="center",
                          fontsize=12, fontweight="bold", color="darkblue")
            ax_chart.set_title("Donut Chart - Acceleration Bin Counts", fontsize=14, fontweight="bold")
            add_custom_legend(ax_chart, "donut", "Frame counts in each acceleration category", "Acceleration categories (Low/Medium/High)")
        
        elif chart == "heatmap":
            sns.heatmap(np.array([values]), annot=True, fmt="d", cmap="YlGnBu",
                        xticklabels=labels, yticklabels=["Frame Count"], ax=ax_chart,
                        cbar_kws={'label': 'Count'})
            ax_chart.set_title("Heat Map - Acceleration Bin Intensity", fontsize=14, fontweight="bold")
            add_custom_legend(ax_chart, "heatmap", "Frame count intensity scale", "Acceleration categories (Low/Medium/High)")
        
        elif chart == "radar":
            fig.clf()
            fig, axes = plt.subplots(1, 2, figsize=(16, 7))
            ax_chart = plt.subplot(1, 2, 1, projection="polar")
            num = len(labels)
            angles = [n/float(num) * 2*np.pi for n in range(num)]
            angles += angles[:1]
            radar_values = values + values[:1]
            ax_chart.plot(angles, radar_values, "o-", linewidth=2, color='darkred', markersize=8)
            ax_chart.fill(angles, radar_values, alpha=0.25, color='salmon')
            ax_chart.set_xticks(angles[:-1])
            ax_chart.set_xticklabels(labels)
            ax_chart.set_title("Radar Chart - Acceleration Distribution", fontsize=14, fontweight="bold")
            add_custom_legend(ax_chart, "radar", "Frame count radial distribution", "Acceleration categories (Low/Medium/High)")
        
        elif chart == "histogram":
            ax_chart.hist(values, bins=max(3, len(set(values))), color="skyblue", edgecolor="black", alpha=0.7)
            ax_chart.set_title("Histogram - Frame Counts per Acceleration Bin", fontsize=14, fontweight="bold")
            ax_chart.set_xlabel("Frame Counts", fontsize=12, fontweight="bold")
            ax_chart.set_ylabel("Frequency", fontsize=12, fontweight="bold")
            ax_chart.grid(True, alpha=0.3)
            add_custom_legend(ax_chart, "histogram", "Frequency of frame counts", "Frame count bins")
        
        elif chart == "stackedbar":
            ax_chart.bar(labels, values, color="teal", label="Total Count", alpha=0.8)
            ax_chart.bar(labels, [v/2 for v in values], color="orange", label="50% Mark", alpha=0.6)
            ax_chart.set_title("Stacked Bar Chart - Acceleration Bins", fontsize=14, fontweight="bold")
            ax_chart.set_xlabel("Acceleration Bin", fontsize=12, fontweight="bold")
            ax_chart.set_ylabel("Frame Count", fontsize=12, fontweight="bold")
            ax_chart.legend()
            ax_chart.grid(True, alpha=0.3)
            add_custom_legend(ax_chart, "stackedbar", "Total frames and 50% threshold", "Acceleration categories (Low/Medium/High)")
        
        elif chart == "scatter":
            x_positions = range(len(labels))
            ax_chart.scatter(x_positions, values, color="purple", s=150, alpha=0.7, edgecolors='black')
            ax_chart.set_xticks(x_positions)
            ax_chart.set_xticklabels(labels)
            ax_chart.set_title("Scatter Plot - Acceleration Bins", fontsize=14, fontweight="bold")
            ax_chart.set_xlabel("Acceleration Bin", fontsize=12, fontweight="bold")
            ax_chart.set_ylabel("Frame Count", fontsize=12, fontweight="bold")
            ax_chart.grid(True, alpha=0.3)
            add_custom_legend(ax_chart, "scatter", "Frame count for each acceleration category", "Acceleration categories (Low/Medium/High)")
        
        elif chart == "density":
            if len(set(values)) > 1:
                sns.kdeplot(values, fill=True, color="green", ax=ax_chart, alpha=0.7)
                ax_chart.set_title("Density Plot - Frame Count Distribution", fontsize=14, fontweight="bold")
                ax_chart.set_xlabel("Frame Count", fontsize=12, fontweight="bold")
                ax_chart.set_ylabel("Density", fontsize=12, fontweight="bold")
                add_custom_legend(ax_chart, "density", "Probability density of frame counts", "Frame count values")
            else:
                ax_chart.text(0.5, 0.5, "Insufficient variation\nfor density plot", 
                             ha="center", va="center", transform=ax_chart.transAxes,
                             fontsize=12, bbox=dict(boxstyle="round", facecolor="yellow", alpha=0.5))
                ax_chart.set_title("Density Plot - Frame Count Distribution", fontsize=14, fontweight="bold")
        
        add_stats(axes[1])
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        outpath = os.path.join(output_dir, f"acceleration_bin_{chart}_with_stats.png")
        plt.savefig(outpath, dpi=300, bbox_inches="tight", facecolor="white")
        plt.close()
        print(f"✅ Saved: {outpath}")
    
    print(f"\n🎉 All selected charts saved to: {output_dir}")

if __name__ == "__main__":
    print("🚗 Acceleration Analysis Plotting Tool")
    print("=" * 40)
    print("This module is designed to be imported and used with real nuScenes data.")
    print("Use the main.py file or test scripts to run the complete analysis.")