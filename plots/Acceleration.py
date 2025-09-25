import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os

def add_custom_legend(ax, chart_type, y_axis_desc, x_axis_desc):
    """Add custom legend with chart descriptions"""
    legend_text = f"📊 {chart_type.title()} Chart\n📈 Y-axis: {y_axis_desc}\n📈 X-axis: {x_axis_desc}"
    
    # Add legend for all chart types
    legend_box = ax.text(0.02, 0.98, legend_text, transform=ax.transAxes, fontsize=9,
                        verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3", 
                        facecolor='lightyellow', alpha=0.8))

def plot_acceleration(data_tuple, output_dir="/home/abhinavk5/Desktop/Nuscenes_EDA/figures/exploratory"):
    """
    Plot acceleration bin and control distribution with user-selected chart type.
    Args:
        data_tuple: tuple of (acceleration_bins, control_data) from load_acceleration_bin_data
        output_dir: directory to save plots
    """
    acceleration_data, control_data = data_tuple
    os.makedirs(output_dir, exist_ok=True)
    
    # Define acceleration labels with ranges
    acc_labels = ["Low Acceleration\n(<1.0 m/s²)", "Medium Acceleration\n(1.0-3.0 m/s²)", "High Acceleration\n(>3.0 m/s²)"]
    acc_keys = ["Low Acceleration", "Medium Acceleration", "High Acceleration"]
    acc_values = [acceleration_data.get(k, 0) for k in acc_keys]
    acc_total = sum(acc_values)
    
    # Define control labels
    control_labels = [
        "No Control\n(Coasting)",
        "Light Brake\n(1-3)",
        "Medium Brake\n(4-7)",
        "Heavy Brake\n(8-10)",
        "Light Throttle\n(1-50)",
        "Medium Throttle\n(51-150)",
        "High Throttle\n(151-200)"
    ]
    control_values = [control_data.get(k, 0) for k in [
        "No Control",
        "Light Brake",
        "Medium Brake",
        "Heavy Brake",
        "Light Throttle",
        "Medium Throttle",
        "High Throttle"
    ]]
    control_total = sum(control_values)
    
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
    
    acc_colors = list(plt.cm.Set2(np.linspace(0, 1, len(acc_labels))))
    control_colors = list(plt.cm.Set3(np.linspace(0, 1, len(control_labels))))
    
    stats_text = f"""Vehicle Motion Analysis Summary\n
Acceleration Distribution:
Total Frames: {acc_total}"""
    
    for label, value in zip(acc_labels, acc_values):
        pct = (value / acc_total * 100) if acc_total > 0 else 0
        stats_text += f"\n• {label}: {value} ({pct:.1f}%)"
        
    stats_text += f"\n\nControl Input Distribution:\nTotal Frames: {control_total}"
    
    for label, value in zip(control_labels, control_values):
        pct = (value / control_total * 100) if control_total > 0 else 0
        stats_text += f"\n• {label}: {value} ({pct:.1f}%)"
    
    def add_stats(ax):
        ax.axis("off")
        ax.text(0.05, 0.95, stats_text, transform=ax.transAxes,
                fontsize=11, verticalalignment="top", fontfamily="monospace",
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgreen", alpha=0.8))
        ax.set_title("Statistical Summary", fontsize=14, fontweight="bold")
    
    print(f"\nSaving plots to: {output_dir}")
    for chart in selected_charts:
        fig, axes = plt.subplots(1, 3, figsize=(24, 7))
        fig.suptitle(f"{chart.capitalize()} Chart - Vehicle Motion Analysis",
                     fontsize=16, fontweight="bold", y=0.98)
        
        # Acceleration chart in first subplot
        ax_acc = axes[0]
        # Control distribution in second subplot
        ax_control = axes[1]
        # Stats in third subplot
        ax_stats = axes[2]
        
        if chart == "bar":
            # Acceleration distribution
            bars_acc = ax_acc.bar(acc_labels, acc_values, color=plt.cm.Set2(np.linspace(0, 1, 3)), 
                                edgecolor="black", linewidth=1.2)
            for bar, val in zip(bars_acc, acc_values):
                ax_acc.text(bar.get_x()+bar.get_width()/2, val, f"{val}",
                          ha="center", va="bottom", fontweight="bold", fontsize=10)
            ax_acc.set_title("Acceleration Distribution", fontsize=14, fontweight="bold")
            ax_acc.set_xlabel("Acceleration Categories", fontsize=12, fontweight="bold")
            ax_acc.set_ylabel("Frame Count", fontsize=12, fontweight="bold")
            ax_acc.grid(True, alpha=0.3)
            ax_acc.tick_params(axis='x', rotation=45)
            
            # Control distribution
            control_colors = plt.cm.Set3(np.linspace(0, 1, 7))
            bars_control = ax_control.bar(control_labels, control_values, color=control_colors,
                                        edgecolor="black", linewidth=1.2)
            for bar, val in zip(bars_control, control_values):
                ax_control.text(bar.get_x()+bar.get_width()/2, val, f"{val}",
                              ha="center", va="bottom", fontweight="bold", fontsize=10)
            ax_control.set_title("Brake & Throttle Distribution", fontsize=14, fontweight="bold")
            ax_control.set_xlabel("Control Input Categories", fontsize=12, fontweight="bold")
            ax_control.set_ylabel("Frame Count", fontsize=12, fontweight="bold")
            ax_control.grid(True, alpha=0.3)
            ax_control.tick_params(axis='x', rotation=45)
            
            # Add stats and save plot
            add_stats(ax_stats)
            plt.tight_layout()
            save_path = os.path.join(output_dir, f"acceleration_{chart}_chart.png")
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            plt.close()
            print(f"Saved plot: {save_path}")
        
        elif chart == "pie":
            # Acceleration distribution pie
            ax_acc.pie(acc_values, labels=acc_labels, autopct="%1.1f%%",
                      startangle=90, colors=acc_colors, explode=[0.05]*len(acc_labels))
            ax_acc.set_title("Acceleration Distribution", fontsize=14, fontweight="bold")
            add_custom_legend(ax_acc, "pie", "Frame count percentage in each acceleration bin", "Acceleration categories")
            
            # Control distribution pie
            ax_control.pie(control_values, labels=control_labels, autopct="%1.1f%%",
                         startangle=90, colors=control_colors, explode=[0.05]*len(control_labels))
            ax_control.set_title("Brake & Throttle Distribution", fontsize=14, fontweight="bold")
            add_custom_legend(ax_control, "pie", "Frame count percentage for each control input", "Control input categories")
            
            # Add stats and save plot
            add_stats(ax_stats)
            plt.tight_layout()
            save_path = os.path.join(output_dir, f"acceleration_{chart}_chart.png")
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            plt.close()
            print(f"Saved plot: {save_path}")
        
        elif chart == "donut":
            # Acceleration distribution donut
            wedges, texts, autotexts = ax_acc.pie(acc_values, labels=acc_labels,
                                                 startangle=90, colors=acc_colors,
                                                 pctdistance=0.85, explode=[0.02]*len(acc_labels),
                                                 autopct=lambda p: f"{int(round(p/100.*acc_total))}")
            centre_circle = plt.Circle((0, 0), 0.70, fc="white")
            ax_acc.add_artist(centre_circle)
            ax_acc.text(0, 0, f"Total\n{acc_total}", ha="center", va="center",
                       fontsize=12, fontweight="bold", color="darkblue")
            ax_acc.set_title("Acceleration Distribution", fontsize=14, fontweight="bold")
            add_custom_legend(ax_acc, "donut", "Frame counts in each acceleration category", "Acceleration categories")
            
            # Control distribution donut
            wedges, texts, autotexts = ax_control.pie(control_values, labels=control_labels,
                                                     startangle=90, colors=control_colors,
                                                     pctdistance=0.85, explode=[0.02]*len(control_labels),
                                                     autopct=lambda p: f"{int(round(p/100.*control_total))}")
            centre_circle = plt.Circle((0, 0), 0.70, fc="white")
            ax_control.add_artist(centre_circle)
            ax_control.text(0, 0, f"Total\n{control_total}", ha="center", va="center",
                          fontsize=12, fontweight="bold", color="darkblue")
            ax_control.set_title("Brake & Throttle Distribution", fontsize=14, fontweight="bold")
            add_custom_legend(ax_control, "donut", "Frame counts for each control input", "Control input categories")
            
            # Add stats and save plot
            add_stats(ax_stats)
            plt.tight_layout()
            save_path = os.path.join(output_dir, f"acceleration_{chart}_chart.png")
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            plt.close()
            print(f"Saved plot: {save_path}")
        
        elif chart == "heatmap":
            # Acceleration heatmap
            sns.heatmap(np.array([acc_values]), annot=True, fmt="d", cmap="YlGnBu",
                       xticklabels=acc_labels, yticklabels=["Frame Count"], ax=ax_acc,
                       cbar_kws={'label': 'Count'})
            ax_acc.set_title("Acceleration Distribution", fontsize=14, fontweight="bold")
            ax_acc.tick_params(axis='x', rotation=45)
            add_custom_legend(ax_acc, "heatmap", "Frame count intensity", "Acceleration categories")
            
            # Control heatmap
            sns.heatmap(np.array([control_values]), annot=True, fmt="d", cmap="YlOrRd",
                       xticklabels=control_labels, yticklabels=["Frame Count"], ax=ax_control,
                       cbar_kws={'label': 'Count'})
            ax_control.set_title("Brake & Throttle Distribution", fontsize=14, fontweight="bold")
            ax_control.tick_params(axis='x', rotation=45)
            add_custom_legend(ax_control, "heatmap", "Frame count intensity", "Control input categories")
            
            # Add stats to the third subplot and adjust layout
            # Add stats and save plot
            add_stats(ax_stats)
            plt.tight_layout()
            
            save_path = os.path.join(output_dir, f"acceleration_{chart}_chart.png")
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            plt.close()
            print(f"Saved plot: {save_path}")
        
        if chart == "radar":
            # Acceleration radar chart
            ax_acc = plt.subplot(131, projection="polar")
            num_acc = len(acc_labels)
            angles_acc = [n/float(num_acc) * 2*np.pi for n in range(num_acc)]
            angles_acc += angles_acc[:1]
            acc_radar_values = acc_values + acc_values[:1]
            ax_acc.plot(angles_acc, acc_radar_values, "o-", linewidth=2, color='darkred', markersize=8)
            ax_acc.fill(angles_acc, acc_radar_values, alpha=0.25, color='salmon')
            ax_acc.set_xticks(angles_acc[:-1])
            ax_acc.set_xticklabels(acc_labels)
            ax_acc.set_title("Radar Chart - Acceleration Distribution", fontsize=14, fontweight="bold")
            add_custom_legend(ax_acc, "radar", "Frame count radial distribution", "Acceleration categories")
            
            # Control radar chart
            ax_control = plt.subplot(132, projection="polar")
            num_control = len(control_labels)
            angles_control = [n/float(num_control) * 2*np.pi for n in range(num_control)]
            angles_control += angles_control[:1]
            control_radar_values = control_values + control_values[:1]
            ax_control.plot(angles_control, control_radar_values, "o-", linewidth=2, color='darkblue', markersize=8)
            ax_control.fill(angles_control, control_radar_values, alpha=0.25, color='lightblue')
            ax_control.set_xticks(angles_control[:-1])
            ax_control.set_xticklabels(control_labels)
            ax_control.set_title("Radar Chart - Control Distribution", fontsize=14, fontweight="bold")
            add_custom_legend(ax_control, "radar", "Frame count radial distribution", "Control categories")
            
            # Add stats and save plot
            add_stats(ax_stats)
            plt.tight_layout()
            save_path = os.path.join(output_dir, f"acceleration_{chart}_chart.png")
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            plt.close()
            print(f"Saved plot: {save_path}")
        
        if chart == "histogram":
            # Acceleration histogram
            ax_acc.hist(acc_values, bins=max(3, len(set(acc_values))), color="skyblue", edgecolor="black", alpha=0.7)
            ax_acc.set_title("Histogram - Acceleration Distribution", fontsize=14, fontweight="bold")
            ax_acc.set_xlabel("Frame Counts", fontsize=12, fontweight="bold")
            ax_acc.set_ylabel("Frequency", fontsize=12, fontweight="bold")
            ax_acc.grid(True, alpha=0.3)
            add_custom_legend(ax_acc, "histogram", "Frequency of frame counts", "Frame count bins")
            
            # Control histogram
            ax_control.hist(control_values, bins=max(3, len(set(control_values))), color="lightgreen", edgecolor="black", alpha=0.7)
            ax_control.set_title("Histogram - Control Distribution", fontsize=14, fontweight="bold")
            ax_control.set_xlabel("Frame Counts", fontsize=12, fontweight="bold")
            ax_control.set_ylabel("Frequency", fontsize=12, fontweight="bold")
            ax_control.grid(True, alpha=0.3)
            add_custom_legend(ax_control, "histogram", "Frequency of frame counts", "Control categories")
            
            # Add stats and save plot
            add_stats(ax_stats)
            plt.tight_layout()
            save_path = os.path.join(output_dir, f"acceleration_{chart}_chart.png")
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            plt.close()
            print(f"Saved plot: {save_path}")
        
        if chart == "stackedbar":
            # Acceleration stacked bar
            ax_acc.bar(acc_labels, acc_values, color="teal", label="Total Count", alpha=0.8)
            ax_acc.bar(acc_labels, [v/2 for v in acc_values], color="orange", label="50% Mark", alpha=0.6)
            ax_acc.set_title("Stacked Bar - Acceleration Distribution", fontsize=14, fontweight="bold")
            ax_acc.set_xlabel("Acceleration Bin", fontsize=12, fontweight="bold")
            ax_acc.set_ylabel("Frame Count", fontsize=12, fontweight="bold")
            ax_acc.legend()
            ax_acc.grid(True, alpha=0.3)
            add_custom_legend(ax_acc, "stackedbar", "Total frames and 50% threshold", "Acceleration categories")
            
            # Control stacked bar
            ax_control.bar(control_labels, control_values, color="darkblue", label="Total Count", alpha=0.8)
            ax_control.bar(control_labels, [v/2 for v in control_values], color="lightblue", label="50% Mark", alpha=0.6)
            ax_control.set_title("Stacked Bar - Control Distribution", fontsize=14, fontweight="bold")
            ax_control.set_xlabel("Control Categories", fontsize=12, fontweight="bold")
            ax_control.set_ylabel("Frame Count", fontsize=12, fontweight="bold")
            ax_control.legend()
            ax_control.grid(True, alpha=0.3)
            ax_control.tick_params(axis='x', rotation=45)
            add_custom_legend(ax_control, "stackedbar", "Total frames and 50% threshold", "Control categories")
            
            # Add stats and save plot
            add_stats(ax_stats)
            plt.tight_layout()
            save_path = os.path.join(output_dir, f"acceleration_{chart}_chart.png")
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            plt.close()
            print(f"Saved plot: {save_path}")
        
        if chart == "scatter":
            # Acceleration scatter
            x_positions_acc = range(len(acc_labels))
            ax_acc.scatter(x_positions_acc, acc_values, color="purple", s=150, alpha=0.7, edgecolors='black')
            ax_acc.set_xticks(x_positions_acc)
            ax_acc.set_xticklabels(acc_labels)
            ax_acc.set_title("Scatter Plot - Acceleration Distribution", fontsize=14, fontweight="bold")
            ax_acc.set_xlabel("Acceleration Bin", fontsize=12, fontweight="bold")
            ax_acc.set_ylabel("Frame Count", fontsize=12, fontweight="bold")
            ax_acc.grid(True, alpha=0.3)
            add_custom_legend(ax_acc, "scatter", "Frame count for each acceleration category", "Acceleration categories")
            
            # Control scatter
            x_positions_control = range(len(control_labels))
            ax_control.scatter(x_positions_control, control_values, color="green", s=150, alpha=0.7, edgecolors='black')
            ax_control.set_xticks(x_positions_control)
            ax_control.set_xticklabels(control_labels)
            ax_control.set_title("Scatter Plot - Control Distribution", fontsize=14, fontweight="bold")
            ax_control.set_xlabel("Control Categories", fontsize=12, fontweight="bold")
            ax_control.set_ylabel("Frame Count", fontsize=12, fontweight="bold")
            ax_control.grid(True, alpha=0.3)
            ax_control.tick_params(axis='x', rotation=45)
            add_custom_legend(ax_control, "scatter", "Frame count for each control category", "Control categories")
            
            # Add stats and save plot
            add_stats(ax_stats)
            plt.tight_layout()
            save_path = os.path.join(output_dir, f"acceleration_{chart}_chart.png")
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            plt.close()
            print(f"Saved plot: {save_path}")
        
        if chart == "density":
            # Acceleration density
            if len(set(acc_values)) > 1:
                sns.kdeplot(acc_values, fill=True, color="green", ax=ax_acc, alpha=0.7)
                ax_acc.set_title("Density Plot - Acceleration Distribution", fontsize=14, fontweight="bold")
                ax_acc.set_xlabel("Frame Count", fontsize=12, fontweight="bold")
                ax_acc.set_ylabel("Density", fontsize=12, fontweight="bold")
                add_custom_legend(ax_acc, "density", "Probability density of frame counts", "Acceleration categories")
            else:
                ax_acc.text(0.5, 0.5, "Insufficient variation\nfor density plot", 
                         ha="center", va="center", transform=ax_acc.transAxes,
                         fontsize=12, bbox=dict(boxstyle="round", facecolor="yellow", alpha=0.5))
                ax_acc.set_title("Density Plot - Acceleration Distribution", fontsize=14, fontweight="bold")
                
            # Control density
            if len(set(control_values)) > 1:
                sns.kdeplot(control_values, fill=True, color="blue", ax=ax_control, alpha=0.7)
                ax_control.set_title("Density Plot - Control Distribution", fontsize=14, fontweight="bold")
                ax_control.set_xlabel("Frame Count", fontsize=12, fontweight="bold")
                ax_control.set_ylabel("Density", fontsize=12, fontweight="bold")
                add_custom_legend(ax_control, "density", "Probability density of frame counts", "Control categories")
            else:
                ax_control.text(0.5, 0.5, "Insufficient variation\nfor density plot", 
                            ha="center", va="center", transform=ax_control.transAxes,
                            fontsize=12, bbox=dict(boxstyle="round", facecolor="yellow", alpha=0.5))
                ax_control.set_title("Density Plot - Control Distribution", fontsize=14, fontweight="bold")
            
            # Add stats and save plot
            add_stats(ax_stats)
            plt.tight_layout()
            save_path = os.path.join(output_dir, f"acceleration_{chart}_chart.png")
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            plt.close()
            print(f"Saved plot: {save_path}")
    
    print(f"\n🎉 All selected charts saved to: {output_dir}")

if __name__ == "__main__":
    print("🚗 Acceleration Analysis Plotting Tool")
    print("=" * 40)
    print("This module is designed to be imported and used with real nuScenes data.")
    print("Use the main.py file or test scripts to run the complete analysis.")