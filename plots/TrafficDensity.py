import matplotlib.pyplot as plt
import numpy as np
 
def plot_traffic_density_across_frames(density_data: dict):
    """
    Plot user-chosen chart of traffic density categories (Low/Medium/High) across frames.
    """
 
    # Ensure all categories exist even if 0
    categories = ["Low", "Medium", "High"]
    counts = [density_data.get("Low", 0),
              density_data.get("Medium", 0),
              density_data.get("High", 0)]
 
    print("\nChoose a chart type to visualize Traffic Density across Frames:")
    print("1. Bar Chart")
    print("2. Pie Chart")
    print("3. Donut Chart")
    print("4. Heat Map")
    print("5. Radar Chart")
    print("6. Histogram")
    print("7. Stacked Bar Chart")
    print("8. Scatter Plot")
    print("9. Density Plot")
 
    choice = input("Enter number: ").strip()
 
    plt.figure(figsize=(8, 6))
    plt.title("Traffic Density across Frames")
    plt.xlabel("Traffic Density")
    plt.ylabel("Frame Count")
 
    def add_custom_legend(ax, chart_type, y_axis_desc, x_axis_desc):
        """Add custom legend with chart information and axis descriptions"""
        legend_text = f"📊 {chart_type.title()} Chart\n📈 Y-axis: {y_axis_desc}\n📈 X-axis: {x_axis_desc}"
        if chart_type.lower() in ['bar', 'histogram', 'stackedbar']:
            ax.text(0.02, 0.98, legend_text, transform=ax.transAxes, fontsize=9,
                    verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3", 
                    facecolor='lightyellow', alpha=0.8))

    if choice == "1":  # Bar Chart
        plt.bar(categories, counts, color=['green','orange','red'])
        add_custom_legend(plt.gca(), "bar", "Number of frames per traffic density", "Traffic density categories")

    elif choice == "2":  # Pie Chart
        plt.pie(counts, labels=categories, autopct="%1.1f%%", startangle=90)
        add_custom_legend(plt.gca(), "pie", "Number of frames per traffic density", "Traffic density categories")

    elif choice == "3":  # Donut Chart
        wedges, texts, autotexts = plt.pie(counts, labels=categories, autopct="%1.1f%%", startangle=90)
        centre_circle = plt.Circle((0, 0), 0.70, fc='white')
        fig = plt.gcf()
        fig.gca().add_artist(centre_circle)
        from matplotlib.lines import Line2D
        legend_text = "Distribution of traffic density across frames."
        plt.legend(handles=[Line2D([0], [0], color='none', label=legend_text)],
                   loc='upper center', bbox_to_anchor=(0.5, -0.15), fontsize=10, frameon=True)
        add_custom_legend(plt.gca(), "donut", "Number of frames per traffic density", "Traffic density categories")

    elif choice == "4":  # Heat Map (simple 1D heat representation)
        plt.imshow([counts], cmap="Reds", aspect="auto")
        plt.xticks(range(len(categories)), categories)
        plt.yticks([])
        add_custom_legend(plt.gca(), "heatmap", "Number of frames per traffic density", "Traffic density categories")

    elif choice == "5":  # Radar Chart
        labels = np.array(categories)
        stats = np.array(counts)
        angles = np.linspace(0, 2*np.pi, len(labels), endpoint=False)
        stats = np.concatenate((stats, [stats[0]]))
        angles = np.concatenate((angles, [angles[0]]))
        ax = plt.subplot(111, polar=True)
        ax.plot(angles, stats, 'o-', linewidth=2)
        ax.fill(angles, stats, alpha=0.25)
        ax.set_thetagrids(angles[:-1] * 180/np.pi, labels)
        add_custom_legend(ax, "radar", "Number of frames per traffic density", "Traffic density categories")

    elif choice == "6":  # Histogram
        plt.hist(counts, bins=[0,10,30,50], color='skyblue', rwidth=0.8)
        add_custom_legend(plt.gca(), "histogram", "Number of frames per traffic density", "Traffic density categories")

    elif choice == "7":  # Stacked Bar Chart (single stacked bar)
        plt.bar(["Density"], [counts[0]], label="Low", color="green")
        plt.bar(["Density"], [counts[1]], bottom=[counts[0]], label="Medium", color="orange")
        plt.bar(["Density"], [counts[2]], bottom=[counts[0]+counts[1]], label="High", color="red")
        plt.legend()
        add_custom_legend(plt.gca(), "stackedbar", "Number of frames per traffic density", "Traffic density categories")

    elif choice == "8":  # Scatter Plot
        plt.scatter(categories, counts, s=100, color='purple')
        add_custom_legend(plt.gca(), "scatter", "Number of frames per traffic density", "Traffic density categories")

    elif choice == "9":  # Density Plot (Kernel-like, simulated)
        x = np.repeat(categories, counts)
        if len(x) > 0:
            positions = [categories.index(val) for val in x]
            plt.hist(positions, bins=np.arange(-0.5,3.5,1), density=True, alpha=0.5)
            plt.xticks(range(len(categories)), categories)
        else:
            plt.bar(categories, [0,0,0])
        add_custom_legend(plt.gca(), "density", "Number of frames per traffic density", "Traffic density categories")

    else:
        print("⚠️ Invalid choice — showing default Bar Chart.")
        plt.bar(categories, counts, color=['green','orange','red'])
        add_custom_legend(plt.gca(), "bar", "Number of frames per traffic density", "Traffic density categories")
 
    plt.tight_layout()
    plt.show()