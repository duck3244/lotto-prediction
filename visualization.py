"""
Data Visualization Module (RTX 4060 Optimized)
- Lottery number frequency visualization
- Recent draws heatmap
- Odd/Even distribution visualization
- Prediction result visualization
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

# Set graph style
plt.style.use('ggplot')
mpl.rcParams['font.family'] = 'DejaVu Sans'  # Default English font

def visualize_data(df, frequencies, output_dir="visualization"):
    """
    Generate lottery data visualizations

    Args:
        df: Lottery data DataFrame
        frequencies: Number frequency dictionary
        output_dir: Visualization file save path

    Returns:
        List of created visualization file paths
    """
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)

    print("\nGenerating data visualizations...")

    # List of saved file paths
    created_files = []

    # 1. Number frequency chart
    freq_path = create_frequency_chart(frequencies, output_path)
    created_files.append(freq_path)

    # 2. Recent 50 draws heatmap
    heatmap_path = create_recent_heatmap(df, output_path)
    created_files.append(heatmap_path)

    # 3. Odd/Even ratio pie chart
    pie_path = create_odd_even_chart(df, output_path)
    created_files.append(pie_path)

    # 4. Number range distribution chart
    range_path = create_range_distribution_chart(df, output_path)
    created_files.append(range_path)

    print("Visualization images have been created:")
    for file_path in created_files:
        print(f"- {file_path}")

    return created_files

def create_frequency_chart(frequencies, output_dir):
    """Create number frequency chart"""
    file_path = output_dir / 'number_frequency.png'

    plt.figure(figsize=(15, 8))
    nums = list(range(1, 46))
    freqs = [frequencies.get(num, 0) for num in nums]

    # Bar chart
    bars = plt.bar(nums, freqs, color='skyblue', alpha=0.7)

    # Highlight top 5 numbers
    top_5 = sorted(frequencies.items(), key=lambda x: x[1], reverse=True)[:5]
    top_5_nums = [num for num, _ in top_5]

    for i, num in enumerate(nums):
        if num in top_5_nums:
            bars[i].set_color('crimson')
            bars[i].set_alpha(1.0)

    plt.title('Lottery Number Frequency', fontsize=18)
    plt.xlabel('Lottery Number', fontsize=14)
    plt.ylabel('Frequency', fontsize=14)
    plt.xticks(nums)
    plt.grid(True, alpha=0.3)

    # Add average line
    avg_freq = sum(freqs) / len(freqs)
    plt.axhline(y=avg_freq, color='green', linestyle='--', alpha=0.7,
                label=f'Average Frequency: {avg_freq:.1f}')

    # Add legend
    plt.legend(fontsize=12)

    # Add labels to top 5 numbers
    for num, freq in top_5:
        plt.text(num, freq + 2, f"{freq}", ha='center', va='bottom',
                 fontsize=11, fontweight='bold', color='red')

    plt.tight_layout()
    plt.savefig(file_path, dpi=150)
    plt.close()

    return file_path

def create_recent_heatmap(df, output_dir, recent_count=50):
    """Create recent draws heatmap"""
    file_path = output_dir / 'recent_numbers_heatmap.png'

    plt.figure(figsize=(14, 10))

    # Extract recent draws data
    recent_count = min(recent_count, len(df))
    recent_df = df.head(recent_count).copy()

    # Prepare heatmap data
    heatmap_data = np.zeros((recent_count, 45))

    for i, (_, row) in enumerate(recent_df.iterrows()):
        numbers = [row[f'번호{j}'] for j in range(1, 7)]
        for num in numbers:
            heatmap_data[i, num-1] = 1

    # Color map options: 'hot', 'inferno', 'plasma', 'viridis'
    plt.imshow(heatmap_data, cmap='hot', aspect='auto')
    plt.colorbar(label='Drawn')
    plt.title(f'Recent {recent_count} Draws Number Distribution', fontsize=18)
    plt.xlabel('Lottery Number', fontsize=14)
    plt.ylabel('Draw (Most Recent First)', fontsize=14)

    # Set x-axis labels
    plt.xticks(np.arange(0, 45, 5), np.arange(1, 46, 5))

    # Set y-axis labels (draw numbers)
    y_ticks = np.arange(0, recent_count, 5)
    plt.yticks(y_ticks, recent_df['회차'].iloc[y_ticks])

    plt.tight_layout()
    plt.savefig(file_path, dpi=150)
    plt.close()

    return file_path

def create_odd_even_chart(df, output_dir):
    """Create odd/even distribution pie chart"""
    file_path = output_dir / 'odd_even_distribution.png'

    plt.figure(figsize=(10, 8))

    odd_even_counts = {'Odd Dominant': 0, 'Even Dominant': 0, 'Balanced': 0}

    for _, row in df.iterrows():
        numbers = [row[f'번호{j}'] for j in range(1, 7)]
        odd_count = sum(1 for num in numbers if num % 2 == 1)
        even_count = 6 - odd_count

        if odd_count > even_count:
            odd_even_counts['Odd Dominant'] += 1
        elif even_count > odd_count:
            odd_even_counts['Even Dominant'] += 1
        else:
            odd_even_counts['Balanced'] += 1

    labels = odd_even_counts.keys()
    sizes = odd_even_counts.values()
    colors = ['#ff9999', '#66b3ff', '#99ff99']
    explode = (0.1, 0.1, 0.1)  # Separate slices

    plt.pie(sizes, explode=explode, labels=labels, colors=colors,
            autopct='%1.1f%%', shadow=True, startangle=90,
            textprops={'fontsize': 14})
    plt.axis('equal')
    plt.title('Odd/Even Distribution in Winning Numbers', fontsize=18)

    # Add total value
    plt.figtext(0.5, 0.01, f"Analysis of {len(df)} total draws",
                ha="center", fontsize=12, bbox={"facecolor":"white", "alpha":0.5, "pad":5})

    plt.tight_layout()
    plt.savefig(file_path, dpi=150)
    plt.close()

    return file_path

def create_range_distribution_chart(df, output_dir):
    """Create number range distribution chart"""
    file_path = output_dir / 'range_distribution.png'

    plt.figure(figsize=(12, 8))

    # Define ranges
    ranges = [(1, 10), (11, 20), (21, 30), (31, 40), (41, 45)]
    range_labels = ['1-10', '11-20', '21-30', '31-40', '41-45']

    # Calculate frequency for each range
    range_counts = [0, 0, 0, 0, 0]

    for _, row in df.iterrows():
        numbers = [row[f'번호{j}'] for j in range(1, 7)]
        for num in numbers:
            for i, (start, end) in enumerate(ranges):
                if start <= num <= end:
                    range_counts[i] += 1
                    break

    # Calculate total number count (draws * 6)
    total_numbers = len(df) * 6

    # Calculate expected ratios (proportional to number count)
    expected_ratios = []
    for i, (start, end) in enumerate(ranges):
        range_size = end - start + 1
        expected_ratio = (range_size / 45) * total_numbers
        expected_ratios.append(expected_ratio)

    # Create chart
    x = range(len(range_labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 8))
    actual = ax.bar(np.array(x) - width/2, range_counts, width, label='Actual', color='royalblue')
    expected = ax.bar(np.array(x) + width/2, expected_ratios, width, label='Expected', color='lightcoral')

    # Customize chart
    ax.set_title('Number Range Distribution', fontsize=18)
    ax.set_xlabel('Number Range', fontsize=14)
    ax.set_ylabel('Frequency', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(range_labels)
    ax.legend(fontsize=12)

    # Add value labels
    def add_labels(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.0f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=11)

    add_labels(actual)
    add_labels(expected)

    plt.tight_layout()
    plt.savefig(file_path, dpi=150)
    plt.close()

    return file_path

def visualize_prediction_comparison(lstm_prediction, ensemble_prediction, output_dir):
    """Compare LSTM and ensemble prediction results visualization"""
    file_path = output_dir / 'prediction_comparison.png'

    plt.figure(figsize=(10, 6))

    # All number range
    all_numbers = list(range(1, 46))

    # Number status (0: not selected, 1: LSTM only, 2: Ensemble only, 3: Both)
    number_status = np.zeros(45)

    for num in lstm_prediction:
        number_status[num-1] += 1

    for num in ensemble_prediction:
        number_status[num-1] += 2

    # Color mapping
    colors = ['whitesmoke', 'lightblue', 'lightgreen', 'gold']
    labels = ['Not Selected', 'LSTM Prediction', 'Ensemble Prediction', 'Both Models']

    # Create patches for legend
    patches = [plt.Rectangle((0, 0), 1, 1, fc=colors[i]) for i in range(4)]

    # Assign colors by number status
    bar_colors = [colors[int(status)] for status in number_status]

    # Create bar chart
    plt.bar(all_numbers, np.ones(45), color=bar_colors)

    # Add number labels
    for i, num in enumerate(all_numbers):
        status = int(number_status[i])
        if status > 0:  # Only show selected numbers
            plt.text(num, 0.5, str(num), ha='center', va='center',
                     fontweight='bold', fontsize=12, color='black')

    # Chart settings
    plt.title('LSTM vs Ensemble Prediction Comparison', fontsize=18)
    plt.xlabel('Lottery Number', fontsize=14)
    plt.xlim(0.5, 45.5)
    plt.ylim(0, 1)
    plt.yticks([])  # Remove y-axis ticks
    plt.xticks([5, 10, 15, 20, 25, 30, 35, 40, 45])  # Simplify x-axis ticks
    plt.grid(axis='x', linestyle='--', alpha=0.3)

    # Add legend
    plt.legend(patches, labels, loc='upper center', bbox_to_anchor=(0.5, -0.05),
              ncol=4, fontsize=12)

    plt.tight_layout()
    plt.savefig(file_path, dpi=150)
    plt.close()

    return file_path