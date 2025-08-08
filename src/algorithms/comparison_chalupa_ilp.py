"""
WP1c: Compare Chalupa heuristic to ILP solutions
Analyzes how often the heuristic produces exact solutions and performance metrics
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import re
import json
from datetime import datetime


class WP1Analyzer:
    def __init__(self, results_dir="results"):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)

    def parse_result_file(self, filepath):
        """Parse a result file and extract metrics"""
        results = []
        with open(filepath, 'r') as f:
            content = f.read()

        # Extract individual test results
        test_blocks = content.split('-' * 30)

        for block in test_blocks:
            if 'File:' in block:
                result = {}

                # Extract metrics using regex
                patterns = {
                    'file': r'File:\s+(.+)',
                    'predicted': r'Predicted:\s+(\d+)',
                    'actual': r'Actual:\s+(\d+)',
                    'deviation': r'Deviation:\s+(-?\d+)',
                    'correct': r'Correct:\s+(True|False)',
                    'time_taken': r'Time taken:\s+([\d.]+)'
                }

                for key, pattern in patterns.items():
                    match = re.search(pattern, block)
                    if match:
                        if key in ['predicted', 'actual', 'deviation']:
                            result[key] = int(match.group(1))
                        elif key == 'correct':
                            result[key] = match.group(1) == 'True'
                        elif key == 'time_taken':
                            result[key] = float(match.group(1))
                        else:
                            result[key] = match.group(1)

                if result:
                    results.append(result)

        return results

    def load_all_results(self):
        """Load results for all algorithms"""
        data = {
            'chalupa': [],
            'ilp': [],
            'reduced_ilp': []
        }

        for result_file in self.results_dir.glob("*.txt"):
            filename = result_file.name.lower()

            if 'chalupa' in filename:
                data['chalupa'].extend(self.parse_result_file(result_file))
            elif 'reduced_ilp' in filename:
                data['reduced_ilp'].extend(self.parse_result_file(result_file))
            elif 'ilp' in filename and 'reduced' not in filename:
                data['ilp'].extend(self.parse_result_file(result_file))

        return data

    def analyze_accuracy(self, data):
        """Analyze accuracy of different methods"""
        stats = {}

        for method, results in data.items():
            if not results:
                continue

            df = pd.DataFrame(results)

            stats[method] = {
                'total_tests': len(df),
                'correct': df['correct'].sum() if 'correct' in df else 0,
                'accuracy': df['correct'].mean() * 100 if 'correct' in df else 0,
                'avg_deviation': df['deviation'].mean() if 'deviation' in df else 0,
                'max_deviation': df['deviation'].max() if 'deviation' in df else 0,
                'avg_time': df['time_taken'].mean() if 'time_taken' in df else 0,
                'max_time': df['time_taken'].max() if 'time_taken' in df else 0
            }

        return stats

    def analyze_by_instance_size(self, data):
        """Analyze performance by problem size"""
        # Extract instance characteristics from filenames
        size_analysis = {}

        for method, results in data.items():
            if not results:
                continue

            df = pd.DataFrame(results)

            # Parse instance characteristics from filename
            for idx, row in df.iterrows():
                filename = row.get('file', '')

                # Extract parameters from filename patterns
                # e.g., "uniform_n5_s10_r30.txt" or "skewed_cliques3_min5_max20_perturbation30.txt" and so on
                params = self.extract_params(filename)

                for key, value in params.items():
                    df.loc[idx, key] = value

            size_analysis[method] = df

        return size_analysis

    def extract_params(self, filename):
        """Extract parameters from filename"""
        params = {}

        # Extract numbers after specific patterns
        patterns = {
            'num_cliques': r'n(\d+)',
            'clique_size': r's(\d+)',
            'perturbation': r'r(\d+)|perturbation(\d+)',
            'min_size': r'min(\d+)',
            'max_size': r'max(\d+)'
        }

        for key, pattern in patterns.items():
            match = re.search(pattern, filename)
            if match:
                # Take the first non-None group
                value = next((g for g in match.groups() if g is not None), None)
                if value:
                    params[key] = int(value)

        return params

    def create_comparison_plots(self, data, stats):
        """Create visualization plots for WP1c"""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('WP1c: Chalupa Heuristic vs ILP Comparison', fontsize=16)

        # Plot 1: Accuracy comparison
        ax = axes[0, 0]
        methods = list(stats.keys())
        accuracies = [stats[m]['accuracy'] for m in methods]
        ax.bar(methods, accuracies)
        ax.set_ylabel('Accuracy (%)')
        ax.set_title('Solution Accuracy')
        ax.set_ylim(0, 105)
        for i, v in enumerate(accuracies):
            ax.text(i, v + 1, f'{v:.1f}%', ha='center')

        # Plot 2: Average time comparison
        ax = axes[0, 1]
        avg_times = [stats[m]['avg_time'] for m in methods]
        ax.bar(methods, avg_times)
        ax.set_ylabel('Average Time (seconds)')
        ax.set_title('Average Computation Time')
        ax.set_yscale('log')

        # Plot 3: Average deviation
        ax = axes[0, 2]
        avg_devs = [abs(stats[m]['avg_deviation']) for m in methods]
        ax.bar(methods, avg_devs)
        ax.set_ylabel('Average |Deviation|')
        ax.set_title('Average Absolute Deviation from Ground Truth')

        # Plot 4: Time vs Problem Size (if data available)
        ax = axes[1, 0]
        for method, results in data.items():
            if not results:
                continue
            df = pd.DataFrame(results)
            if 'actual' in df.columns and 'time_taken' in df.columns:
                ax.scatter(df['actual'], df['time_taken'], label=method, alpha=0.6)
        ax.set_xlabel('Chromatic Number (Problem Size)')
        ax.set_ylabel('Time (seconds)')
        ax.set_title('Computation Time vs Problem Size')
        ax.set_yscale('log')
        ax.legend()

        # Plot 5: Deviation distribution
        ax = axes[1, 1]
        for method, results in data.items():
            if not results:
                continue
            df = pd.DataFrame(results)
            if 'deviation' in df.columns:
                ax.hist(df['deviation'], alpha=0.5, label=method, bins=20)
        ax.set_xlabel('Deviation from Ground Truth')
        ax.set_ylabel('Frequency')
        ax.set_title('Distribution of Deviations')
        ax.legend()

        # Plot 6: Success rate by perturbation level
        ax = axes[1, 2]
        size_data = self.analyze_by_instance_size(data)
        for method, df in size_data.items():
            if 'perturbation' in df.columns and 'correct' in df.columns:
                grouped = df.groupby('perturbation')['correct'].mean() * 100
                ax.plot(grouped.index, grouped.values, marker='o', label=method)
        ax.set_xlabel('Perturbation Level')
        ax.set_ylabel('Success Rate (%)')
        ax.set_title('Success Rate vs Perturbation Strength')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        # Save plot
        plot_path = Path('figures') / 'wp1c_comparison.png'
        plot_path.parent.mkdir(exist_ok=True)
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"Saved comparison plot to {plot_path}")

        return fig

    def generate_report(self, stats, output_file='wp1c_analysis_report.md'):
        """Generate a markdown report for WP1c"""
        report = []
        report.append("# WP1c Analysis Report: Chalupa vs ILP Comparison\n")
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

        report.append("## Summary Statistics\n")

        # Create comparison table
        report.append("| Method | Tests | Correct | Accuracy | Avg Dev | Max Dev | Avg Time | Max Time |")
        report.append("|--------|-------|---------|----------|---------|---------|----------|----------|")

        for method, stat in stats.items():
            report.append(
                f"| {method} | {stat['total_tests']} | {stat['correct']} | "
                f"{stat['accuracy']:.1f}% | {stat['avg_deviation']:.2f} | "
                f"{stat['max_deviation']} | {stat['avg_time']:.3f}s | {stat['max_time']:.3f}s |"
            )

        report.append("\n## Key Findings\n")

        # Analyze Chalupa performance
        if 'chalupa' in stats and 'ilp' in stats:
            chalupa_acc = stats['chalupa']['accuracy']
            ilp_acc = stats['ilp']['accuracy']

            report.append(f"### Chalupa Heuristic Performance\n")
            report.append(f"- Achieves {chalupa_acc:.1f}% accuracy compared to optimal solutions")
            report.append(f"- Average deviation: {stats['chalupa']['avg_deviation']:.2f} cliques")
            report.append(
                f"- Speed advantage: {stats['ilp']['avg_time'] / stats['chalupa']['avg_time']:.1f}x faster than ILP")

            if chalupa_acc >= 90:
                report.append("- **Excellent**: Chalupa produces near-optimal solutions in most cases")
            elif chalupa_acc >= 70:
                report.append("- **Good**: Chalupa is reliable for approximate solutions")
            else:
                report.append("- **Limited**: Chalupa struggles with these instance types")

        # Compare reduction effectiveness
        if 'ilp' in stats and 'reduced_ilp' in stats:
            report.append(f"\n### Reduction Effectiveness\n")
            time_improvement = stats['ilp']['avg_time'] / stats['reduced_ilp']['avg_time']
            report.append(f"- Reduction speeds up ILP by {time_improvement:.1f}x on average")
            report.append(f"- Maintains {stats['reduced_ilp']['accuracy']:.1f}% accuracy")

        report.append("\n## Recommendations\n")

        # Provide recommendations based on results
        if 'chalupa' in stats:
            if stats['chalupa']['accuracy'] >= 95:
                report.append("1. **Use Chalupa for large instances** where exact solutions are not critical")
            elif stats['chalupa']['accuracy'] >= 80:
                report.append("1. **Use Chalupa as initial bound** for branch-and-bound algorithms")
            else:
                report.append("1. **Prefer exact methods** as Chalupa accuracy is insufficient")

        report.append("2. **Always apply reductions** before running ILP for significant speedup")
        report.append("3. **Consider instance characteristics** when choosing algorithm")

        # Write report
        report_path = Path('results') / output_file
        report_path.parent.mkdir(exist_ok=True)
        with open(report_path, 'w') as f:
            f.write('\n'.join(report))

        print(f"Report saved to {report_path}")

        return '\n'.join(report)


def main():
    """Run WP1c analysis"""
    analyzer = WP1Analyzer()

    print("Loading results...")
    data = analyzer.load_all_results()

    print("Analyzing accuracy...")
    stats = analyzer.analyze_accuracy(data)

    print("\n=== WP1c Results ===")
    for method, stat in stats.items():
        print(f"\n{method.upper()}:")
        print(f"  Accuracy: {stat['accuracy']:.1f}%")
        print(f"  Avg deviation: {stat['avg_deviation']:.2f}")
        print(f"  Avg time: {stat['avg_time']:.3f}s")

    print("\nCreating comparison plots...")
    analyzer.create_comparison_plots(data, stats)

    print("\nGenerating report...")
    report = analyzer.generate_report(stats)

    print("\nWP1c Analysis complete!")
    print("Check 'figures/wp1c_comparison.png' and 'results/wp1c_analysis_report.md'")

    return stats, data


if __name__ == "__main__":
    main()