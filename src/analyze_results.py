import os
import re
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Tuple, Optional
import seaborn as sns
import sys
import argparse
from pathlib import Path

class GraphAnalyzer:
    def __init__(self):
        self.results_data = []
        self.graph_properties = {}

    def parse_results_file(self, results_file: str) -> List[Dict]:
        """Parse the results file to extract performance data."""
        results = []

        with open(results_file, 'r') as f:
            content = f.read()

        # Split by separator lines
        entries = re.split(r'-{30,}', content)

        for entry in entries:
            if 'File:' in entry and 'Time taken:' in entry:
                # Extract filename
                filename_match = re.search(r'File: (graph_\d+\.txt)', entry)
                if not filename_match:
                    continue
                filename = filename_match.group(1)

                # Extract metrics
                predicted_match = re.search(r'Predicted: (\d+)', entry)
                actual_match = re.search(r'Actual: (\d+)', entry)
                time_match = re.search(r'Time taken: ([\d.]+)', entry)
                correct_match = re.search(r'Correct: (True|False)', entry)

                if all([predicted_match, actual_match, time_match, correct_match]):
                    results.append({
                        'filename': filename,
                        'predicted': int(predicted_match.group(1)),
                        'actual': int(actual_match.group(1)),
                        'time': float(time_match.group(1)),
                        'correct': correct_match.group(1) == 'True',
                        'deviation': abs(int(predicted_match.group(1)) - int(actual_match.group(1)))
                    })

        return results

    def parse_graph_file(self, graph_file: str) -> Dict:
        """Parse graph file to extract properties."""
        properties = {}

        with open(graph_file, 'r') as f:
            content = f.read()

        # Parse adjacency list to count edges and vertices
        lines = content.strip().split('\n')
        adjacency_section = []
        properties_section = []

        # Split into adjacency list and properties
        in_properties = False
        for line in lines:
            if ':' in line and not in_properties:
                if any(prop in line for prop in ['Acyclic:', 'Average Degree:', 'Bipartite:']):
                    in_properties = True
                    properties_section.append(line)
                else:
                    adjacency_section.append(line)
            elif in_properties:
                properties_section.append(line)

        # Count vertices and edges from adjacency list
        vertices = len(adjacency_section)
        edges = 0
        for line in adjacency_section:
            if ':' in line:
                neighbors = line.split(':')[1].strip()
                if neighbors:
                    edges += len(neighbors.split())
        edges = edges // 2  # Each edge is counted twice

        properties['vertices'] = vertices
        properties['edges'] = edges

        # Parse properties section
        for line in properties_section:
            if ':' in line:
                key, value = line.split(':', 1)
                key = key.strip()
                value = value.strip()

                # Convert to appropriate type
                if value == 'Yes':
                    properties[key] = True
                elif value == 'No':
                    properties[key] = False
                elif value == 'infinity' or value == 'undefined':
                    properties[key] = None
                else:
                    try:
                        if '.' in value:
                            properties[key] = float(value)
                        else:
                            properties[key] = int(value)
                    except ValueError:
                        properties[key] = value

        return properties

    def infer_graphs_dir(self, results_file: str) -> str:
        """Infer graphs directory from results filename."""
        # Extract filename without extension
        results_path = Path(results_file)
        filename_no_ext = results_path.stem

        # Parse pattern: [directory]_[algorithm].txt
        parts = filename_no_ext.split('_')
        if len(parts) >= 2:
            # Take everything except the last part (algorithm name)
            directory_name = '_'.join(parts[:-1])

            # Construct graphs directory path
            graphs_dir = f"test_graphs/curated/{directory_name}/"

            if os.path.exists(graphs_dir):
                return graphs_dir
            else:
                print(f"Warning: Inferred graphs directory '{graphs_dir}' does not exist")
                return None

        return None

    def load_all_data(self, results_file: str, graphs_dir: str = None):
        """Load all results and graph data."""
        # Parse results
        self.results_data = self.parse_results_file(results_file)

        # Infer graphs directory if not provided
        if graphs_dir is None:
            graphs_dir = self.infer_graphs_dir(results_file)
            if graphs_dir is None:
                print("Error: Could not infer graphs directory and none provided")
                return
            else:
                print(f"Inferred graphs directory: {graphs_dir}")

        # Load graph properties for each file
        for result in self.results_data:
            graph_file = os.path.join(graphs_dir, result['filename'])
            if os.path.exists(graph_file):
                properties = self.parse_graph_file(graph_file)
                self.graph_properties[result['filename']] = properties

                # Add properties to result data
                result.update(properties)
            else:
                print(f"Warning: Graph file not found: {graph_file}")

    def calculate_problem_size_metrics(self) -> Dict[str, List]:
        """Calculate different problem size metrics."""
        metrics = {
            'vertices': [],
            'edges': [],
            'density': [],
            'avg_degree': []
        }

        for result in self.results_data:
            if 'vertices' in result and 'edges' in result:
                vertices = result['vertices']
                edges = result['edges']

                metrics['vertices'].append(vertices)
                metrics['edges'].append(edges)

                # Calculate density
                max_edges = vertices * (vertices - 1) / 2
                density = edges / max_edges if max_edges > 0 else 0
                metrics['density'].append(density)

                # Calculate average degree
                avg_degree = 2 * edges / vertices if vertices > 0 else 0
                metrics['avg_degree'].append(avg_degree)

        return metrics

    def plot_runtime_analysis(self):
        """Create comprehensive runtime analysis plots."""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Chalupa Algorithm Runtime Analysis', fontsize=16)

        # Extract data
        times = [r['time'] for r in self.results_data]
        vertices = [r.get('vertices', 0) for r in self.results_data]
        edges = [r.get('edges', 0) for r in self.results_data]
        chromatic_numbers = [r['actual'] for r in self.results_data]
        densities = [r.get('Density', 0) for r in self.results_data if r.get('Density') is not None]

        # Plot 1: Runtime vs Vertices
        axes[0,0].scatter(vertices, times, alpha=0.7, color='blue')
        axes[0,0].set_xlabel('Number of Vertices')
        axes[0,0].set_ylabel('Runtime (seconds)')
        axes[0,0].set_title('Runtime vs Number of Vertices')

        # Add trend line
        if vertices and times:
            z = np.polyfit(vertices, times, 1)
            p = np.poly1d(z)
            axes[0,0].plot(vertices, p(vertices), "r--", alpha=0.8)

        # Plot 2: Runtime vs Edges
        axes[0,1].scatter(edges, times, alpha=0.7, color='green')
        axes[0,1].set_xlabel('Number of Edges')
        axes[0,1].set_ylabel('Runtime (seconds)')
        axes[0,1].set_title('Runtime vs Number of Edges')

        # Add trend line
        if edges and times:
            z = np.polyfit(edges, times, 1)
            p = np.poly1d(z)
            axes[0,1].plot(edges, p(edges), "r--", alpha=0.8)

        # Plot 3: Runtime vs Chromatic Number
        axes[0,2].scatter(chromatic_numbers, times, alpha=0.7, color='red')
        axes[0,2].set_xlabel('Chromatic Number')
        axes[0,2].set_ylabel('Runtime (seconds)')
        axes[0,2].set_title('Runtime vs Chromatic Number')

        # Plot 4: Runtime vs Density
        if densities and len(densities) == len(times):
            axes[1,0].scatter(densities, times, alpha=0.7, color='purple')
            axes[1,0].set_xlabel('Graph Density')
            axes[1,0].set_ylabel('Runtime (seconds)')
            axes[1,0].set_title('Runtime vs Graph Density')

        # Plot 5: Runtime distribution
        axes[1,1].hist(times, bins=15, alpha=0.7, color='orange')
        axes[1,1].set_xlabel('Runtime (seconds)')
        axes[1,1].set_ylabel('Frequency')
        axes[1,1].set_title('Runtime Distribution')
        axes[1,1].axvline(np.mean(times), color='red', linestyle='--', label=f'Mean: {np.mean(times):.3f}s')
        axes[1,1].legend()

        # Plot 6: Accuracy by problem size
        correct_results = [r for r in self.results_data if r['correct']]
        incorrect_results = [r for r in self.results_data if not r['correct']]

        if correct_results:
            correct_vertices = [r.get('vertices', 0) for r in correct_results]
            correct_times = [r['time'] for r in correct_results]
            axes[1,2].scatter(correct_vertices, correct_times, alpha=0.7, color='green', label='Correct', s=50)

        if incorrect_results:
            incorrect_vertices = [r.get('vertices', 0) for r in incorrect_results]
            incorrect_times = [r['time'] for r in incorrect_results]
            axes[1,2].scatter(incorrect_vertices, incorrect_times, alpha=0.7, color='red', label='Incorrect', s=50, marker='x')

        axes[1,2].set_xlabel('Number of Vertices')
        axes[1,2].set_ylabel('Runtime (seconds)')
        axes[1,2].set_title('Runtime vs Vertices (by Accuracy)')
        axes[1,2].legend()

        plt.tight_layout()
        return fig

    def plot_problem_size_analysis(self):
        """Analyze different problem size metrics."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Problem Size Analysis', fontsize=16)

        times = [r['time'] for r in self.results_data]
        vertices = [r.get('vertices', 0) for r in self.results_data]
        edges = [r.get('edges', 0) for r in self.results_data]

        # Calculate additional metrics
        problem_sizes = []
        for r in self.results_data:
            v = r.get('vertices', 0)
            e = r.get('edges', 0)
            # Different problem size measures to try
            size_measures = {
                'vertices': v,
                'edges': e,
                'v*e': v * e,
                'v^2': v ** 2,
                'e^2': e ** 2,
                'v+e': v + e
            }
            problem_sizes.append(size_measures)

        # Test different size measures
        measures = ['vertices', 'edges', 'v*e', 'v^2']

        for i, measure in enumerate(measures):
            ax = axes[i//2, i%2]
            size_values = [ps[measure] for ps in problem_sizes]

            ax.scatter(size_values, times, alpha=0.7)
            ax.set_xlabel(f'Problem Size ({measure})')
            ax.set_ylabel('Runtime (seconds)')
            ax.set_title(f'Runtime vs {measure}')

            # Calculate correlation
            correlation = np.corrcoef(size_values, times)[0,1] if len(size_values) > 1 else 0
            ax.text(0.05, 0.95, f'Correlation: {correlation:.3f}',
                   transform=ax.transAxes, bbox=dict(boxstyle="round", facecolor='wheat'))

            # Add trend line
            if len(size_values) > 1:
                z = np.polyfit(size_values, times, 1)
                p = np.poly1d(z)
                ax.plot(size_values, p(size_values), "r--", alpha=0.8)

        plt.tight_layout()
        return fig

    def print_summary_statistics(self):
        """Print summary statistics."""
        print("=== Chalupa Algorithm Performance Summary ===")
        print(f"Total graphs analyzed: {len(self.results_data)}")

        # Accuracy
        correct_count = sum(1 for r in self.results_data if r['correct'])
        accuracy = correct_count / len(self.results_data) * 100
        print(f"Accuracy: {correct_count}/{len(self.results_data)} ({accuracy:.1f}%)")

        # Runtime statistics
        times = [r['time'] for r in self.results_data]
        print(f"\nRuntime Statistics:")
        print(f"  Mean: {np.mean(times):.3f} seconds")
        print(f"  Median: {np.median(times):.3f} seconds")
        print(f"  Std Dev: {np.std(times):.3f} seconds")
        print(f"  Min: {min(times):.3f} seconds")
        print(f"  Max: {max(times):.3f} seconds")

        # Problem size statistics
        vertices = [r.get('vertices', 0) for r in self.results_data]
        edges = [r.get('edges', 0) for r in self.results_data]

        if vertices:
            print(f"\nProblem Size Statistics:")
            print(f"  Vertices - Mean: {np.mean(vertices):.1f}, Range: {min(vertices)}-{max(vertices)}")
            print(f"  Edges - Mean: {np.mean(edges):.1f}, Range: {min(edges)}-{max(edges)}")

        # Chromatic number statistics
        chromatic_numbers = [r['actual'] for r in self.results_data]
        print(f"  Chromatic Numbers - Range: {min(chromatic_numbers)}-{max(chromatic_numbers)}")

        # Correlation analysis
        print(f"\nCorrelation with Runtime:")
        if vertices and times:
            corr_vertices = np.corrcoef(vertices, times)[0,1]
            print(f"  Vertices: {corr_vertices:.3f}")

        if edges and times:
            corr_edges = np.corrcoef(edges, times)[0,1]
            print(f"  Edges: {corr_edges:.3f}")

        corr_chromatic = np.corrcoef(chromatic_numbers, times)[0,1]
        print(f"  Chromatic Number: {corr_chromatic:.3f}")

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Analyze Chalupa algorithm results')
    parser.add_argument('results_file', help='Path to results file (e.g., results/public/20-29_chalupa.txt)')
    parser.add_argument('graphs_dir', nargs='?', help='Path to graphs directory (optional, will be inferred if not provided)')

    args = parser.parse_args()

    # Check if results file exists
    if not os.path.exists(args.results_file):
        print(f"Error: Results file '{args.results_file}' not found")
        sys.exit(1)

    # Initialize analyzer
    analyzer = GraphAnalyzer()

    print(f"Loading data from: {args.results_file}")
    analyzer.load_all_data(args.results_file, args.graphs_dir)

    if not analyzer.results_data:
        print("Error: No data loaded")
        sys.exit(1)

    # Print summary statistics
    analyzer.print_summary_statistics()

    # Generate output filenames based on input
    results_path = Path(args.results_file)
    base_name = results_path.stem

    # Ensure results directory exists
    os.makedirs('results', exist_ok=True)

    runtime_output = f'results/analyses/{base_name}_runtime_analysis.png'
    problem_size_output = f'results/analyses/{base_name}_problem_size_analysis.png'

    # Create visualizations
    print(f"\nGenerating runtime analysis plots...")
    fig1 = analyzer.plot_runtime_analysis()
    fig1.savefig(runtime_output, dpi=300, bbox_inches='tight')
    print(f"Saved: {runtime_output}")

    print("Generating problem size analysis plots...")
    fig2 = analyzer.plot_problem_size_analysis()
    fig2.savefig(problem_size_output, dpi=300, bbox_inches='tight')
    print(f"Saved: {problem_size_output}")

    plt.show()
    print(f"\nAnalysis complete!")

if __name__ == "__main__":
    main()
