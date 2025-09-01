import pandas as pd
import matplotlib.pyplot as plt
import re
import sys
import seaborn as sns
import os

def parse_results(filepath):
    results = []
    algorithm = os.path.basename(filepath).replace('perturbed_', '').replace('.txt', '')
    with open(filepath, 'r') as f:
        content = f.read()

    blocks = content.strip().split('------------------------------')
    for block in blocks:
        if not block.strip() or "Test Results for" in block or "Summary" in block:
            continue

        data = {'algorithm': algorithm}
        lines = block.strip().split('\n')

        for line in lines:
            if not line:
                continue

            parts = line.split(': ')
            if len(parts) < 2:
                continue

            key = parts[0].strip()
            value = ": ".join(parts[1:]).strip()

            if key == 'File':
                data['file'] = value
            elif key == 'Predicted':
                data['predicted'] = int(value)
            elif key == 'Actual':
                data['actual'] = int(value)
            elif key == 'Deviation':
                data['deviation'] = int(value)
            elif key == 'Correct':
                data['correct'] = value == 'True'
            elif key == 'Time taken':
                data['time'] = float(value)

        if "file" in data:
            results.append(data)

    return results

def extract_params(filename):
    match_uniform = re.match(r'uniform_n(\d+)_s(\d+)_r(\d+).txt', filename)
    match_skewed = re.match(r'skewed_cliques(\d+)_min(\d+)_max(\d+)_perturbation(\d+).txt', filename)

    if match_uniform:
        n, s, r = map(int, match_uniform.groups())
        return {'type': 'uniform', 'n': n, 's': s, 'r': r, 'perturbation': 0}
    elif match_skewed:
        cliques, min_size, max_size, p = map(int, match_skewed.groups())
        return {'type': 'skewed', 'num_cliques': cliques, 'min_clique_size': min_size, 'max_clique_size': max_size, 'perturbation': p}
    return {}


graph_properties_cache = {}

def get_graph_properties(graph_filename):
    if graph_filename in graph_properties_cache:
        return graph_properties_cache[graph_filename]

    for root, _, files in os.walk('test_graphs'):
        if graph_filename in files:
            filepath = os.path.join(root, graph_filename)
            with open(filepath, 'r') as f:
                content = f.read()

            density_match = re.search(r'Density: (\d+\.?\d*)', content)
            vertices_match = re.search(r'Number of Vertices: (\d+)', content)

            properties = {
                'density': float(density_match.group(1)) if density_match else None,
                'vertices': int(vertices_match.group(1)) if vertices_match else None
            }
            graph_properties_cache[graph_filename] = properties
            return properties

    graph_properties_cache[graph_filename] = {'density': None, 'vertices': None}
    return graph_properties_cache[graph_filename]

def main():
    if len(sys.argv) < 2:
        print("Usage: python src/comparison.py <file1> <file2> ...")
        return
    files = sys.argv[1:]

    all_results = []
    for f in files:
        all_results.extend(parse_results(f))

    df = pd.DataFrame(all_results)

    params_df = df['file'].apply(extract_params).apply(pd.Series)
    graph_props_df = df['file'].apply(get_graph_properties).apply(pd.Series)
    df = pd.concat([df, params_df, graph_props_df], axis=1)

    # Add a problem size and cliques column
    df['problem_size'] = df.apply(lambda row: row['n'] if row['type'] == 'uniform' else row['num_cliques'] * row['max_clique_size'], axis=1)
    df['num_cliques'] = df.apply(lambda row: row['s'] if row['type'] == 'uniform' else row['num_cliques'], axis=1)


    # Plot 1: Time taken by algorithm
    plt.figure(figsize=(10, 6))
    df.groupby('algorithm')['time'].mean().plot(kind='bar')
    plt.title('Average Time Taken by Algorithm')
    plt.ylabel('Average Time (s)')
    plt.yscale('log')
    plt.savefig('results/analyses/time_comparison.png')
    plt.close()

    # Plot 2: Correctness by algorithm
    plt.figure(figsize=(10, 6))
    df.groupby('algorithm')['correct'].value_counts(normalize=True).unstack().plot(kind='bar', stacked=True)
    plt.title('Correctness by Algorithm')
    plt.ylabel('Percentage')
    plt.savefig('results/analyses/correctness_comparison.png')
    plt.close()

    # Plot 3: Time vs Problem Size
    plt.figure(figsize=(12, 8))
    sns.lineplot(data=df, x='problem_size', y='time', hue='algorithm')
    plt.title('Time Taken vs. Problem Size')
    plt.xlabel('Problem Size (n for uniform, cliques*max_size for skewed)')
    plt.ylabel('Time Taken (s)')
    plt.yscale('log')
    plt.grid(True, which="both", ls="--")
    plt.savefig('results/analyses/time_vs_size.png')
    plt.close()

    # Plot 4: Deviation from actual
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df, x='algorithm', y='deviation')
    plt.title('Deviation from Actual by Algorithm')
    plt.xlabel('Algorithm')
    plt.ylabel('Deviation')
    plt.savefig('results/analyses/deviation_comparison.png')
    plt.close()

    # Plot 5: Time vs Perturbation
    plt.figure(figsize=(12, 8))
    sns.lineplot(data=df[df['type'] == 'skewed'], x='perturbation', y='time', hue='algorithm')
    plt.title('Time Taken vs. Perturbation (Skewed Graphs)')
    plt.xlabel('Perturbation')
    plt.ylabel('Average Time (s)')
    plt.yscale('log')
    plt.grid(True, which="both", ls="--")
    plt.savefig('results/analyses/time_vs_perturbation.png')
    plt.close()

    # Plot 6: Correctness vs Perturbation
    plt.figure(figsize=(12, 8))
    correctness_perturbation = df[df['type'] == 'skewed'].groupby(['algorithm', 'perturbation'])['correct'].mean().reset_index()
    sns.lineplot(data=correctness_perturbation, x='perturbation', y='correct', hue='algorithm')
    plt.title('Correctness vs. Perturbation (Skewed Graphs)')
    plt.xlabel('Perturbation')
    plt.ylabel('Correctness (%)')
    plt.grid(True, which="both", ls="--")
    plt.savefig('results/analyses/correctness_vs_perturbation.png')
    plt.close()

    # --- Density Plots ---

    # Plot 7: Time vs. Density
    plt.figure(figsize=(12, 8))
    sns.lineplot(data=df, x='density', y='time', hue='algorithm')
    plt.title('Time Taken vs. Graph Density')
    plt.xlabel('Graph Density')
    plt.ylabel('Average Time (s)')
    plt.yscale('log')
    plt.grid(True, which="both", ls="--")
    plt.savefig('results/analyses/time_vs_density.png')
    plt.close()

    # Plot 8: Correctness vs. Density
    plt.figure(figsize=(12, 8))
    correctness_density = df.groupby(['algorithm', 'density'])['correct'].mean().reset_index()
    sns.lineplot(data=correctness_density, x='density', y='correct', hue='algorithm')
    plt.title('Correctness vs. Graph Density')
    plt.xlabel('Graph Density')
    plt.ylabel('Correctness (%)')
    plt.grid(True, which="both", ls="--")
    plt.savefig('results/analyses/correctness_vs_density.png')
    plt.close()

    print("Plots generated in results/analyses/")

if __name__ == '__main__':
    main()
