import pandas as pd
import matplotlib.pyplot as plt
import re
import sys

def parse_results(filepath):
    results = []
    algorithm = filepath.split('/')[-1].replace('perturbed_', '').replace('.txt', '')
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
    df = pd.concat([df, params_df], axis=1)

    # Add a problem size column
    df['problem_size'] = df.apply(lambda row: row['n'] if row['type'] == 'uniform' else row['num_cliques'] * row['max_clique_size'], axis=1)

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
    for name, group in df.groupby('algorithm'):
        plt.scatter(group['problem_size'], group['time'], label=name, alpha=0.6)
    plt.title('Time Taken vs. Problem Size')
    plt.xlabel('Problem Size (n for uniform, cliques*max_size for skewed)')
    plt.ylabel('Time Taken (s)')
    plt.yscale('log')
    plt.legend()
    plt.grid(True, which="both", ls="--")
    plt.savefig('results/analyses/time_vs_size.png')
    plt.close()

    # Plot 4: Deviation from actual
    plt.figure(figsize=(10, 6))
    df.boxplot(column='deviation', by='algorithm')
    plt.title('Deviation from Actual by Algorithm')
    plt.suptitle('')
    plt.xlabel('Algorithm')
    plt.ylabel('Deviation')
    plt.savefig('results/analyses/deviation_comparison.png')
    plt.close()

    print("Plots generated in results/analyses/")

if __name__ == '__main__':
    main()
