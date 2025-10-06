import pandas as pd
import matplotlib.pyplot as plt
import re
import sys
import seaborn as sns
import os
from pathlib import Path


def parse_results(filepath):
    """Parse results file and extract algorithm results"""
    results = []
    algorithm = os.path.basename(filepath).replace(".txt", "")
    with open(filepath, "r") as f:
        content = f.read()

    blocks = content.strip().split("------------------------------")
    for block in blocks:
        if not block.strip() or "Test Results for" in block or "Summary" in block:
            continue

        data = {"algorithm": algorithm}
        lines = block.strip().split("\n")

        for line in lines:
            if not line:
                continue

            parts = line.split(": ")
            if len(parts) < 2:
                continue

            key = parts[0].strip()
            value = ": ".join(parts[1:]).strip()

            if key == "File":
                data["file"] = value
            elif key == "Predicted":
                data["predicted"] = int(value)
            elif key == "Actual":
                data["actual"] = int(value)
            elif key == "Deviation":
                data["deviation"] = int(value)
            elif key == "Correct":
                data["correct"] = value == "True"
            elif key == "Time taken":
                data["time"] = float(value)

        if "file" in data:
            results.append(data)

    return results


def derive_graph_directory(results_filepath):
    """
    Derive the corresponding test_graphs directory from a results file path.

    Examples:
        results/raw/curated/20-29/chalupa.txt -> test_graphs/curated/20-29
        results/raw/curated/ilp.txt -> test_graphs/curated
        results/raw/generated/perturbed/reduced_ilp.txt -> test_graphs/generated/perturbed

    Args:
        results_filepath: Path to results file

    Returns:
        Path to corresponding test_graphs directory
    """
    path = Path(results_filepath)
    parts = path.parts

    # Find 'raw' in the path (results/raw/...)
    if "raw" in parts:
        idx = parts.index("raw")
        # Get all parts after 'raw' except the filename
        relative_parts = parts[idx + 1 : -1]  # Exclude filename

        # Build test_graphs path
        graph_path = Path("test_graphs")
        for part in relative_parts:
            graph_path = graph_path / part

        return str(graph_path)

    # Fallback: if pattern doesn't match, return None
    print(f"Warning: Could not derive graph directory from {results_filepath}")
    return None


def find_graph_file(filename, search_path):
    """
    Find a graph file by searching a directory and its subdirectories.

    Args:
        filename: Name of the graph file to find
        search_path: Base directory to search in

    Returns:
        Full path to the graph file, or None if not found
    """
    if not search_path or not os.path.exists(search_path):
        return None

    search_path = Path(search_path)

    # Search in the directory and all subdirectories
    for graph_file in search_path.rglob(filename):
        if graph_file.is_file():
            return str(graph_file)

    return None


def parse_graph_file(filepath):
    """Extract properties from a graph file"""
    properties = {
        "vertices": None,
        "edges": None,
        "density": None,
        "clique_number": None,
        "chromatic_number": None,
        "degeneracy": None,
        "average_degree": None,
        "connected": None,
    }

    if not os.path.exists(filepath):
        return properties

    try:
        with open(filepath, "r") as f:
            content = f.read()

        # Count vertices by finding the highest numbered vertex
        vertex_pattern = re.findall(r"^(\d+):", content, re.MULTILINE)
        if vertex_pattern:
            properties["vertices"] = max(map(int, vertex_pattern))

        # Extract properties from the metadata section
        property_patterns = {
            "density": r"Density: ([\d.]+)",
            "clique_number": r"Clique Number: (\d+)",
            "chromatic_number": r"Chromatic Number: (\d+)",
            "degeneracy": r"Degeneracy: (\d+)",
            "average_degree": r"Average Degree: ([\d.]+)",
            "connected": r"Connected: (Yes|No)",
        }

        for prop, pattern in property_patterns.items():
            match = re.search(pattern, content)
            if match:
                try:
                    value = match.group(1)
                    if prop == "connected":
                        properties[prop] = value == "Yes"
                    elif "." in value:
                        properties[prop] = float(value)
                    else:
                        properties[prop] = int(value)
                except (ValueError, IndexError):
                    pass

        # Count edges if not in metadata
        if properties["edges"] is None:
            edge_count = 0
            lines = content.split("\n")
            for line in lines:
                if ":" in line and not any(
                    keyword in line
                    for keyword in [
                        "Acyclic",
                        "Algebraic",
                        "Average",
                        "Bipartite",
                        "Chromatic",
                        "Circumference",
                        "Claw",
                        "Clique",
                        "Connected",
                        "Degeneracy",
                    ]
                ):
                    parts = line.split(":")
                    if len(parts) == 2 and parts[1].strip():
                        neighbors = parts[1].strip().split()
                        edge_count += len(neighbors)
            properties["edges"] = edge_count // 2  # Each edge counted twice

    except Exception as e:
        print(f"Error parsing graph file {filepath}: {e}")

    return properties


def main():
    if len(sys.argv) < 2:
        print("Usage: python src/comparison.py <results_file1> <results_file2> ...")
        print("\nExample:")
        print(
            "  python src/comparison.py results/raw/curated/20-29/chalupa.txt results/raw/curated/20-29/ilp.txt"
        )
        print("  python src/comparison.py results/raw/curated/*.txt")
        return

    results_files = sys.argv[1:]
    all_results = []

    for results_file in results_files:
        print(f"Processing {results_file}...")

        # Parse the results
        file_results = parse_results(results_file)

        # Derive where to look for graph files
        graph_directory = derive_graph_directory(results_file)

        if not graph_directory:
            print(f"  Warning: Could not determine graph directory, skipping...")
            continue

        if not os.path.exists(graph_directory):
            print(
                f"  Warning: Graph directory {graph_directory} does not exist, skipping..."
            )
            continue

        print(f"  Looking for graphs in: {graph_directory}")

        # Enrich each result with graph properties
        found_count = 0
        for result in file_results:
            graph_filename = result["file"]
            graph_path = find_graph_file(graph_filename, graph_directory)

            if graph_path:
                graph_props = parse_graph_file(graph_path)
                result.update(graph_props)
                result["graph_path"] = graph_path
                found_count += 1
            else:
                print(f"  Warning: Could not find graph file {graph_filename}")
                result["graph_path"] = None

        print(f"  Found {found_count}/{len(file_results)} graph files")
        all_results.extend(file_results)

    if not all_results:
        print("\nNo results found!")
        return

    # Create DataFrame
    df = pd.DataFrame(all_results)

    # Filter out rows where we couldn't find the graph file
    original_count = len(df)
    df = df.dropna(subset=["vertices"])
    filtered_count = len(df)

    if filtered_count < original_count:
        print(
            f"\nFiltered out {original_count - filtered_count} results without graph data"
        )

    if df.empty:
        print("No valid results with graph data found!")
        return

    print(f"\nLoaded {len(df)} results from {len(results_files)} files")
    print(f"Algorithms: {sorted(df['algorithm'].unique())}")
    print(f"Vertex counts: {int(df['vertices'].min())}-{int(df['vertices'].max())}")

    # Create output directory
    os.makedirs("results/analyses", exist_ok=True)

    # Generate plots
    generate_plots(df)
    print("\nPlots generated in results/analyses/")


def generate_plots(df):
    """Generate all comparison plots"""

    # Set style
    sns.set_style("whitegrid")

    # Plot 1: Time taken by algorithm
    plt.figure(figsize=(10, 6))
    time_means = df.groupby("algorithm")["time"].mean().sort_values()
    time_means.plot(kind="bar")
    plt.title("Average Time Taken by Algorithm")
    plt.ylabel("Average Time (s)")
    plt.xlabel("Algorithm")
    plt.yscale("log")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig("results/analyses/time_comparison.png", dpi=150)
    plt.close()

    # Plot 2: Correctness by algorithm
    plt.figure(figsize=(10, 6))
    correctness_data = (
        df.groupby("algorithm")["correct"].mean().sort_values(ascending=False)
    )
    correctness_data.plot(kind="bar", color="steelblue")
    plt.title("Correctness Rate by Algorithm")
    plt.ylabel("Correctness Rate")
    plt.xlabel("Algorithm")
    plt.ylim(0, 1.05)
    plt.axhline(y=1.0, color="green", linestyle="--", alpha=0.5, label="100%")
    plt.xticks(rotation=45, ha="right")
    plt.legend()
    plt.tight_layout()
    plt.savefig("results/analyses/correctness_comparison.png", dpi=150)
    plt.close()

    # Plot 3: Time vs Problem Size (vertices)
    plt.figure(figsize=(12, 8))
    for algo in sorted(df["algorithm"].unique()):
        algo_data = df[df["algorithm"] == algo]
        plt.scatter(
            algo_data["vertices"], algo_data["time"], alpha=0.6, label=algo, s=50
        )
    plt.title("Time Taken vs. Number of Vertices")
    plt.xlabel("Number of Vertices")
    plt.ylabel("Time Taken (s)")
    plt.yscale("log")
    plt.legend()
    plt.grid(True, which="both", ls="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig("results/analyses/time_vs_vertices.png", dpi=150)
    plt.close()

    # Plot 4: Deviation from actual
    plt.figure(figsize=(12, 6))
    # Only include results with valid deviations
    valid_dev = df[df["deviation"].notna()]
    if not valid_dev.empty:
        sns.boxplot(data=valid_dev, x="algorithm", y="deviation", palette="Set2")
        plt.title("Deviation from Actual Clique Cover by Algorithm")
        plt.xlabel("Algorithm")
        plt.ylabel("Deviation")
        plt.axhline(y=0, color="green", linestyle="--", alpha=0.5, label="Perfect")
        plt.xticks(rotation=45, ha="right")
        plt.legend()
        plt.tight_layout()
        plt.savefig("results/analyses/deviation_comparison.png", dpi=150)
        plt.close()

    # Plot 5: Time vs Density
    if "density" in df.columns and df["density"].notna().any():
        plt.figure(figsize=(12, 8))
        for algo in sorted(df["algorithm"].unique()):
            algo_data = df[(df["algorithm"] == algo) & (df["density"].notna())]
            if not algo_data.empty:
                plt.scatter(
                    algo_data["density"], algo_data["time"], alpha=0.6, label=algo, s=50
                )
        plt.title("Time Taken vs. Graph Density")
        plt.xlabel("Graph Density")
        plt.ylabel("Time Taken (s)")
        plt.yscale("log")
        plt.legend()
        plt.grid(True, which="both", ls="--", alpha=0.3)
        plt.tight_layout()
        plt.savefig("results/analyses/time_vs_density.png", dpi=150)
        plt.close()

    # Plot 6: Time vs Clique Number
    if "clique_number" in df.columns and df["clique_number"].notna().any():
        plt.figure(figsize=(12, 8))
        for algo in sorted(df["algorithm"].unique()):
            algo_data = df[(df["algorithm"] == algo) & (df["clique_number"].notna())]
            if not algo_data.empty:
                plt.scatter(
                    algo_data["clique_number"],
                    algo_data["time"],
                    alpha=0.6,
                    label=algo,
                    s=50,
                )
        plt.title("Time Taken vs. Clique Number")
        plt.xlabel("Clique Number")
        plt.ylabel("Time Taken (s)")
        plt.yscale("log")
        plt.legend()
        plt.grid(True, which="both", ls="--", alpha=0.3)
        plt.tight_layout()
        plt.savefig("results/analyses/time_vs_clique_number.png", dpi=150)
        plt.close()

    # Plot 7: Accuracy vs Vertices (showing how accuracy degrades with size)
    plt.figure(figsize=(12, 8))
    # Bin vertices into groups for better visualization
    df["vertex_bin"] = pd.cut(df["vertices"], bins=10)
    accuracy_by_size = (
        df.groupby(["algorithm", "vertex_bin"])["correct"].mean().reset_index()
    )

    for algo in sorted(df["algorithm"].unique()):
        algo_data = accuracy_by_size[accuracy_by_size["algorithm"] == algo]
        if not algo_data.empty:
            # Use the midpoint of each bin for x-axis
            x_values = [interval.mid for interval in algo_data["vertex_bin"]]
            plt.plot(x_values, algo_data["correct"], marker="o", label=algo, alpha=0.7)

    plt.title("Correctness vs. Graph Size")
    plt.xlabel("Number of Vertices")
    plt.ylabel("Correctness Rate")
    plt.ylim(0, 1.05)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("results/analyses/correctness_vs_size.png", dpi=150)
    plt.close()


if __name__ == "__main__":
    main()
