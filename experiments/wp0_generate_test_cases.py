"""
Generate and save test cases for clique covering experiments.
"""

import os
import json
import pickle
from src.data.simulator import GraphGenerator, GraphConfig
from src.visualization.plot import visualize_graph

def save_test_case(G_original, G_perturbed, communities, stats_original, stats_perturbed, case_name, output_dir="data"):
    """Save a test case to disk."""
    os.makedirs(output_dir, exist_ok=True)

    # Save graphs as pickle files
    with open(f"{output_dir}/{case_name}_original.pkl", "wb") as f:
        pickle.dump(G_original, f)
    with open(f"{output_dir}/{case_name}_perturbed.pkl", "wb") as f:
        pickle.dump(G_perturbed, f)

    # Save metadata as JSON
    metadata = {
        "communities": {str(k): v for k, v in communities.items()},
        "stats_original": stats_original,
        "stats_perturbed": stats_perturbed
    }
    with open(f"{output_dir}/{case_name}_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Saved test case {case_name} to {output_dir}")

def generate_test_suite():
    """Generate a comprehensive test suite with different parameters."""
    # Generate uniform distribution examples
    for size in [5, 10, 20]:
        for num_cliques in [3, 5, 10]:
            for removal_prob in [0.1, 0.3]:
                config = GraphConfig(
                    num_cliques=num_cliques,
                    distribution_type="uniform",
                    uniform_size=size,
                    edge_removal_prob=removal_prob,
                    edge_addition_prob=removal_prob/4
                )

                result = GraphGenerator.generate_test_case(config)
                G_original, G_perturbed, communities, stats_original, stats_perturbed = result

                case_name = f"uniform_n{num_cliques}_s{size}_r{int(removal_prob*100)}"
                save_test_case(G_original, G_perturbed, communities, stats_original, stats_perturbed, case_name)

                # Visualize the first few examples
                if size <= 10 and num_cliques <= 5:
                    visualize_graph(G_original, communities, "original", case_name)
                    visualize_graph(G_perturbed, communities, "perturbed", case_name)

    # Generate skewed distribution examples
    for min_size in [2, 6, 10]:
        for max_size in [14, 18, 22]:
            for num_cliques in [3, 5, 10]:
                for removal_prob in [0.1, 0.3]:
                    config = GraphConfig(
                        num_cliques=num_cliques,
                        distribution_type="skewed",
                        min_size=min_size,
                        max_size=max_size,
                        edge_removal_prob=removal_prob,
                        edge_addition_prob=removal_prob/4
                    )

                    result = GraphGenerator.generate_test_case(config)
                    G_original, G_perturbed, communities, stats_original, stats_perturbed = result

                    case_name = f"skewed_n{num_cliques}_min{min_size}_max{max_size}_r{int(removal_prob*100)}"
                    save_test_case(G_original, G_perturbed, communities, stats_original, stats_perturbed, case_name)

                    visualize_graph(G_original, communities, "original", case_name)
                    visualize_graph(G_perturbed, communities, "perturbed", case_name)

if __name__ == "__main__":
    generate_test_suite()
