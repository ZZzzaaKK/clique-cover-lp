"""
Utilities for loading and saving graph data.
"""
import os
import json
import pickle
import networkx as nx
from typing import Dict, List, Tuple, Optional

def save_test_case(G_original, G_perturbed, communities, stats_original, stats_perturbed,
                  case_name, output_dir="data"):
    """
    Save a test case to disk.

    Args:
        G_original: Original graph with disjoint cliques
        G_perturbed: Perturbed graph
        communities: Dictionary mapping nodes to their community id
        stats_original: Statistics for the original graph
        stats_perturbed: Statistics for the perturbed graph
        case_name: Identifier for this test case
        output_dir: Directory to save the data
    """
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

def load_test_case(case_name, data_dir="data"):
    """
    Load a test case from disk.

    Args:
        case_name: Identifier for the test case
        data_dir: Directory where the data is stored

    Returns:
        tuple: (original_graph, perturbed_graph, communities, stats_original, stats_perturbed)
    """
    # Load graphs
    with open(f"{data_dir}/{case_name}_original.pkl", "rb") as f:
        G_original = pickle.load(f)
    with open(f"{data_dir}/{case_name}_perturbed.pkl", "rb") as f:
        G_perturbed = pickle.load(f)

    # Load metadata
    with open(f"{data_dir}/{case_name}_metadata.json", "r") as f:
        metadata = json.load(f)

    # Convert community keys back to integers
    communities = {int(k): v for k, v in metadata["communities"].items()}

    return G_original, G_perturbed, communities, metadata["stats_original"], metadata["stats_perturbed"]

def list_available_test_cases(data_dir="data"):
    """
    List all available test cases in the data directory.

    Args:
        data_dir: Directory where the data is stored

    Returns:
        list: Names of available test cases
    """
    if not os.path.exists(data_dir):
        return []

    # Look for metadata files and extract case names
    case_names = set()
    for filename in os.listdir(data_dir):
        if filename.endswith("_metadata.json"):
            case_name = filename.replace("_metadata.json", "")
            case_names.add(case_name)

    return sorted(list(case_names))
