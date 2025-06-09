import numpy as np

def random_permutation(vertices):
    """
    Returns a random permutation (shuffle) of the input list.

    Args:
        vertices: A list or array to be randomly permuted

    Returns:
        A new list with the same elements in random order
    """
    rng = np.random.default_rng()
    permuted = list(vertices)  # Convert to list and make a copy
    rng.shuffle(permuted)  # Shuffle in-place
    return permuted


def uniformly_random(lower_bound, list):
    rng = np.random.default_rng()
    subarray = list[lower_bound:]
    return rng.choice(subarray, size=1)[0]

def jump(element, new_index, list):
    result = list.copy()
    try:
        result.remove(element)
    except Exception as e:
        print("Jump went wrong! No element was removed, which shouldn't happen as far as I understand.")
        print(f"Error message: {e}")
    finally:
        result.insert(new_index, element)

    return result
