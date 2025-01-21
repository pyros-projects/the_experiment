import os
import random
import json

from the_experiment.rules.rules import prompt_to_completion


def generate_example(to_omit_list=None):
    """
    Generates a single example (scenario) with random initial states for A, B, C, D, E,
    then applies the structural equations, and returns a textual representation.
    """

    max_attempts = 100  # Prevent infinite loops
    for _ in range(max_attempts):
        # Random initial states for A,B,C,D,E
        A_init = random.randint(0, 1)
        B_init = random.randint(0, 1)
        C_init = random.randint(0, 1)
        D_init = random.randint(0, 1)
        E_init = random.randint(0, 1)

        A_str = str(A_init)
        B_str = str(B_init)
        C_str = str(C_init)
        D_str = str(D_init)
        E_str = str(E_init)

        text_before = f"{A_str},{B_str},{C_str},{D_str},{E_str}"

        # If sequence is not in omit list (or there is no omit list), use it
        if to_omit_list is None or text_before not in to_omit_list:
            break
    else:
        print(
            f"Warning: Could not generate sequence not in omit_list after {max_attempts} attempts"
        )
        return None

    return prompt_to_completion(text_before)


def generate_dataset(n_examples, out_path, to_omit_list=None):
    data = []
    for _ in range(n_examples):
        ex = generate_example(to_omit_list)
        if ex is not None:
            data.append(ex)

    # Create necessary folders in out_path
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    with open(out_path, "w") as f:
        for d in data:
            f.write(json.dumps(d) + "\n")
