import os
import random
import json

def bool2int(b):
    return 1 if b else 0

def int2bool(i):
    return True if i == 1 else False

def manual_test(prompt):
    split = prompt.split(",")
    a = int(split[0])
    b = int(split[1])
    c = int(split[2])
    d = int(split[3])
    e = int(split[4])
    
    A_b = int2bool(a)
    B_b = int2bool(b)
    C_b = int2bool(c)
    D_b = int2bool(d)
    E_b = int2bool(e)
   
    # Apply structural equations:
    # B = A XOR C
    new_B = A_b ^ C_b
    # C = NOT D
    new_C = not D_b
    # D = B AND E
    new_D = B_b and E_b
    # E = A OR C
    new_E = A_b or new_C

    # Convert back to int
    A_str = str(a)
    B_str = str(b)
    C_str = str(c)
    D_str = str(d)
    E_str = str(e)

    new_B_str = str(bool2int(new_B))
    new_C_str = str(bool2int(new_C))
    new_D_str = str(bool2int(new_D))
    new_E_str = str(bool2int(new_E))
    old_sum = a + b + c + d + e
    new_sum = new_B + new_C + new_D + new_E 
     # Build textual scenario
    # Format it as a short piece of text that the model can do next-token prediction on.
    text_before = (f"{A_str},{B_str},{C_str},{D_str},{E_str}")
    # We'll ask the model to output: "B -> 0, C -> 1, D -> 0, E -> 1" etc.
    text_after = (f"\n{new_B_str},{new_C_str},{new_D_str},{new_E_str}\n{old_sum} - {new_sum}\n")
    # We'll return a single text example that we can treat as one training instance
    # Possibly we want the model to predict the text_after line given text_before
    # We'll store them together in a single JSON line.
    return {
        "prompt": text_before,
        "completion": text_after
    }

def generate_example(to_omit_list=None):
    """
    Generates a single example (scenario) with random initial states for A, B, C, D, E,
    then applies the structural equations, and returns a textual representation.
    """
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
    
    text_before = (f"{A_str},{B_str},{C_str},{D_str},{E_str}")
    
    # If we want to omit certain scenarios, we can do that here  
    if (to_omit_list is not None) and (text_before in to_omit_list):
        return

    # Convert to bool for easier logic
    A_b = int2bool(A_init)
    B_b = int2bool(B_init)
    C_b = int2bool(C_init)
    D_b = int2bool(D_init)
    E_b = int2bool(E_init)

    # Apply structural equations:
    # B = A XOR C
    new_B = A_b ^ C_b
    # C = NOT D
    new_C = not D_b
    # D = B AND E
    new_D = B_b and E_b
    # E = A OR C
    new_E = A_b or new_C

    # Convert back to int
    
    old_sum = A_init + B_init + C_init + D_init + E_init

    new_B_str = str(bool2int(new_B))
    new_C_str = str(bool2int(new_C))
    new_D_str = str(bool2int(new_D))
    new_E_str = str(bool2int(new_E))
    new_sum = new_B + new_C + new_D + new_E 

    # Build textual scenario
    # Format it as a short piece of text that the model can do next-token prediction on.
    
    # We'll ask the model to output: "B -> 0, C -> 1, D -> 0, E -> 1" etc.
    text_after = (f"\n{new_B_str},{new_C_str},{new_D_str},{new_E_str}\n{old_sum} - {new_sum}\n")

    # We'll return a single text example that we can treat as one training instance
    # Possibly we want the model to predict the text_after line given text_before
    # We'll store them together in a single JSON line.
    return {
        "prompt": text_before,
        "completion": text_after
    }

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


