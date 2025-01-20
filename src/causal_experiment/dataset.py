import os
import random
import json
from devtools import debug

def bool2int(b:bool):
    return 1 if b else 0

def int2bool(i:int):
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
    # new_A = A XOR C
    new_A = A_b ^ C_b
    # new_B = NOT D
    new_B = not D_b
    # new_C = B AND E
    new_C = B_b and E_b
    # new_D = A OR new_B
    new_D = A_b or new_B

    new_A_str = str(bool2int(new_A))
    new_B_str = str(bool2int(new_B))
    new_C_str = str(bool2int(new_C))
    new_D_str = str(bool2int(new_D))
    
    old_sum = a + b + c + d + e
    new_sum = new_B + new_C + new_D + new_A

    text_before = (f"{a},{b},{c},{d},{e}")
    text_after = (f"\n{new_A_str},{new_B_str},{new_C_str},{new_D_str}\n{old_sum} - {new_sum}\n")

    return {
        "prompt": text_before,
        "completion": text_after
    }

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
        
        text_before = (f"{A_str},{B_str},{C_str},{D_str},{E_str}")
        
        # If sequence is not in omit list (or there is no omit list), use it
        if to_omit_list is None or text_before not in to_omit_list:
            break
    else:
        print(f"Warning: Could not generate sequence not in omit_list after {max_attempts} attempts")
        return None

    # Convert to bool for easier logic
    A_b = int2bool(A_init)
    B_b = int2bool(B_init)
    C_b = int2bool(C_init)
    D_b = int2bool(D_init)
    E_b = int2bool(E_init)

    # Apply structural equations:
    # new_A = A XOR C
    new_A = A_b ^ C_b
    # new_B = NOT D
    new_B = not D_b
    # new_C = B AND E
    new_C = B_b and E_b
    # new_D = A OR new_B
    new_D = A_b or new_B

    
    old_sum = A_init + B_init + C_init + D_init + E_init
    new_sum = new_A + new_B + new_C + new_D

    new_A_str = str(bool2int(new_A))
    new_B_str = str(bool2int(new_B))
    new_C_str = str(bool2int(new_C))
    new_D_str = str(bool2int(new_D))
    

    text_after = (f"\n{new_A_str},{new_B_str},{new_C_str},{new_D_str}\n{old_sum} - {new_sum}\n")
    
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


