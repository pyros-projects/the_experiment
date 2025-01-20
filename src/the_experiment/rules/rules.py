
RULES="""new_A = A XOR C
new_B = NOT D  
new_C = B AND E
new_D = A OR new_B
"""

def bool2int(b:bool):
    return 1 if b else 0

def int2bool(i:int):
    return True if i == 1 else False

def prompt_to_completion(input_text: str) -> dict:
    """THE TRUTH"""
    split = input_text.split(",")
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