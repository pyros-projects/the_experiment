import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from devtools import debug

from the_experiment.dataset import bool2int
DEFAULT_PATH = "./out/tiny-gpt2-causal/final"



def load_model():
    model_path = DEFAULT_PATH
    tokenizer = GPT2TokenizerFast.from_pretrained(model_path)
    model = GPT2LMHeadModel.from_pretrained(model_path)
    model.eval()
    return model, tokenizer

def eval_model(prompt_text: str) -> str:
    try:
        model, tokenizer = load_model()
        input_ids = tokenizer.encode(prompt_text, return_tensors="pt")
        
        with torch.no_grad():
            output = model.generate(
                input_ids,
                max_length=64,
                num_return_sequences=1,
                do_sample=False
            )
            
        output = tokenizer.decode(output[0])
        return str(output)
    except Exception as e:
        return None

    

def eval_model_bool(a,b,c,d,e) -> str:
    a = str(bool2int(a))
    b = str(bool2int(b))
    c = str(bool2int(c))
    d = str(bool2int(d))
    e = str(bool2int(e))
    prompt = (f"{a},{b},{c},{d},{e}")
    return eval_model(prompt)

def eval_model_int(a,b,c,d,e) -> str:
    a = str(a)
    b = str(b)
    c = str(c)
    d = str(d)
    e = str(e)
    prompt = (f"{a},{b},{c},{d},{e}")
    return eval_model(prompt)