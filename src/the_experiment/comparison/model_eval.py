import torch
from the_experiment.comparison.load_model import load_model , load_rnn, load_models
from the_experiment.dataset import bool2int
from typing import List, Dict, Optional, Tuple

class ModelEvaluator:
    def __init__(self):
        self.model, self.rnn_model, self.cnn_model, self.tokenizer = load_models()

    def reload_models(self):
        self.model, self.rnn_model, self.cnn_model, self.tokenizer = load_models()

model_evaluator = ModelEvaluator()

def eval_model(prompt_text: str) -> str:
    try:

        input_ids = model_evaluator.tokenizer.encode(prompt_text, return_tensors="pt")
        
        with torch.no_grad():
            output = model_evaluator.model.generate(
                input_ids,
                max_length=64,
                num_return_sequences=1,
                do_sample=False
            )
            
        output = model_evaluator.tokenizer.decode(output[0])
        return str(output)
    except Exception as e:
        return None

def eval_rnn(prompt_text: str) -> str:
    try:
        model = model_evaluator.rnn_model
        with torch.no_grad():
            # Tokenize input
            inputs = model_evaluator.tokenizer(prompt_text, return_tensors="pt")
            if torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}
            
            # Generate output token by token
            max_length = 100
            generated_ids = []
            # Initialize hidden state
            batch_size = 1
            num_layers = 2
            hidden_dim = 128
            h0 = torch.zeros(num_layers, batch_size, hidden_dim)
            c0 = torch.zeros(num_layers, batch_size, hidden_dim)
            if torch.cuda.is_available():
                h0 = h0.cuda()
                c0 = c0.cuda()
            hidden = (h0, c0)
            
            # Start with input sequence
            current_ids = inputs["input_ids"]
            
            for _ in range(max_length):
                # Get logits and hidden state
                logits, hidden = model(current_ids, hidden)
                
                # Get next token probabilities from the last position
                next_token_logits = logits[:, -1, :]
                
                # Sample next token
                next_token = torch.argmax(next_token_logits, dim=-1)
                generated_ids.append(next_token.item())
                
                # Break if we hit the end token
                if next_token.item() == model_evaluator.tokenizer.eos_token_id:
                    break
                
                # Update input for next iteration - ensure correct dimensions
                current_ids = next_token.view(1, 1)
            
            # Decode the generated sequence
            output_text = model_evaluator.tokenizer.decode(generated_ids)
            return output_text
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