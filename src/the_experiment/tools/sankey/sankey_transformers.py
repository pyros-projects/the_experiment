from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import json
from dataclasses import dataclass
from typing import List, Dict
import math

@dataclass
class LogProb:
    token_id: str
    decoded_token: str
    probability: float
    children: List["LogProb"] = None

    def __post_init__(self):
        if self.children is None:
            self.children = []

def get_top_n_continuations(model, tokenizer, prompt: str, n: int) -> List[Dict]:
    """Get top n most probable next tokens."""
    inputs = tokenizer(prompt, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits[:, -1, :]
        probs = torch.nn.functional.softmax(logits, dim=-1)[0]
        
        top_probs, top_indices = torch.topk(probs, n)
        
        continuations = []
        for prob, idx in zip(top_probs, top_indices):
            token = tokenizer.decode(idx)
            continuations.append({
                'token_id': str(idx.item()),
                'decoded_token': token,
                'probability': prob.item()
            })
        
    return continuations

def build_sankey(model_name: str, prompt: str, n_tokens: int, depth: int) -> Dict:
    """
    Build a JSON structure for Sankey diagram from token probabilities.
    
    Args:
        model_name: Name or path of the HuggingFace model
        prompt: Initial text prompt
        n_tokens: Number of top tokens to consider at each step
        depth: How many levels deep to generate
        
    Returns:
        Dictionary containing nodes and links for Sankey diagram
    """
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    nodes = []
    links = []
    
    def explore_continuations(current_prompt: str, level: int, parent_idx: int = None):
        if level >= depth:
            return
            
        continuations = get_top_n_continuations(model, tokenizer, current_prompt, n_tokens)
        
        for continuation in continuations:
            # Add node
            current_idx = len(nodes)
            nodes.append({
                'id': f"{level}_{continuation['decoded_token']}_{current_idx}",
                'name': continuation['decoded_token'],
                'probability': continuation['probability'],
                'level': level
            })
            
            # Add link from parent if not root
            if parent_idx is not None:
                links.append({
                    'source': parent_idx,
                    'target': current_idx,
                    'value': continuation['probability']
                })
            
            # Explore next level
            next_prompt = current_prompt + continuation['decoded_token']
            explore_continuations(next_prompt, level + 1, current_idx)
    
    # Start exploration from initial prompt
    explore_continuations(prompt, 0)
    
    return {
        'nodes': nodes,
        'links': links,
        'metadata': {
            'prompt': prompt,
            'depth': depth,
            'n_tokens': n_tokens,
            'model': model_name
        }
    }

