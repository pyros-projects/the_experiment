from dataclasses import dataclass
from typing import List, Dict
from vllm import LLM, SamplingParams
import math
import json

@dataclass
class LogProb:
    token_id: str
    decoded_token: str
    probability: float
    children: List["LogProb"] = None

    def __post_init__(self):
        if self.children is None:
            self.children = []

def convert_to_probability(logprob: float) -> float:
    return math.exp(logprob)

def get_top_n_continuations(model: LLM, prompt: str, n: int, sampling_params: SamplingParams) -> List[LogProb]:
    outputs = model.generate([prompt], sampling_params)
    first_output = outputs[0].outputs[0]
    
    top_logprobs = sorted(
        first_output.logprobs[0].items(), 
        key=lambda x: x[1].logprob, 
        reverse=True
    )[:n]
    
    return [
        LogProb(
            token_id=str(token_id),
            decoded_token=str(logprob.decoded_token),
            probability=convert_to_probability(logprob.logprob),
            children=[]
        )
        for token_id, logprob in top_logprobs
    ]

def build_sankey(model: str, prompt: str, n_logprobs: int, temperature: float, top_k: int, depth: int) -> Dict:
    """
    Build a JSON structure for Sankey diagram from token probabilities.
    """
    llm = LLM(model=model)
    sampling_params = SamplingParams(
        temperature=temperature,
        top_k=top_k,
        logprobs=n_logprobs
    )
    
    def build_tree(current_prompt: str, current_depth: int) -> List[LogProb]:
        if current_depth >= depth:
            return []
            
        continuations = get_top_n_continuations(llm, current_prompt, n_logprobs, sampling_params)
        
        for node in continuations:
            new_prompt = current_prompt + node.decoded_token
            node.children = build_tree(new_prompt, current_depth + 1)
            
        return continuations

    # Build the tree
    tree = build_tree(prompt, 0)
    
    # Convert tree to Sankey format
    nodes = []
    links = []
    
    def process_node(node: LogProb, level: int, parent_idx: int = None):
        current_idx = len(nodes)
        
        # Add node
        nodes.append({
            "id": f"{level}_{node.decoded_token}_{current_idx}",
            "name": node.decoded_token,
            "probability": node.probability,
            "level": level
        })
        
        # Add link from parent
        if parent_idx is not None:
            links.append({
                "source": parent_idx,
                "target": current_idx,
                "value": node.probability
            })
        
        # Process children
        for child in node.children:
            process_node(child, level + 1, current_idx)
    
    # Process all root nodes
    for root_node in tree:
        process_node(root_node, 0)
    
    return {
        "nodes": nodes,
        "links": links,
        "metadata": {
            "prompt": prompt,
            "depth": depth,
            "n_logprobs": n_logprobs
        }
    }

if __name__ == "__main__":
    # Example usage
    sankey_data = build_sankey(
        model="facebook/opt-125m",
        prompt="I should ride a bycicle because ",
        n_logprobs=3,
        temperature=1.0,
        top_k=50,
        depth=5
    )
    
    # Save to JSON file
    with open('data/token_probabilities.json', 'w') as f:
        json.dump(sankey_data, f, indent=2)