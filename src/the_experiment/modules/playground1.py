from fasthtml.common import *
import torch
from transformers import GPT2LMHeadModel
import numpy as np

# Initialize basic FastHTML app
app, rt = fast_app()

# Load model
MODEL_PATH = "./out/tiny-gpt2-causal/final"
try:
    model = GPT2LMHeadModel.from_pretrained(MODEL_PATH)
    model.eval()
except Exception as e:
    model = None
    print(f"Error loading model: {e}")

def get_layer_weights(layer_idx=0):
    """Get weights from a specific layer"""
    try:
        # Get MLP weights as they're simpler to start with
        weights = model.transformer.h[layer_idx].mlp.c_fc.weight.detach().numpy()
        
        # Get basic stats
        stats = {
            'mean': np.mean(weights),
            'std': np.std(weights),
            'min': np.min(weights),
            'max': np.max(weights)
        }
        
        return weights, stats
    except Exception as e:
        print(f"Error getting weights: {e}")
        return None, None

@rt("/")
def get():
    """Display weights from first layer MLP"""
    if model is None:
        return Titled("Weight Display Test", 
                     P("Error: Model could not be loaded"))
    
    # Get weights from first layer
    weights, stats = get_layer_weights(0)
    if weights is None:
        return Titled("Weight Display Test", 
                     P("Error: Could not extract weights"))
    
    # Create a simple table of the first few weights (10x10)
    table_cells = []
    for i in range(min(10, weights.shape[0])):
        row = []
        for j in range(min(10, weights.shape[1])):
            # Format weight to 3 decimal places
            val = f"{weights[i,j]:.3f}"
            row.append(Td(val))
        table_cells.append(Tr(*row))
    
    return Titled(
        "Weight Display Test",
        Div(
            # Layer info
            H2("Layer 0 MLP Weights"),
            P(f"Weight matrix shape: {weights.shape}"),
            
            # Statistics
            H3("Statistics:"),
            Ul(
                Li(f"Mean: {stats['mean']:.3f}"),
                Li(f"Std Dev: {stats['std']:.3f}"),
                Li(f"Min: {stats['min']:.3f}"),
                Li(f"Max: {stats['max']:.3f}")
            ),
            
            # Weight preview
            H3("First 10x10 weights:"),
            Table(*table_cells)
        )
    )


serve()