from fasthtml.common import *
import torch
from transformers import GPT2LMHeadModel
import numpy as np
import plotly.express as px
import json

# Initialize FastHTML app with HTMX
app, rt = fast_app(
    pico=True,
    htmx=True,
    style=(
        Style("""
            .controls {
                margin: 1rem 0;
                display: flex;
                gap: 1rem;
            }
            .control-group {
                flex: 1;
            }
            .heatmap-container {
                width: 100%;
                height: 800px;
                margin: 20px 0;
            }
            .stats-list {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 1rem;
                margin: 1rem 0;
            }
            .stat-card {
                background: #f8f9fa;
                padding: 1rem;
                border-radius: 4px;
            }
        """)
    ),
    hdrs=(
        Script(src="https://cdn.plot.ly/plotly-2.24.1.min.js"),
        
    )
)

# Load model
MODEL_PATH = "./out/tiny-gpt2-causal/final"
try:
    model = GPT2LMHeadModel.from_pretrained(MODEL_PATH)
    model.eval()
except Exception as e:
    model = None
    print(f"Error loading model: {e}")

def get_layer_weights(layer_idx=0, component="mlp"):
    """Get weights from a specific layer and component"""
    try:
        if component == "mlp":
            weights = model.transformer.h[layer_idx].mlp.c_fc.weight.detach().numpy()
        else:
            qkv_weight = model.transformer.h[layer_idx].attn.c_attn.weight.detach().numpy()
            split_size = qkv_weight.shape[1] // 3
            weights = {
                'query': qkv_weight[:, :split_size],
                'key': qkv_weight[:, split_size:2*split_size],
                'value': qkv_weight[:, 2*split_size:]
            }
        
        if isinstance(weights, dict):
            stats = {name: {
                'mean': np.mean(w),
                'std': np.std(w),
                'min': np.min(w),
                'max': np.max(w)
            } for name, w in weights.items()}
        else:
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

def create_heatmap(weights, title):
    """Create a Plotly heatmap figure"""
    # Convert numpy array to native Python types for JSON serialization
    weights_list = weights.astype(float).tolist()
    max_abs = float(max(abs(weights.min()), abs(weights.max())))
    
    # Create the figure data directly
    data = [{
        'type': 'heatmap',
        'z': weights_list,
        'colorscale': 'RdBu_r',
        'zmin': -max_abs,
        'zmax': max_abs
    }]
    
    # Create the layout
    layout = {
        'title': {'text': title, 'x': 0.5},
        'width': 1000,
        'height': 600,
        'xaxis': {'title': 'Column', 'side': 'bottom'},
        'yaxis': {'title': 'Row', 'autorange': 'reversed'},
        'coloraxis': {'colorbar': {'title': 'Weight Value'}}
    }
    
    # Combine into final figure
    figure = {'data': data, 'layout': layout}
    
    return json.dumps(figure)

def create_stats_cards(stats):
    """Create statistics cards"""
    return Div(
        Div(
            Div(f"{stats['mean']:.3f}", cls="stat-value"),
            Div("Mean", cls="stat-label"),
            cls="stat-card"
        ),
        Div(
            Div(f"{stats['std']:.3f}", cls="stat-value"),
            Div("Standard Deviation", cls="stat-label"),
            cls="stat-card"
        ),
        Div(
            Div(f"{stats['min']:.3f}", cls="stat-value"),
            Div("Minimum", cls="stat-label"),
            cls="stat-card"
        ),
        Div(
            Div(f"{stats['max']:.3f}", cls="stat-value"),
            Div("Maximum", cls="stat-label"),
            cls="stat-card"
        ),
        cls="stats-list"
    )

@rt("/weights")
def get(layer: int = 0, component: str = "mlp"):
    """HTMX endpoint for weight updates"""
    weights, stats = get_layer_weights(layer, component)
    
    if component == "mlp":
        # Create MLP heatmap
        plot_data = create_heatmap(
            weights, 
            f"Layer {layer} MLP Weights"
        )
        
        return Div(
            H3(f"Layer {layer} MLP Weights"),
            P(f"Weight matrix shape: {weights.shape[0]}×{weights.shape[1]}"),
            create_stats_cards(stats),
            # Single container for MLP
            Div(
                Div(id="heatmap", cls="heatmap-container"),
                Script(f"Plotly.newPlot('heatmap', {plot_data});")
            )
        )
    else:
        # Create containers for each attention component
        sections = []
        for name in ['query', 'key', 'value']:
            plot_data = create_heatmap(
                weights[name],
                f"Layer {layer} {name.title()} Weights"
            )
            
            sections.extend([
                H3(f"Layer {layer} {name.title()} Weights"),
                P(f"Weight matrix shape: {weights[name].shape[0]}×{weights[name].shape[1]}"),
                create_stats_cards(stats[name]),
                Div(
                    Div(id=f"heatmap-{name}", cls="heatmap-container"),
                    Script(f"Plotly.newPlot('heatmap-{name}', {plot_data});")
                )
            ])
            
        return Div(*sections)

@rt("/")
def get():
    """Main page with controls"""
    if model is None:
        return Titled("Weight Display Test", 
                     P("Error: Model could not be loaded"))
    
    return Titled(
        "Weight Visualization",
        Div(
            # Controls
            Div(
                Div(
                    Label("Layer:"),
                    Select(
                        *[Option(f"Layer {i}", value=str(i)) 
                          for i in range(len(model.transformer.h))],
                        name="layer",
                        hx_get="/weights",
                        hx_target="#weight-display",
                        hx_trigger="change",
                        hx_include="[name='component']"
                    ),
                    cls="control-group"
                ),
                Div(
                    Label("Component:"),
                    Select(
                        Option("MLP", value="mlp", selected=True),
                        Option("Attention", value="attention"),
                        name="component",
                        hx_get="/weights",
                        hx_target="#weight-display",
                        hx_trigger="change",
                        hx_include="[name='layer']"
                    ),
                    cls="control-group"
                ),
                cls="controls"
            ),
            
            # Weight display area
            Div(id="weight-display"),
            
            # Initial load script
            Script("""
                htmx.ajax('GET', '/weights?layer=0&component=mlp', 
                         {target: '#weight-display'});
            """)
        )
    )

if __name__ == "__main__":
    serve()