from fasthtml.common import *
import torch
from transformers import GPT2LMHeadModel
import numpy as np

# Initialize FastHTML app with HTMX
app, rt = fast_app(
    pico=True,
    htmx=True,
    style=(
        Style("""
            .weight-table { 
                border-collapse: collapse;
                font-family: monospace;
            }
            .weight-table td {
                padding: 8px;
                text-align: center;
                min-width: 70px;
                height: 30px;
            }
            .controls {
                margin: 1rem 0;
                display: flex;
                gap: 1rem;
            }
            .control-group {
                flex: 1;
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

def get_color(value):
    """Convert weight value to RGB color"""
    # Normalize to [-1, 1] range
    value = max(-1, min(1, value))
    
    if value > 0:
        # Red for positive values
        return f"rgb({int(value * 255)}, 0, 0)"
    else:
        # Blue for negative values
        return f"rgb(0, 0, {int(-value * 255)})"

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

def create_weight_table(weights, max_rows=10, max_cols=10):
    """Create a colored table from weights"""
    if weights is None:
        return P("No weights available")
    
    table_cells = []
    for i in range(min(max_rows, weights.shape[0])):
        row = []
        for j in range(min(max_cols, weights.shape[1])):
            value = weights[i,j]
            # Scale value for color intensity (adjust 0.1 to control sensitivity)
            color_val = value / 0.1
            cell = Td(
                f"{value:+.3f}",
                style=f"background-color: {get_color(color_val)}; color: white;"
            )
            row.append(cell)
        table_cells.append(Tr(*row))
    
    return Table(*table_cells, cls="weight-table")

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
        return Div(
            H3(f"Layer {layer} MLP Weights"),
            P(f"Weight matrix shape: {weights.shape[0]}×{weights.shape[1]}"),
            create_stats_cards(stats),
            H4("First 10×10 weights:"),
            create_weight_table(weights)
        )
    else:
        sections = []
        for name in ['query', 'key', 'value']:
            sections.extend([
                H3(f"Layer {layer} {name.title()} Weights"),
                P(f"Weight matrix shape: {weights[name].shape[0]}×{weights[name].shape[1]}"),
                create_stats_cards(stats[name]),
                H4("First 10×10 weights:"),
                create_weight_table(weights[name])
            ])
        return Div(*sections)

@rt("/")
def get():
    """Main page with controls"""
    if model is None:
        return Titled("Weight Display Test", 
                     P("Error: Model could not be loaded"))
    
    return Titled(
        "Weight Display Test",
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


serve()