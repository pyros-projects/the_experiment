from fasthtml.common import *
from fasthtml.components import Sl_card, Sl_select,Sl_split_panel
import numpy as np
import json

from the_experiment.models.model_eval import MODEL_EVALUATOR

def get_layer_weights(model, layer_idx=0, component="mlp"):
    """Get weights from a specific layer and component"""
    try:
        if component == "mlp":
            weights = model.transformer.h[layer_idx].mlp.c_fc.weight.detach().numpy()
        else:
            qkv_weight = model.transformer.h[layer_idx].attn.c_attn.weight.detach().numpy()
            split_size = qkv_weight.shape[1] // 3
            weights = {
                "query": qkv_weight[:, :split_size],
                "key": qkv_weight[:, split_size : 2 * split_size],
                "value": qkv_weight[:, 2 * split_size :],
            }

        if isinstance(weights, dict):
            stats = {
                name: {
                    "mean": float(np.mean(w)),
                    "std": float(np.std(w)),
                    "min": float(np.min(w)),
                    "max": float(np.max(w)),
                }
                for name, w in weights.items()
            }
        else:
            stats = {
                "mean": float(np.mean(weights)),
                "std": float(np.std(weights)),
                "min": float(np.min(weights)),
                "max": float(np.max(weights)),
            }

        return weights, stats
    except Exception as e:
        print(f"Error getting weights: {e}")
        return None, None
    

def create_heatmap(weights, title):
    """Create a Plotly heatmap figure"""
    weights_list = weights.astype(float).tolist()
    max_abs = float(max(abs(weights.min()), abs(weights.max())))

    data = [{
        "type": "heatmap",
        "z": weights_list,
        "colorscale": "RdBu_r",
        "zmin": -max_abs,
        "zmax": max_abs,
    }]

    layout = {
        "title": None,  # Remove title - we'll use card header
        "margin": {"l": 40, "r": 20, "t": 20, "b": 30},  # Tighter margins
        "width": None,  
        "height": 350,  # Slightly shorter
        "xaxis": {
            "title": None,
            "side": "bottom",
            "showgrid": False,
        },
        "yaxis": {
            "title": None,
            "autorange": "reversed",
            "showgrid": False,
        },
        "coloraxis": {
            "colorbar": {
                "title": None,
                "thickness": 15,  # Thinner colorbar
                "len": 0.9,
            }
        },
    }

    return json.dumps({"data": data, "layout": layout})

def create_stats_panel(stats, shape):
    """Create a compact stats panel"""
    return Div(cls="grid grid-cols-4 gap-2 mb-2 text-sm")(
        Div(cls="px-2 py-1 bg-slate-50 rounded")(
            Div("Mean", cls="text-slate-500"),
            Strong(f"{stats['mean']:.3f}"),
        ),
        Div(cls="px-2 py-1 bg-slate-50 rounded")(
            Div("Std Dev", cls="text-slate-500"),
            Strong(f"{stats['std']:.3f}"),
        ),
        Div(cls="px-2 py-1 bg-slate-50 rounded")(
            Div("Min", cls="text-slate-500"),
            Strong(f"{stats['min']:.3f}"),
        ),
        Div(cls="px-2 py-1 bg-slate-50 rounded")(
            Div("Max", cls="text-slate-500"),
            Strong(f"{stats['max']:.3f}"),
        ),
    )

def weight_view(model):
    return Sl_card(cls="w-[100%]")(
        Div(Strong("Weight Visualization"), slot="header"),
        Div(cls="grid grid-cols-2 gap-4 p-4")(
            # Layer selector
            Div(
                Sl_select(
                    *[Option(f"Layer {i}", value=str(i)) 
                      for i in range(len(model.transformer.h))],
                    name="layer",
                    hx_get="/weights",
                    hx_target="#weight-display",
                    hx_trigger="change",
                    hx_include="[name='component']",
                    placeholder="Select Layer",
                ),
            ),
            # Component selector
            Div(
                Sl_select(
                    Option("MLP", value="mlp", selected=True),
                    Option("Attention", value="attention"),
                    name="component",
                    hx_get="/weights",
                    hx_target="#weight-display",
                    hx_trigger="change",
                    hx_include="[name='layer']",
                    placeholder="Select Component",
                ),
            ),
        ),
        # Weight display area
        Div(id="weight-display", cls="mt-2 w-[100%] h-[100%]"),
        Script("""
            htmx.ajax('GET', '/weights?layer=0&component=mlp', {
                target: '#weight-display'
            });
            
            window.addEventListener('resize', function() {
                Plotly.Plots.resize(document.getElementsByClassName('js-plotly-plot'));
            });
        """),
    )

def WeightView(rt):
    try:
        model = MODEL_EVALUATOR.model
        model.eval()
    except Exception as e:
        model = None
        print(f"Error loading model: {e}")

    if model is None:
        return Sl_card(
            Div(Strong("Error"), slot="header"),
            P("Please ensure a model is trained and selected.", cls="p-4 text-red-500")
        )

    @rt("/weights")
    def get(layer: int = 0, component: str = "mlp"):
        weights, stats = get_layer_weights(model, layer, component)
        
        if weights is None:
            return Div(cls="p-4 text-red-500")(
                "Could not load weights for the selected configuration."
            )

        if component == "mlp":
            plot_data = create_heatmap(weights, f"Layer {layer} MLP Weights")
            
            def render_heatmap():
                return Div(cls="space-y-2 w-[100%] h-[100%]")(
                create_stats_panel(stats, weights.shape),
                P(f"Matrix: {weights.shape[0]}×{weights.shape[1]}", 
                  cls="text-xs text-slate-500 mb-2"),
                Div(id="heatmap", cls="w-full bg-white rounded shadow-sm"),
                Script(f"Plotly.newPlot('heatmap', {plot_data}, undefined, {{responsive: true}});"))
            
            return render_heatmap()
        else:
            sections = []
            for name in ["query", "key", "value"]:
                plot_data = create_heatmap(weights[name], 
                                         f"Layer {layer} {name.title()} Weights")
                
                sections.append(Div(cls="space-y-2 mb-6")(
                    Strong(f"{name.title()} Weights", cls="text-sm"),
                    create_stats_panel(stats[name], weights[name].shape),
                    P(f"Matrix: {weights[name].shape[0]}×{weights[name].shape[1]}", 
                      cls="text-xs text-slate-500 mb-2"),
                    Div(id=f"heatmap-{name}", cls="w-full bg-white rounded shadow-sm"),
                    Script(f"""
                        Plotly.newPlot('heatmap-{name}', {plot_data}, 
                            undefined, {{responsive: true}});
                    """)
                ))

            return Div(cls="space-y-4")(*sections)

    return  Sl_split_panel(Div(weight_view(model), slot="start"), Div(slot="end")(Div()))