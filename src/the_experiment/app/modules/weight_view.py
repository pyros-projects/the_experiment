from fasthtml.common import *
from fasthtml.components import Sl_card, Sl_select,Sl_split_panel, Sl_tab_group, Sl_tab, Sl_tab_panel, Sl_radio_group, Sl_radio_button, Sl_option
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
        "showscale": True,  # Ensure colorbar is always shown
    }]

    layout = {
        "title": None,
        "margin": {"l": 40, "r": 20, "t": 20, "b": 30},
        "width": None,  # Allow dynamic width
        "height": None, # Allow dynamic height
        "autosize": True, # Enable autosizing
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
                "thickness": 15,
                "len": 0.9,
                "tickformat": ".3f"  # Format colorbar numbers
            }
        },
    }

    return json.dumps({"data": data, "layout": layout})

def create_stats_panel(stats, shape):
    """Create a compact stats panel"""
    return Div(cls="grid grid-cols-1 gap-2 mb-2 text-sm")(
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
    return Sl_card(cls="w-full h-full")(  # Changed to full width/height
        Div(Strong("Weight Visualization"), slot="header"),
        Div(cls="grid grid-cols-2 gap-4 p-4")(
            # Layer selector
            Div(
                Sl_select(cls="text-primary bg-background w-[200px] h-auto border",name="layer",
                    hx_get="/weights",
                    hx_target="#weight-display",
                    hx_trigger="sl-change",
                    hx_include="[name='component']",
                    placeholder="Select Layer",)(
                    *[Sl_option(f"Layer {i}", value=str(i)) 
                      for i in range(len(model.transformer.h))],
                    
                ),
            ),
            # Component selector
            Div(
                Sl_select(cls="text-primary bg-background w-[200px] h-auto border")(
                    Sl_option("MLP", value="mlp", selected=True),
                    Sl_option("Attention", value="attention"),
                    name="component",
                    hx_get="/weights",
                    hx_target="#weight-display",
                    hx_trigger="sl-change",
                    hx_include="[name='layer']",
                    placeholder="Select Component",
                ),
            ),
        ),
        Div(id="weight-display", cls="mt-2 w-full h-[calc(100vh-200px)]"),  # Set explicit height
        Script("""
            // Initial load
            htmx.ajax('GET', '/weights?layer=0&component=mlp', {
                target: '#weight-display'
            });
            
            // Handle resize
            function resizePlots() {
                const plots = document.getElementsByClassName('js-plotly-plot');
                for(let plot of plots) {
                    Plotly.Plots.resize(plot);
                }
            }
            
            // Debounce function to prevent too many resize calls
            function debounce(func, wait) {
                let timeout;
                return function() {
                    clearTimeout(timeout);
                    timeout = setTimeout(() => func(), wait);
                };
            }
            
            window.addEventListener('resize', debounce(resizePlots, 250));
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
        
        
    @rt("/attention-view")
    def get(weight_group: str, layer: int = 0):
        weights, stats = get_layer_weights(model, layer, "attention")
        return create_attention_view(weight_group, weights[weight_group], stats[weight_group])

    def create_attention_view(name, weights, stats):
        plot_data = create_heatmap(weights, f"{name.title()} Weights")
        return Div(cls="space-y-2 w-full h-full")(
            create_stats_panel(stats, weights.shape),
            P(f"Matrix: {weights.shape[0]}×{weights.shape[1]}", 
            cls="text-xs text-slate-500 mb-2"),
            Div(id=f"heatmap-{name}", cls="w-full h-[70vh] bg-white rounded shadow-sm"),
            Script(f"""
                Plotly.newPlot('heatmap-{name}', {plot_data}, 
                    undefined, {{
                        responsive: true,
                        useResizeHandler: true
                    }});
            """)
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
            
            return Div(cls="space-y-2 w-full h-full")(
                create_stats_panel(stats, weights.shape),
                P(f"Matrix: {weights.shape[0]}×{weights.shape[1]}", 
                cls="text-xs text-slate-500 mb-2"),
                Div(id="heatmap", cls="w-full h-[70vh] bg-white rounded shadow-sm"),
                Script(f"""
                    Plotly.newPlot('heatmap', {plot_data}, undefined, {{
                        responsive: true,
                        useResizeHandler: true
                    }});
                """)
            )
        else:
            return Div(
                # Radio group for selection
                Sl_radio_group(
                    id="attention-tabs",
                    name="weight_group",
                    cls="attention-tabs mb-4",
                    hx_get="/attention-view",
                    hx_target="#attention-content",
                    hx_trigger="sl-change",
                    hx_include="[name='layer']"
                )(
                    Sl_radio_button("Query", value="query", checked=True),
                    Sl_radio_button("Key", value="key"),
                    Sl_radio_button("Value", value="value"),
                ),
                # Content area
                Div(id="attention-content", cls="w-full")(
                    # Initial view (Query)
                    create_attention_view("query", weights["query"], stats["query"])
                )
            )

    return  Sl_split_panel(Div(weight_view(model), slot="start"), Div(slot="end")(Div()))