from fasthtml.common import *
from fasthtml.components import (
    Sl_card,
    Sl_tab,
    Sl_tab_group,
    Sl_tab_panel,
)

from the_experiment.app.modules.weight_view import WeightView
from the_experiment.models.model_eval import MODEL_EVALUATOR


def AnalyzerView(rt):
    try:
        model = MODEL_EVALUATOR.model
        model.eval()
    except Exception as e:
        model = None
        print(f"Error loading model: {e}")

    if model is None:
        return Sl_card(
            Div(Strong("Error"), slot="header"),
            P("Please ensure a model is trained and selected.", cls="p-4 text-red-500"),
        )
    else:
        return (
            Sl_tab_group()(
                # Tab headers
                Sl_tab("Sankeynator", slot="nav", panel="sankey"),
                Sl_tab("Weight Watcher", slot="nav", panel="weight"),
                # Tab panels
                Sl_tab_panel(WeightView(rt), name="sankey"),
                Sl_tab_panel(WeightView(rt), name="weight"),
            ),
        )
