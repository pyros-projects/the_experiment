from fasthtml.common import *
from fasthtml.components import Sl_card, Sl_select,Sl_split_panel, Sl_tab_group, Sl_tab, Sl_tab_panel, Sl_radio_group, Sl_radio_button, Sl_option
import numpy as np
import json
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
            P("Please ensure a model is trained and selected.", cls="p-4 text-red-500")
        )
    else:
        return Sankeynator()