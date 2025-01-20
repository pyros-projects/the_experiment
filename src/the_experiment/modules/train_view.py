from dataclasses import dataclass
from fasthtml.common import *
from fasthtml.components import Sl_tab_group,Sl_tab,Sl_tab_panel,Sl_checkbox, Sl_button

def TrainView(rt):
    return Div(cls="flex grid grid-cols-1 gap-4 w-[500px] m-4")(
                Div(
                Sl_checkbox("Train LLM",cls="m-4"),
                Sl_checkbox("Train RNN",cls="m-4"),
                Sl_checkbox("Train CNN",cls="m-4")),
                Sl_button("Train")
            )