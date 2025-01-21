from fasthtml.common import *
from fasthtml.components import (
    Sl_button,
    Sl_card,
    Sl_menu,
    Sl_menu_item,
    Sl_split_panel,
)
from monsterui.all import *
from dataclasses import dataclass

from the_experiment.app.components.calculator_components import InputGrid
from the_experiment.app.modules.rules_playground import BooleanState, calculate_new_state
from the_experiment.app.components.dataset_list import create_dataset_table

@dataclass
class Prompt:
    prompt = BooleanState()

@dataclass
class RemovalList:
    list_of_removed_prompts: list[Prompt]
    

def render_menu():
    return Sl_menu(
        Sl_menu_item("Remove sequences", value="remove"),
        Sl_menu_item("Generate dataset", value="generate"),
        Sl_menu_item("Stats", value="stats"),
        style="max-width: 200px;",
    )


def render_horizontal_components(components):
    return (
        Ul(*[DivHStacked(Li(component)) for component in components], cls="space-y-4"),
    )


def remove_sequences(state):
    return Sl_card(
        Div(Strong("Remove sequences"), slot="header"),
        Div(cls=" grid grid-cols-1 w-[500px]")(
            Div(
                InputGrid(
                    state.A,
                    state.B,
                    state.C,
                    state.D,
                    state.E,
                    "/dataset_toggle",
                    "#dataset-main",
                ),
                cls="justify-items-center",
            ),
            Sl_button("Add to removed sequences", cls="mt-4"),
        ),
    )


def split_panel_1(state):
    return Sl_split_panel(position="30")(
        Div(render_menu(), slot="start"), Div(slot="end")(remove_sequences(state))
    )


def render_state(state: BooleanState,dataset_list_view):
    new_A, new_B, new_C, new_D, old_sum, new_sum = calculate_new_state(state)
    return Grid(id="dataset-main")(
        Sl_split_panel(position="50")(
            Div(split_panel_1(state), slot="start"), Div(slot="end")(dataset_list_view)
        )
    )


def DatasetView(rt):
    
    dataset_list_view = create_dataset_table(rt)
    
    @rt("/dataset_toggle/{var}")
    def post(var: str):
        # Toggle the state variable
        current = getattr(dataset_state, var)
        setattr(dataset_state, var, not current)
        return render_state(dataset_state)

    return render_state(BooleanState(),dataset_list_view)
