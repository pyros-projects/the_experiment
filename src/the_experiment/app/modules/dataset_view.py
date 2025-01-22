from fasthtml.common import *
from fasthtml.components import (
    Sl_button,
    Sl_card,
    Sl_menu,
    Sl_menu_item,
    Sl_split_panel,
    Sl_tab_group,
    Sl_tab,
    Sl_tab_panel,
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

dataset_state = BooleanState()    

def render_menu(state):
    return  Sl_tab_group()(
                Sl_tab("Remove Sequences", slot="nav", panel="remove", active=True),
                Sl_tab("Generate Dataset", slot="nav", panel="generate"),
                Sl_tab("Stats", slot="nav", panel="stats"),
            
                Sl_tab_panel(
                    remove_sequences(state),name="remove"
                ),
                Sl_tab_panel(
                    remove_sequences(state),name="generate"
                ),
                Sl_tab_panel(
                    remove_sequences(state),name="stats"
                )
        ),


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




def render_state(state: BooleanState,dataset_list_view):
    new_A, new_B, new_C, new_D, old_sum, new_sum = calculate_new_state(state)
    return Div(Sl_split_panel(position="50",)(
            Div(slot="start")(render_menu(state)), 
            Div(slot="end")(dataset_list_view)
        ),id="dataset-main")


def DatasetView(rt):
    
    dataset_list_view = create_dataset_table(rt)
    
    @rt("/dataset_toggle/{var}")
    def post(var: str):
        # Toggle the state variable
        current = getattr(dataset_state, var)
        setattr(dataset_state, var, not current)
        return render_state(dataset_state,dataset_list_view)

    return render_state(BooleanState(),dataset_list_view)
