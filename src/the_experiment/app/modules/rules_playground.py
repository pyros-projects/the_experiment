from dataclasses import dataclass
from fasthtml.common import *
from monsterui.all import *
from fasthtml.components import Sl_card
from the_experiment.app.components.calculator_components import (
    CnnOutputGrid,
    InputGrid,
    OutputGrid,
    ModelOutputGrid,
    RnnOutputGrid,
)
from the_experiment.rules.rules import RULES
from the_experiment.models.model_eval import MODEL_EVALUATOR


@dataclass
class BooleanState:
    A: bool = False
    B: bool = False
    C: bool = False
    D: bool = False
    E: bool = False


def calculate_new_state(state: BooleanState):
    # Apply structural equations:
    new_A = state.A != state.C  # XOR
    new_B = not state.D  # NOT
    new_C = state.B and state.E  # AND
    new_D = state.A or new_B  # OR with new_B

    old_sum = sum([state.A, state.B, state.C, state.D, state.E])
    new_sum = sum([new_A, new_B, new_C, new_D])

    return new_A, new_B, new_C, new_D, old_sum, new_sum


def render_state(state: BooleanState):
    new_A, new_B, new_C, new_D, old_sum, new_sum = calculate_new_state(state)

    app = Main(
        # Input state
        H1("Rules Playground", cls="text-3xl font-bold mt-0"),
        Div(cls="grid grid-cols-3 gap-4 w-[1200px] ml-4 mt-4")(
            Sl_card(cls="rounded-lg shadow-lg")(
                Div(Strong("Input Sequence"), slot="header"),
                Div(
                    # H2("Input State", cls="text-xl font-bold mb-4"),
                    InputGrid(state.A, state.B, state.C, state.D, state.E),
                    Div(f"Input Sum: {old_sum}", cls="mt-4 font-bold"),
                    cls="space-y-4 justify-items-center h-[100%]",
                ),
            ),
            Sl_card(cls="rounded-lg shadow-lg w-[400px]")(
                Div(
                    Strong(f"Model Inference // LLM ({MODEL_EVALUATOR.active_folder})"),
                    slot="header",
                ),
                ModelOutputGrid(state.A, state.B, state.C, state.D, state.E),
            ),
            Sl_card(cls="rounded-lg shadow-lg w-[400px]")(
                Div(
                    Strong(f"Model Inference // RNN ({MODEL_EVALUATOR.active_folder})"),
                    slot="header",
                ),
                RnnOutputGrid(state.A, state.B, state.C, state.D, state.E),
            ),
            Sl_card(cls="rounded-lg shadow-lg w-[400px]")(
                Div(Strong("Fact // Transformation Rules"), slot="header"),
                Div(
                    Pre(RULES, cls="bg-gray-100 p-4 rounded-lg  w-[100%]"),
                    OutputGrid(new_A, new_B, new_C, new_D),
                    Div(f"Output Sum: {new_sum}", cls="mt-4 font-bold"),
                    cls="mt-0 space-y-4 justify-items-center h-[275px]",
                ),
            ),
            Sl_card(cls="rounded-lg shadow-lg w-[400px]")(
                Div(
                    Strong(f"Model Inference // CNN ({MODEL_EVALUATOR.active_folder})"),
                    slot="header",
                ),
                CnnOutputGrid(state.A, state.B, state.C, state.D, state.E),
            ),
            Div(),
        ),
        id="main-content",
        cls="max-w-2xl",
    )
    return app


def RulesPlaygroundView(rt):
    @rt("/")
    def get():
        return render_state(state)

    @rt("/toggle/{var}")
    def post(var: str):
        # Toggle the state variable
        current = getattr(state, var)
        setattr(state, var, not current)
        return render_state(state)

    return render_state(BooleanState())


state = BooleanState()
