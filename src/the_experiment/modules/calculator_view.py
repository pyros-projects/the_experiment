from dataclasses import dataclass
from fasthtml.common import *

from the_experiment.components.calculator_components import InputGrid, OutputGrid, ModelOutputGrid,RnnOutputGrid

RULES="""new_A = A XOR C
new_B = NOT D  
new_C = B AND E
new_D = A OR new_B
"""


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
        Div(cls="grid grid-cols-2 gap-0 w-[1000px]")(Div(
            H2("Input State", cls="text-xl font-bold mb-4"),
            InputGrid(state.A, state.B, state.C, state.D, state.E),

            Div(f"Input Sum: {old_sum}", cls="mt-4 font-bold"),
            cls="space-y-4 justify-items-center"
        ),
        ModelOutputGrid(state.A, state.B, state.C, state.D, state.E),
        # Rules explanation
        
        Div(
            H2("Transformation Rules", cls="text-xl font-bold mb-4"),
            Pre(RULES, cls="bg-gray-100 p-4 rounded-lg  w-[70%]"),
            H2("Output State", cls="text-xl font-bold mb-4"),
            OutputGrid(new_A,new_B,new_C,new_D),    
            Div(f"Output Sum: {new_sum}", cls="mt-4 font-bold"),
            cls="mt-4 space-y-4 justify-items-center"
        ),
        RnnOutputGrid(state.A, state.B, state.C, state.D, state.E),
        # Output state
    
        # Model output
        ),
        
        
        id="main-content",
        cls="max-w-2xl"
    )
    return app


def CalculatorView(rt):
    @rt("/")
    def get():
        return Titled("Boolean Logic Transformer", 
                    render_state(state))

    @rt("/toggle/{var}")
    def post(var: str):
        # Toggle the state variable
        current = getattr(state, var)
        setattr(state, var, not current)
        return render_state(state)

    
    return render_state(BooleanState())


state = BooleanState()