from dataclasses import dataclass
from fasthtml.common import *


hdrs = (Script(src="https://cdn.tailwindcss.com"),Link(rel="stylesheet", href="https://cdn.jsdelivr.net/npm/daisyui@4.11.1/dist/full.min.css"))
app, rt = fast_app(hdrs=hdrs, live=True)

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


def InputGrid():
    cls = "justify-items-center"
    return Div(cls="grid grid-cols-5 gap-2 w-[50%]")(
           Div(H1("A"),cls=cls),
           Div(H1("B"),cls=cls),
           Div(H1("C"),cls=cls),
           Div(H1("D"),cls=cls),
           Div(H1("E"),cls=cls),
           Div(boolean_circle(state.A, "A", "/toggle/A"),cls=cls),
           Div(boolean_circle(state.B, "B", "/toggle/B"),cls=cls),
           Div(boolean_circle(state.C, "C", "/toggle/C"),cls=cls),
           Div(boolean_circle(state.D, "D", "/toggle/D"),cls=cls),
           Div(boolean_circle(state.E, "E", "/toggle/E"),cls=cls),
    )

def OutputGrid(newA,newB,newC,newD):
    cls = "justify-items-center"
    return Div(cls="grid grid-cols-4 gap-2 w-[50%]")(
           Div(H1("new_A"),cls=cls),
           Div(H1("new_B"),cls=cls),
           Div(H1("new_C"),cls=cls),
           Div(H1("new_D"),cls=cls),
           Div(boolean_circle(newA, "new_A",None),cls=cls),
           Div(boolean_circle(newB, "new_B",None),cls=cls),
           Div(boolean_circle(newC, "new_C",None),cls=cls),
           Div(boolean_circle(newD, "new_D",None),cls=cls),
    )
    
        


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

def boolean_circle(value: bool, name: str, onclick: str):
    color = "bg-blue-500" if value else "bg-gray-200"
    text_color = "text-white" if value else "text-gray-800"
    return Div(
        value and "1" or "0",
        cls=f"rounded-full w-12 h-12 flex items-center justify-center cursor-pointer {color} {text_color}",
        hx_post=onclick,
        hx_target="#main-content"
    )

def boolean_row(label: str, value: bool, onclick: str = None):
    return Div(
        Div(label, cls="font-bold w-24"),
        boolean_circle(value, label, onclick),
        cls="flex items-center gap-4"
    )

def render_state(state: BooleanState):
    new_A, new_B, new_C, new_D, old_sum, new_sum = calculate_new_state(state)
    
    app = Main(
        # Input state
        
        Div(
            H2("Input State", cls="text-xl font-bold mb-4"),
            InputGrid(),

            Div(f"Input Sum: {old_sum}", cls="mt-4 font-bold"),
            cls="space-y-4 mb-8"
        ),

        # Rules explanation
        Div(
            H2("Transformation Rules", cls="text-xl font-bold mb-4"),
            Pre("""new_A = A XOR C
new_B = NOT D  
new_C = B AND E
new_D = A OR new_B""", 
                cls="bg-gray-100 p-4 rounded-lg  w-[50%]"),
            cls="mb-8"
        ),

        # Output state
        Div(
            H2("Output State", cls="text-xl font-bold mb-4"),
            OutputGrid(new_A,new_B,new_C,new_D),    
            Div(f"Output Sum: {new_sum}", cls="mt-4 font-bold"),
            cls="space-y-4"
        ),
        id="main-content",
        cls="max-w-2xl"
    )
    return app


state = BooleanState()



# Add Tailwind CSS
# style = """
# @import url('https://cdn.jsdelivr.net/npm/tailwindcss@2.2.19/dist/tailwind.min.css');
# """

#app.hdrs += (Style(style),)

serve()