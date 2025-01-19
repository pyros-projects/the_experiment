from fasthtml.common import *


@dataclass 
class BooleanState:
    A: bool = False
    B: bool = False
    C: bool = False
    D: bool = False
    E: bool = False


# Define the app and route
hdrs = ( Script(src="https://cdn.tailwindcss.com"),Script(src="https://cdn.jsdelivr.net/npm/alpinejs@3.x.x/dist/cdn.min.js"))
# Define the app and route
app, rt = fast_app(hdrs=hdrs, live=True)

state = BooleanState()

@rt("/")
def get():
    return Titled("Boolean Logic Transformer", 
                render_state(state,rt))



def calculate_new_state(state: BooleanState):
    # Apply structural equations:
    new_A = state.A != state.C  # XOR
    new_B = not state.D  # NOT
    new_C = state.B and state.E  # AND
    new_D = state.A or new_B  # OR with new_B
    
    old_sum = sum([state.A, state.B, state.C, state.D, state.E])
    new_sum = sum([new_A, new_B, new_C, new_D])
    
    return new_A, new_B, new_C, new_D, old_sum, new_sum

def boolean_circle(value: bool, name: str, onclick: str = None):
    color = "bg-blue-500" if value else "bg-gray-200"
    text_color = "text-white" if value else "text-gray-800"
    circle = Div(
        value and "1" or "0",
        cls=f" cursor-pointer hover:opacity-80 rounded-full w-16 h-16 flex items-center justify-center text-xl {color} {text_color} transition-colors duration-200",
        hx_post=onclick,
        hx_target="#main-content",
    )

    return circle

def set_routes(rt):
    print("Setting routes")
    @rt("/toggle/{var}")
    def post(var: str):
        # Toggle the state variable
        current = getattr(state, var)
        print(f"Current value of {var}: {current}")
        setattr(state, var, not current)
        return render_state(state)


def render_state(state: BooleanState):
    new_A, new_B, new_C, new_D, old_sum, new_sum = calculate_new_state(state)
    
    app = Main(
        # Input state
        Div(
            H2("Input State", cls="text-xl font-bold mb-4"),
            boolean_circle("A", state.A, "/toggle/A"),
            boolean_circle("B", state.B, "/toggle/B"),
            boolean_circle("C", state.C, "/toggle/C"), 
            boolean_circle("D", state.D, "/toggle/D"),
            boolean_circle("E", state.E, "/toggle/E"),
            Div(f"Input Sum: {old_sum}", cls="mt-4 font-bold"),
            cls="space-y-4 mb-8"
        ),

        # Rules explanation
        Div(
            H2("Transformation Rules", cls="text-xl font-bold mb-4"),
            Pre("""new_A = A XOR C
new_B = NOT D  
new_C = B AND E
new_D = A OR new_B""", cls="bg-gray-100 p-4 rounded-lg"),
            cls="mb-8"
        ),

        # Output state
        Div(
            H2("Output State", cls="text-xl font-bold mb-4"),
            boolean_circle("new_A", new_A),
            boolean_circle("new_B", new_B),
            boolean_circle("new_C", new_C),
            boolean_circle("new_D", new_D),
            Div(f"Output Sum: {new_sum}", cls="mt-4 font-bold"),
            cls="space-y-4"
        ),
        id="main-content",
        cls="max-w-2xl mx-auto p-6"
    )
    return app

# def render_state(state: BooleanState,rt):

#     new_A, new_B, new_C, new_D, old_sum, new_sum = calculate_new_state(state)
    
#     app = Main(
#         # Three column layout
#         Div(
#             # Input State Column
#             Div(
#                 H2("Input State", cls="text-2xl font-bold mb-6 text-center"),
#                 Div(
#                     Div(
#                         Div("A", cls="text-lg font-bold mb-2 text-center"),
#                         boolean_circle(state.A, "A", "/toggle/A"),
#                         cls="space-y-2"
#                     ),
#                     Div(
#                         Div("B", cls="text-lg font-bold mb-2 text-center"),
#                         boolean_circle(state.B, "B", "/toggle/B"),
#                         cls="space-y-2"
#                     ),
#                     Div(
#                         Div("C", cls="text-lg font-bold mb-2 text-center"),
#                         boolean_circle(state.C, "C", "/toggle/C"),
#                         cls="space-y-2"
#                     ),
#                     Div(
#                         Div("D", cls="text-lg font-bold mb-2 text-center"),
#                         boolean_circle(state.D, "D", "/toggle/D"),
#                         cls="space-y-2"
#                     ),
#                     Div(
#                         Div("E", cls="text-lg font-bold mb-2 text-center"),
#                         boolean_circle(state.E, "E", "/toggle/E"),
#                         cls="space-y-2"
#                     ),
#                     cls="flex justify-between items-start mb-4"
#                 ),
#                 Div(f"Input Sum: {old_sum}", cls="text-center font-bold mt-4"),
#                 cls="px-8"
#             ),

#             # Rules Column
#             Div(
#                 H2("Transformation Rules", cls="text-2xl font-bold mb-6 text-center"),
#                 Pre("""new_A = A XOR C
# new_B = NOT D
# new_C = B AND E
# new_D = A OR new_B""", 
#                     cls="bg-gray-100 p-6 rounded-lg text-lg"
#                 ),
#                 cls="px-8"
#             ),

#             # Output State Column
#             Div(
#                 H2("Output State", cls="text-2xl font-bold mb-6 text-center"),
#                 Div(
#                     Div(
#                         Div("new_A", cls="text-lg font-bold mb-2 text-center"),
#                         boolean_circle(new_A, "new_A"),
#                         cls="space-y-2"
#                     ),
#                     Div(
#                         Div("new_B", cls="text-lg font-bold mb-2 text-center"),
#                         boolean_circle(new_B, "new_B"),
#                         cls="space-y-2"
#                     ),
#                     Div(
#                         Div("new_C", cls="text-lg font-bold mb-2 text-center"),
#                         boolean_circle(new_C, "new_C"),
#                         cls="space-y-2"
#                     ),
#                     Div(
#                         Div("new_D", cls="text-lg font-bold mb-2 text-center"),
#                         boolean_circle(new_D, "new_D"),
#                         cls="space-y-2"
#                     ),
#                     cls="flex justify-between items-start mb-4"
#                 ),
#                 Div(f"Output Sum: {new_sum}", cls="text-center font-bold mt-4"),
#                 cls="px-8"
#             ),
#             cls="grid grid-cols-3 gap-8 max-w-7xl mx-auto"
#         ),
#         id="main-content",
#         cls="p-8"
#     )
#     return app


