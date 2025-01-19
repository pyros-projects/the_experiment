from fasthtml.common import *

# Define the app and route
app, rt = fast_app()

@rt("/")
def home():
    # The input form for A, B, C, D, and E
    form = Form(
        Group(
            CheckboxX(id="A", label="A"),
            CheckboxX(id="B", label="B"),
            CheckboxX(id="C", label="C"),
            CheckboxX(id="D", label="D"),
            CheckboxX(id="E", label="E"),
        ),
        Button("Submit", type="submit", cls="primary"),
        hx_post="/calculate", target_id="results"
    )
    return Titled("Structural Equations Demo", form, Div(id="results"))

@rt("/calculate")
def calculate(A: bool = False, B: bool = False, C: bool = False, D: bool = False, E: bool = False):
    # Parse initial values
    A_b, B_b, C_b, D_b, E_b = bool(A), bool(B), bool(C), bool(D), bool(E)
    
    # Calculate new values using structural equations
    new_A = A_b ^ C_b
    new_B = not D_b
    new_C = B_b and E_b
    new_D = A_b or new_B

    # Calculate old and new sums
    old_sum = int(A_b) + int(B_b) + int(C_b) + int(D_b) + int(E_b)
    new_sum = int(new_A) + int(new_B) + int(new_C) + int(new_D)

    # Render results
    results = Div(
        H2("Results"),
        Ul(
            Li(f"Initial A: {A_b}, B: {B_b}, C: {C_b}, D: {D_b}, E: {E_b}"),
            Li(f"New A (A XOR C): {new_A}"),
            Li(f"New B (NOT D): {new_B}"),
            Li(f"New C (B AND E): {new_C}"),
            Li(f"New D (A OR New B): {new_D}"),
            Li(f"Old Sum: {old_sum}"),
            Li(f"New Sum: {new_sum}")
        ),
        cls="results"
    )
    return results

serve()
