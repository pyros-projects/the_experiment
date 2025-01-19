import random
import argparse
from causal_experiment.dataset import generate_dataset
from causal_experiment.train_small_causal_model import training
from devtools import debug
from fasthtml.common import *

from fasthtml.common import *

# Define the app and route
hdrs = ( Script(src="https://cdn.tailwindcss.com"),Script(src="https://cdn.jsdelivr.net/npm/alpinejs@3.x.x/dist/cdn.min.js"))


# Define the app and route
app, rt = fast_app(hdrs=hdrs, default_hdrs=False, pico=False, live=True)

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

def boolean_circle(value: bool, name: str, onclick: str = None):
    color = "bg-blue-500" if value else "bg-gray-200"
    text_color = "text-white" if value else "text-gray-800"
    circle = Div(
        str(int(value)),
        cls=f" cursor-pointer hover:opacity-80 rounded-full w-16 h-16 flex items-center justify-center text-xl {color} {text_color} transition-colors duration-200",
        hx_post=onclick,
        hx_target="#main-content"
    )

    return circle

def render_state(state: BooleanState):
    new_A, new_B, new_C, new_D, old_sum, new_sum = calculate_new_state(state)
    
    app = Main(
        # Three column layout
        Div(
            # Input State Column
            Div(
                H2("Input State", cls="text-2xl font-bold mb-6 text-center"),
                Div(
                    Div(
                        Div("A", cls="text-lg font-bold mb-2 text-center"),
                        boolean_circle(state.A, "A", "/toggle/A"),
                        cls="space-y-2"
                    ),
                    Div(
                        Div("B", cls="text-lg font-bold mb-2 text-center"),
                        boolean_circle(state.B, "B", "/toggle/B"),
                        cls="space-y-2"
                    ),
                    Div(
                        Div("C", cls="text-lg font-bold mb-2 text-center"),
                        boolean_circle(state.C, "C", "/toggle/C"),
                        cls="space-y-2"
                    ),
                    Div(
                        Div("D", cls="text-lg font-bold mb-2 text-center"),
                        boolean_circle(state.D, "D", "/toggle/D"),
                        cls="space-y-2"
                    ),
                    Div(
                        Div("E", cls="text-lg font-bold mb-2 text-center"),
                        boolean_circle(state.E, "E", "/toggle/E"),
                        cls="space-y-2"
                    ),
                    cls="flex justify-between items-start mb-4"
                ),
                Div(f"Input Sum: {old_sum}", cls="text-center font-bold mt-4"),
                cls="px-8"
            ),

            # Rules Column
            Div(
                H2("Transformation Rules", cls="text-2xl font-bold mb-6 text-center"),
                Pre("""new_A = A XOR C
new_B = NOT D
new_C = B AND E
new_D = A OR new_B""", 
                    cls="bg-gray-100 p-6 rounded-lg text-lg"
                ),
                cls="px-8"
            ),

            # Output State Column
            Div(
                H2("Output State", cls="text-2xl font-bold mb-6 text-center"),
                Div(
                    Div(
                        Div("new_A", cls="text-lg font-bold mb-2 text-center"),
                        boolean_circle(new_A, "new_A"),
                        cls="space-y-2"
                    ),
                    Div(
                        Div("new_B", cls="text-lg font-bold mb-2 text-center"),
                        boolean_circle(new_B, "new_B"),
                        cls="space-y-2"
                    ),
                    Div(
                        Div("new_C", cls="text-lg font-bold mb-2 text-center"),
                        boolean_circle(new_C, "new_C"),
                        cls="space-y-2"
                    ),
                    Div(
                        Div("new_D", cls="text-lg font-bold mb-2 text-center"),
                        boolean_circle(new_D, "new_D"),
                        cls="space-y-2"
                    ),
                    cls="flex justify-between items-start mb-4"
                ),
                Div(f"Output Sum: {new_sum}", cls="text-center font-bold mt-4"),
                cls="px-8"
            ),
            cls="grid grid-cols-3 gap-8 max-w-7xl mx-auto"
        ),
        id="main-content",
        cls="p-8"
    )
    return app


state = BooleanState()

@rt("/")
def get():
    return Titled("Boolean Logic Transformer", 
                  render_state(state))

@rt("/toggle/{var}")
def post(var: str):
    # Toggle the state variable
    current = getattr(state, var)
    print(f"Current value of {var}: {current}")
    setattr(state, var, not current)
    return render_state(state)







def main() -> None:
    parser = argparse.ArgumentParser(description='Sequence handling script')
    
    # Create mutually exclusive group for commands
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--test', type=str, help='Test with a single sequence (format: 0,1,0,1,0)')
    group.add_argument('--train', action='store_true', help='Train the model')
    group.add_argument('--generate', action='store_true', help='Generate sequences')
    group.add_argument('--calculator', action='store_true')
    
    # Optional output parameter for generate
    parser.add_argument('-o', '--omit', type=str, 
                       help='Omit sequences (format: seq1;seq2.... where seq = 0,1,0,1,0)')

    args = parser.parse_args()
    
    if args.test:
        prompt = args.test
        result = call_test(prompt)
        print(f"Test result: {result}")
        
        
    elif args.train:
        call_training()
        
    elif args.calculator:
        

        

        serve()
        
    elif args.generate:
        if args.omit:
            to_omit_list = parse_sequences(args.omit)
            print(f"Generating without input sequences: {to_omit_list}")
            call_generate_dataset(to_omit_list)
        else:
            print("Generating complete dataset")
            call_generate_dataset()


def parse_sequences(seq_str):
    """Parse multiple sequences separated by semicolon"""
    if not seq_str:
        return None
    return seq_str.split(";")
 

def call_generate_dataset(to_omit_list=None) -> None:
    random.seed(42)
    generate_dataset(20000, "dataset/train.jsonl",to_omit_list)
    generate_dataset(2000, "dataset/valid.jsonl",to_omit_list)
    generate_dataset(2000, "dataset/test.jsonl",to_omit_list)
    
def call_training() -> None:
    training()
    
    
def call_test(prompt_text: str) -> dict:
    import torch
    from transformers import GPT2LMHeadModel, GPT2TokenizerFast

    from causal_experiment.dataset import manual_test
    manual_res = manual_test(prompt_text)
    debug(manual_res)
    
    model_path = "./out/tiny-gpt2-causal/final"
    tokenizer = GPT2TokenizerFast.from_pretrained(model_path)
    model = GPT2LMHeadModel.from_pretrained(model_path)
    model.eval()



    input_ids = tokenizer.encode(prompt_text, return_tensors="pt")
    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_length=64,
            num_return_sequences=1,
            do_sample=False
        )
        
    output = tokenizer.decode(output[0])
    debug(output)
    return str(output)

