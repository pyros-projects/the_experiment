import random
import argparse
from the_experiment.dataset import generate_dataset, manual_test
from the_experiment.comparison.model_eval import eval_model, eval_rnn, eval_cnn
from the_experiment.modules.calculator_view import CalculatorView
from the_experiment.train_small_causal_model import training
from the_experiment.comparison.train_rnn import training_rnn
from the_experiment.comparison.train_cnn import training_cnn
from devtools import debug
from fasthtml.common import *

hdrs = (Script(src="https://cdn.tailwindcss.com"),
        Link(rel="stylesheet", href="https://cdn.jsdelivr.net/npm/daisyui@4.11.1/dist/full.min.css"))
app, rt = fast_app(hdrs=hdrs, live=True)

@rt("/")
def get():
    return CalculatorView(rt)


def main() -> None:
    parser = argparse.ArgumentParser(description='Sequence handling script')
    
    # Create mutually exclusive group for commands
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--test', type=str, help='Test with a single sequence (format: 0,1,0,1,0)')
    group.add_argument('--train', action='store_true', help='Train the model')
    group.add_argument('--train_rnn', action='store_true', help='Train a rnn for comparison')
    group.add_argument('--train_cnn', action='store_true', help='Train a cnn for comparison')
    group.add_argument('--generate', action='store_true', help='Generate sequences')
    group.add_argument('--testui', action='store_true')
    
    # Optional output parameter for generate
    parser.add_argument('-o', '--omit', type=str, 
                       help='Omit sequences (format: seq1;seq2.... where seq = 0,1,0,1,0)')

    args = parser.parse_args()
    
    if args.test:
        prompt = args.test
        result = call_test(prompt)
        print(f"Test result: {result}")
        
        
    elif args.train:
        training()

    elif args.train_rnn:
        training_rnn()

    elif args.train_cnn:
        training_cnn()
        
    elif args.testui:
        # start webserver
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
    

    
def call_test(prompt_text: str) -> dict:
    # calculate by algorithm
    manual_res = manual_test(prompt_text)
    debug(manual_res)
    
    # calculate by gpt2
    output = eval_model(prompt_text)
    debug(output)

    # calculate by rnn
    output_rnn = eval_rnn(prompt_text)
    debug(output_rnn)

    # calculate by cnn
    output_cnn = eval_cnn(prompt_text)
    debug(output_cnn)
    return str(output)

