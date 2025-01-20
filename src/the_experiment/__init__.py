import random
import argparse
from fasthtml.common import *
from fasthtml.components import Sl_tab_group,Sl_tab,Sl_tab_panel
from devtools import debug
from monsterui.all import *

from the_experiment.dataset import generate_dataset, manual_test
from the_experiment.comparison.model_eval import eval_model, eval_rnn, eval_cnn
from the_experiment.modules.calculator_view import CalculatorView
from the_experiment.modules.dataset_view import DatasetView
from the_experiment.modules.playground3 import WeightHeatmap
from the_experiment.modules.train_view import TrainView
from the_experiment.train_small_causal_model import training
from the_experiment.comparison.train_rnn import training_rnn
from the_experiment.comparison.train_cnn import training_cnn
from the_experiment.modules.shoelace_app import app as shoelace_app
from the_experiment.modules.frankenui02 import tasks_homepage

app, rt = shoelace_app


@rt("/")
def get():
    #return CalculatorView(rt)
    return Body(
            H1("The Experiment v0.3",cls="mb-4 text-2xl font-bold text-gray-800"),
            Sl_tab_group()(
                Sl_tab('Test', slot='nav', panel='test'),
                Sl_tab('Dataset', slot='nav', panel='dataset'),
                Sl_tab('Train', slot='nav', panel='train'),
                Sl_tab('list', slot='nav', panel='list'),
                Sl_tab('Microscope', slot='nav', panel='microscope'),
                Sl_tab_panel(CalculatorView(rt), name='test'),
                Sl_tab_panel(DatasetView(rt), name='dataset'),
                Sl_tab_panel(TrainView(rt), name='train'),
                Sl_tab_panel(tasks_homepage, name='list'),
                Sl_tab_panel(WeightHeatmap(rt), name='microscope'),
            ),
        )


def main() -> None:
    parser = argparse.ArgumentParser(description='Sequence handling script')
    
    # Create mutually exclusive group for commands
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--test', type=str, help='Test with a single sequence (format: 0,1,0,1,0)')
    group.add_argument('--train', type=str, help='Train the model')
    group.add_argument('--train_all', type=str, help='Train all models')
    group.add_argument('--train_rnn', type=str, help='Train a rnn for comparison')
    group.add_argument('--train_cnn', type=str, help='Train a cnn for comparison')
    group.add_argument('--generate', action='store_true', help='Generate sequences')
    group.add_argument('--testui', action='store_true')
    group.add_argument('--app', action='store_true')
    
    # Optional output parameter for generate
    parser.add_argument('-o', '--omit', type=str, 
                       help='Omit sequences (format: seq1;seq2.... where seq = 0,1,0,1,0)')

    parser.add_argument('-f', '--folder', type=str, 
                       help='folder for model loading')

    args = parser.parse_args()
    
    if args.test:
        if args.folder:
            prompt = args.test
            result = call_test(prompt)
            print(f"Test result: {result}")
        else:
            print("No folder specified")

    if args.app:
        import webview
        webview.create_window("Flask App", "http://127.0.0.1:5001")
        webview.start()


    elif args.train:
        folder = args.train
        training(folder)

    elif args.train_all:
        folder = args.train_all
        training(folder) 
        training_rnn(folder)
        training_cnn(folder)

    elif args.train_rnn:
        folder = args.train_rnn
        training_rnn(folder)

    elif args.train_cnn:
        folder = args.train_cnn
        training_cnn(folder)
        
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


