import random
import argparse
from fasthtml.common import *
from fasthtml.components import Sl_tab_group,Sl_tab,Sl_tab_panel,Sl_select,Sl_option, Sl_icon
from devtools import debug
from monsterui.all import *

from the_experiment.app.modules.shoelace_app import app as shoelace_app
from the_experiment.app.modules.calculator_view import CalculatorView
from the_experiment.app.modules.dataset_view import DatasetView
from the_experiment.app.modules.heatmap import WeightHeatmap
from the_experiment.app.modules.train_view import TrainView
from the_experiment.models.cnn.train_cnn import training_cnn
from the_experiment.models.cnn2.train_cnn2 import training_cnn2
from the_experiment.models.dataset import generate_dataset
from the_experiment.models.gpt2.train_small_causal_model import training
from the_experiment.models.mann.train_mann import training_mann
from the_experiment.models.model_eval import MODEL_EVALUATOR
from the_experiment.models.rnn.train_rnn import training_rnn
from the_experiment.rules.rules import prompt_to_completion

# from the_experiment.comparison.train_mann import training_mann
# from the_experiment.models.dataset import generate_dataset
# from the_experiment.comparison.model_eval import ModelEvaluator
# from the_experiment.modules.calculator_view import CalculatorView
# from the_experiment.modules.dataset_view import DatasetView
# from the_experiment.modules.heatmap import WeightHeatmap
# from the_experiment.modules.train_view import TrainView
# from the_experiment.rules.rules import prompt_to_completion
# from the_experiment.comparison.model_eval import MODEL_EVALUATOR
# from the_experiment.models.gpt2.train_small_causal_model import training
# from the_experiment.comparison.train_rnn import training_rnn
# from the_experiment.comparison.train_cnn import training_cnn
# from the_experiment.comparison.train_cnn2 import training_cnn2
# from the_experiment.modules.shoelace_app import app as shoelace_app
# from the_experiment.components.dataset_list import tasks_homepage
# from the_experiment.comparison.load_model import output_folders

app, rt = shoelace_app



@rt("/")
def get():
    folders = MODEL_EVALUATOR.folder_contents
    pre_selection = ""
    if len(folders) > 0:
        pre_selection = "option-1"
        folder = folders[0].folder
        debug(f"Loaded folder: {folder}")
    #return CalculatorView(rt)
    return (Title("The Experiment"),Body(
                DivHStacked(cls="text-background bg-primary")(
                    Div(
                        H1("The Experiment",cls="m-auto text-background bg-primary pl-3"),
                        P('LLM experiment toolkit v0.3.5',cls="ml-auto text-background bg-primary pl-3")
                    ),
                    P("Training run selection:",cls="ml-8"),
                    Sl_select(cls="ml-4 w-[200px]",size="small",placeholder="Select model folder",value=f"{pre_selection}")(
                        *[Sl_option(
                            Sl_icon(slot="suffix",src="icon/llm") if folder.has_llm else Div(),
                            Sl_icon(slot="suffix",src="icon/rnn") if folder.has_rnn else Div(),
                            Sl_icon(slot="suffix",src="icon/cnn") if folder.has_cnn else Div(),
                            Strong(f"{folder.folder}"), value=f"option-{i+1}") for i,folder in enumerate(folders)]

                )),
            Sl_tab_group()(
                Sl_tab('Test', slot='nav', panel='test'),
                Sl_tab('Dataset', slot='nav', panel='dataset'),
                Sl_tab('Train', slot='nav', panel='train'),
                
                Sl_tab('Heatmaps', slot='nav', panel='heatmaps'),
                Sl_tab_panel(CalculatorView(rt), name='test'),
                Sl_tab_panel(DatasetView(rt), name='dataset'),
                Sl_tab_panel(TrainView(rt), name='train'),
                Sl_tab_panel(WeightHeatmap(rt), name='heatmaps'),
            ),
        ))
    
@rt("/icon/{icon}")
def get(icon:str):
    with open(f"src/the_experiment/app/assets/{icon}.svg") as f:
        return f.read()


serve()
def main() -> None:
    parser = argparse.ArgumentParser(description='Sequence handling script')
    
    # Create mutually exclusive group for commands
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--test', type=str, help='Test with a single sequence (format: 0,1,0,1,0)')
    group.add_argument('--train', type=str, help='Train the model')
    group.add_argument('--train_all', type=str, help='Train all models')
    group.add_argument('--train_rnn', type=str, help='Train a rnn for comparison')
    group.add_argument('--train_cnn', type=str, help='Train a cnn for comparison')
    group.add_argument('--train_cnn2', type=str, help='Train a cnn for comparison')
    group.add_argument('--train_mann', type=str, help='Train a mann for comparison')
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
        
    elif args.train_cnn2:
        folder = args.train_cnn2
        training_cnn2(folder)
        
    elif args.train_mann:
        folder = args.train_mann
        training_mann(folder)
        
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
    

    
def call_test(folder,prompt_text: str) -> dict:
    # calculate by algorithm
    manual_res = prompt_to_completion(prompt_text)
    debug(manual_res)
    
    model_eval = MODEL_EVALUATOR
    # calculate by gpt2
    output = model_eval.eval_model(prompt_text)
    debug(output)

    # calculate by rnn
    output_rnn = model_eval.eval_rnn(prompt_text)
    debug(output_rnn)

    # calculate by cnn
    output_cnn = model_eval.eval_cnn(prompt_text)
    debug(output_cnn)
    return str(output)


