import json
import random
from fasthtml.common import *
from monsterui.all import *
from fasthtml.svg import *
from devtools import debug
from causal_experiment.compontens.DatasetTable import DatasetTable
from causal_experiment.dataset import generate_dataset
from causal_experiment.visualizer.dataset_visualizer import create_visualizations


train_set = ["A,B,C,D,E"]
def DatasetView(rt):
    
    
        
        
    @rt('/dataset/generate')
    def post():
        global train_set
        print("incrementing")
        random.seed(42)
        train_set = generate_dataset(20000, "dataset/train.jsonl",None)
        valid_set = generate_dataset(2000, "dataset/valid.jsonl",None)
        test_set = generate_dataset(2000, "dataset/test.jsonl",None)
        create_visualizations(train_set)
        return DatasetTable(train_set)
    

    
    inp = Input(id="omit-input", name="omit", placeholder="Enter sequence to omit")
    add = Form(
        Group(inp, Button("Generate")),
        hx_post="/dataset/generate",
        target_id='dataset-list',
        hx_swap="afterbegin",
        cls='flex gap-4 m-8'
    )

    

    return Div(cls='flex flex-col', uk_filter="target: .js-filter")(
        Div(cls='flex px-4 py-2')(
            H3('Dataset')
        ),
        add,
        Div(id="dataset-list")
    )