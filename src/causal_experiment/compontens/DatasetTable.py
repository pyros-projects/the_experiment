import json
import random
from fasthtml.common import *
from monsterui.all import *
from fasthtml.svg import *
from devtools import debug
from causal_experiment.dataset import generate_dataset


def _create_tbl_data(d):
    split_prompt = d['prompt'].split(",")
    d["a_old"] = int(split_prompt[0])
    d["b_old"] = int(split_prompt[1])
    d["c_old"] = int(split_prompt[2])
    d["d_old"] = int(split_prompt[3])
    d["e_old"] = int(split_prompt[4])
    split_completion = d['completion'].split("\n")
    split_completion_bool = split_completion[1].split(",")
    split_completion_numbers = split_completion[2].split("-")
    
    d["a_new"] = int(split_completion_bool[0])
    d["b_new"] = int(split_completion_bool[1])
    d["c_new"] = int(split_completion_bool[2])
    d["d_new"] = int(split_completion_bool[3])
    return { 'A': d["a_old"],'B': d["b_old"],'C': d["c_old"],'D': d["d_old"], 'E': d["e_old"],
            'New A': d["a_new"],'New B': d["b_new"],'New C': d["c_new"],'New D': d["d_new"],}
    
def footer():
    hw_cls = 'h-4 w-4'
    return DivFullySpaced(cls='px-2 py-2 ml-8 w-[500px]')(
        Div('1 of 100 row(s) selected.', cls='flex-1 text-sm text-muted-foreground'),
        Div(cls='flex flex-none items-center space-x-8')(
            DivCentered('Page 1 of 10', cls='w-[100px] text-sm font-medium'),
            DivLAligned(
                UkIconLink(icon='chevrons-left', button=True),
                UkIconLink(icon='chevron-left', button=True),
                UkIconLink(icon='chevron-right', button=True),
                UkIconLink(icon='chevrons-right', button=True))))
    
def header_render(col):
    cls = 'p-2 uk-table-shrink'
    match col:
        case _: return Th(col, cls=cls)
        
def cell_render(col, val):
    def _Td(*args,cls='', **kwargs): return Td(*args, cls=f'p-2 {cls}',**kwargs)
    match col:
        case "A": return _Td(val,cls='uk-table-shrink')  
        case "B": return _Td(val,cls='uk-table-shrink')  
        case "C": return _Td(val,cls='uk-table-shrink')  
        case "D": return _Td(val,cls='uk-table-shrink')  
        case "E": return _Td(val,cls='uk-table-shrink')
        case "New A": return _Td(val,cls='uk-table-shrink')
        case "New B": return _Td(val,cls='uk-table-shrink')
        case "New C": return _Td(val,cls='uk-table-shrink')
        case "New D": return _Td(val,cls='uk-table-shrink')       
        case _: raise ValueError(f"Unknown column: {col}")

def DatasetTable(data):
    data = [_create_tbl_data(d) for d in data[:100]]

    task_columns = ["A","B","C","D","E",
                    "New A","New B","New C","New D"]
    tasks_table = Div(cls='uk-overflow-auto ml-8 rounded-md border border-border h-[500px] w-[500px]',id="table-dataset")(
    TableFromDicts(
        header_data=task_columns,
        body_data=data,
        cls=(TableT.middle, TableT.divider, TableT.hover, TableT.small, ),
        body_cell_render=cell_render,
        header_cell_render=header_render,
    sortable=True))
    
    table_controls = Input(cls='w-[250px]', placeholder='Filter task')

    
    return Div(
                DivFullySpaced(cls='m-8')(
                    Div(cls='flex flex-1 gap-4')(table_controls)),
                tasks_table,footer()
                ),