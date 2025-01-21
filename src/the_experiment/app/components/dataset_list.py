from fasthtml.common import *
from monsterui.all import *
from typing import List
import json

from the_experiment.app.modules.rules_playground import BooleanState


    

    
def load_datasets(page=1, per_page=10, search=None):
    # Simulated dataset loading with pagination and search
    with open("./dataset/train.jsonl", "r") as f:
        data = [json.loads(line) for line in f]
    
    # Apply search if provided
    if search:
        data = [d for d in data if search in d["prompt"] or search in d["completion"]]
    
    # Calculate pagination
    total = len(data)
    start_idx = (page - 1) * per_page
    end_idx = start_idx + per_page
    
    return {
        "data": data[start_idx:end_idx],
        "total": total,
        "pages": (total + per_page - 1) // per_page
    }
    
def split_data(value, delimiter=","):
    """Split a string into columns based on delimiter"""
    return [v.strip() for v in value.split(delimiter)]

def header_render(col, delimiter=None, subcolumns=None):
    # Default simple header when no delimiter
    if not delimiter:
        return Th(col, cls="p-2")
    
    # Only create sub-headers if delimiter is specified
    return Th(
        Table(cls="w-full")(
            Tr(Th(col, cls="text-center", colspan=len(subcolumns))),
            Tr(*[Th(f"Col {i+1}", cls="text-center px-2 text-xs") 
                for i in range(len(subcolumns))])
        ),
        cls="p-2"
    )

def cell_render(col, val, delimiter=None):
    # Default simple cell when no delimiter
    if not delimiter:
        return Td(cls="p-2")(
            Pre(val, cls="whitespace-pre-wrap font-mono text-sm")
        )
    
    # Only split into sub-columns if delimiter is specified
    values = split_data(val, delimiter)
    return Td(cls="p-2")(
        Table(cls="w-full")(
            Tr(*[Td(v, cls="text-center px-2 font-mono text-sm") for v in values])
        )
    )
    
def create_dataset_table(rt):
    @rt("/datasets")
    def get(page: int = 1, per_page: int = 10, search: str = None, 
           prompt_delimiter: str = "", completion_delimiter: str = ""):
        result = load_datasets(page, per_page, search)
        
        # Only calculate subcolumns if delimiters are set
        prompt_cols = None
        completion_cols = None
        if prompt_delimiter:
            prompt_cols = len(split_data(result["data"][0]["prompt"], prompt_delimiter))
        if completion_delimiter:
            completion_cols = len(split_data(result["data"][0]["completion"].strip(), completion_delimiter))

        table_controls = DivFullySpaced(cls="mt-8")(
            Div(cls="flex flex-1 gap-4")(
                Input(
                    cls="w-[250px]", 
                    placeholder="Search datasets",
                    name="search",
                    value=search,
                    hx_get="/datasets",
                    hx_trigger="keyup changed delay:500ms",
                    hx_target="#dataset-content",
                    hx_include="[name='prompt_delimiter'],[name='completion_delimiter'],[name='per_page']"
                ),
                Input(
                    cls="w-[80px]",
                    placeholder="Prompt split",
                    name="prompt_delimiter",
                    value=prompt_delimiter,
                    hx_get="/datasets",
                    hx_trigger="keyup changed delay:300ms",
                    hx_target="#dataset-content",
                    hx_include="[name='search'],[name='completion_delimiter'],[name='per_page']"
                ),
                Input(
                    cls="w-[80px]",
                    placeholder="Completion split", 
                    name="completion_delimiter",
                    value=completion_delimiter,
                    hx_get="/datasets",
                    hx_trigger="keyup changed delay:300ms",
                    hx_target="#dataset-content",
                    hx_include="[name='search'],[name='prompt_delimiter'],[name='per_page']"
                ),
                Select(
                    *[Option(str(n)) for n in [10, 20, 50, 100]],
                    name="per_page",
                    value=str(per_page),
                    hx_get="/datasets",
                    hx_trigger="change",
                    hx_target="#dataset-content",
                    hx_include="[name='search'],[name='prompt_delimiter'],[name='completion_delimiter']",
                    cls="w-[100px]"
                )
            )
        )

        dataset_table = Div(cls="overflow-auto mt-4 rounded-md border border-border")(
            TableFromDicts(
                header_data=["Prompt", "Completion"],
                body_data=[{
                    "Prompt": d["prompt"], 
                    "Completion": d["completion"].strip()
                } for d in result["data"]],
                body_cell_render=lambda col, val: cell_render(
                    col, val, 
                    delimiter=(prompt_delimiter if col == "Prompt" else completion_delimiter)
                ),
                header_cell_render=lambda col: header_render(
                    col,
                    delimiter=(prompt_delimiter if col == "Prompt" else completion_delimiter),
                    subcolumns=[f"Col {i+1}" for i in range(
                        prompt_cols if col == "Prompt" else completion_cols
                    )] if (prompt_delimiter if col == "Prompt" else completion_delimiter) else None
                ),
                sortable=False
            )
        )




        footer = DivFullySpaced(cls="mt-4 px-2 py-2")(
            Div(f"Showing {(page-1)*per_page + 1} to {min(page*per_page, result['total'])} of {result['total']} entries", 
                cls="flex-1 text-sm text-muted-foreground"),
            Div(cls="flex flex-none items-center space-x-8")(
                DivCentered(f"Page {page} of {result['pages']}", 
                           cls="w-[100px] text-sm font-medium"),
                DivLAligned(
                    Button("<<", 
                          hx_get=f"/datasets?page=1&per_page={per_page}", 
                          hx_target="#dataset-content",
                          disabled=page==1),
                    Button("<", 
                          hx_get=f"/datasets?page={page-1}&per_page={per_page}", 
                          hx_target="#dataset-content",
                          disabled=page==1),
                    Button(">", 
                          hx_get=f"/datasets?page={page+1}&per_page={per_page}", 
                          hx_target="#dataset-content",
                          disabled=page==result['pages']),
                    Button(">>", 
                          hx_get=f"/datasets?page={result['pages']}&per_page={per_page}", 
                          hx_target="#dataset-content",
                          disabled=page==result['pages'])
                )
            )
        )

        return Div(id="dataset-content")(
            table_controls,
            dataset_table,
            footer
        )

    return Div(cls="p-8")(
        H2("Dataset Explorer"),
        P("Browse and search through the training datasets", 
          cls=TextFont.muted_sm),
        Div(id="dataset-content", 
            hx_get="/datasets",
            hx_trigger="load")
    )