from fasthtml.common import *

from the_experiment.models.model_eval import MODEL_EVALUATOR
from the_experiment.rules.rules import int2bool


def boolean_circle(value: bool, name: str, onclick: str, htmx_target="#main-content"):
    color = "bg-blue-500" if value else "bg-gray-200"
    text_color = "text-white" if value else "text-gray-800"
    return Div(
        value and "1" or "0",
        cls=f"rounded-full w-12 h-12 flex items-center justify-center cursor-pointer {color} {text_color}",
        hx_post=onclick,
        hx_target=htmx_target,
        hx_swap="outerHTML",
    )


def boolean_row(label: str, value: bool, onclick: str = None):
    return Div(
        Div(label, cls="font-bold w-24"),
        boolean_circle(value, label, onclick),
        cls="flex items-center gap-4",
    )


def InputGrid(A, B, C, D, E, htmx_path="/toggle", htmx_target="#main-content"):
    cls = "justify-items-center"
    return Div(cls="grid grid-cols-5 gap-2 w-[70%]")(
        Div(H1("A"), cls=cls),
        Div(H1("B"), cls=cls),
        Div(H1("C"), cls=cls),
        Div(H1("D"), cls=cls),
        Div(H1("E"), cls=cls),
        Div(boolean_circle(A, "A", htmx_path + "/A", htmx_target), cls=cls),
        Div(boolean_circle(B, "B", htmx_path + "/B", htmx_target), cls=cls),
        Div(boolean_circle(C, "C", htmx_path + "/C", htmx_target), cls=cls),
        Div(boolean_circle(D, "D", htmx_path + "/D", htmx_target), cls=cls),
        Div(boolean_circle(E, "E", htmx_path + "/E", htmx_target), cls=cls),
    )


def OutputGrid(newA, newB, newC, newD):
    cls = "justify-items-center"
    return Div(cls="grid grid-cols-4 gap-2 w-[60%]")(
        Div(H1("new_A"), cls=cls),
        Div(H1("new_B"), cls=cls),
        Div(H1("new_C"), cls=cls),
        Div(H1("new_D"), cls=cls),
        Div(boolean_circle(newA, "new_A", None), cls=cls),
        Div(boolean_circle(newB, "new_B", None), cls=cls),
        Div(boolean_circle(newC, "new_C", None), cls=cls),
        Div(boolean_circle(newD, "new_D", None), cls=cls),
    )


def ModelOutputGrid(A, B, C, D, E, folder=MODEL_EVALUATOR.active_folder):
    cls = "justify-items-center"
    model_output = MODEL_EVALUATOR.eval_model_bool(A, B, C, D, E)

    if model_output is None or model_output == "":
        return Div(
            H2("Model Output State", cls="text-xl font-bold mb-4"),
            Div("Error with loading model", cls="mt-4 font-bold"),
            cls="space-y-4",
        )

    split_output = model_output.split("\n")
    new_vals = split_output[1].split(",")
    newA = int2bool(new_vals[0])
    sum_in = split_output[2].split("-")

    return Div(
        Pre(f"{model_output}", cls="bg-gray-100 p-4 rounded-lg  w-[100%]"),
        Div(cls="grid grid-cols-4 gap-2 w-[60%]")(
            Div(H1("new_A"), cls=cls),
            Div(H1("new_B"), cls=cls),
            Div(H1("new_C"), cls=cls),
            Div(H1("new_D"), cls=cls),
            Div(boolean_circle(new_vals[0] == "1", "new_A", None), cls=cls),
            Div(boolean_circle(new_vals[1] == "1", "new_B", None), cls=cls),
            Div(boolean_circle(new_vals[2] == "1", "new_C", None), cls=cls),
            Div(boolean_circle(new_vals[3] == "1", "new_D", None), cls=cls),
        ),
        Div(f"Input Sum: {sum_in[0]} - Output Sum: {sum_in[1]}", cls="mt-4 font-bold"),
        cls="space-y-4 mt-0 justify-items-center  h-[275px]",
    )


def RnnOutputGrid(A, B, C, D, E):
    cls = "justify-items-center"

    model_output = MODEL_EVALUATOR.eval_rnn_bool(A, B, C, D, E)

    if model_output is None or model_output == "":
        return Div(
            H2("Model Output State", cls="text-xl font-bold mb-4"),
            Div("Error with loading model", cls="mt-4 font-bold"),
            cls="space-y-4",
        )

    split_output = model_output.split("\n")
    new_vals = split_output[1].split(",")
    newA = int2bool(new_vals[0])

    sum_in = split_output[2].split("-")

    return Div(
        Pre(f"{model_output}\n ", cls="bg-gray-100 p-4 rounded-lg  w-[100%]"),
        Div(cls="grid grid-cols-4 gap-2 w-[60%]")(
            Div(H1("new_A"), cls=cls),
            Div(H1("new_B"), cls=cls),
            Div(H1("new_C"), cls=cls),
            Div(H1("new_D"), cls=cls),
            Div(boolean_circle(new_vals[0] == "1", "new_A", None), cls=cls),
            Div(boolean_circle(new_vals[1] == "1", "new_B", None), cls=cls),
            Div(boolean_circle(new_vals[2] == "1", "new_C", None), cls=cls),
            Div(boolean_circle(new_vals[3] == "1", "new_D", None), cls=cls),
        ),
        Div(f"Input Sum: {sum_in[0]} - Output Sum: {sum_in[1]}", cls="mt-4 font-bold"),
        cls="space-y-4 mt-0 justify-items-center h-[275px]",
    )


def CnnOutputGrid(A, B, C, D, E):
    model_output = MODEL_EVALUATOR.eval_cnn_bool(A, B, C, D, E)

    if model_output is None or model_output == "":
        return Div(
            H2("Model Output State", cls="text-xl font-bold mb-4"),
            Div("Error with loading model", cls="mt-4 font-bold"),
            cls="space-y-4",
        )

    return Div(
        Pre(f"{model_output}\n ", cls="bg-gray-100 p-4 rounded-lg  w-[100%]"),
        cls="space-y-4 mt-0 justify-items-center h-[275px]",
    )
