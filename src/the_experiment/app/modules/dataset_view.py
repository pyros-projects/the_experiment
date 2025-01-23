import json
from dataclasses import dataclass

from fasthtml.common import *
from fasthtml.components import (
    Sl_button,
    Sl_card,
    Sl_checkbox,
    Sl_divider,
    Sl_input,
    Sl_radio_button,
    Sl_radio_group,
    Sl_split_panel,
    Sl_tab,
    Sl_tab_group,
    Sl_tab_panel,
)
from monsterui.all import *

from the_experiment.app.components.calculator_components import InputGrid
from the_experiment.app.components.dataset_list import create_dataset_table
from the_experiment.app.modules.rules_playground import BooleanState
from the_experiment.models.dataset import generate_dataset as call_generate_dataset
from the_experiment.utils.helper import DISPLAY_MODE, bool_str


@dataclass
class BooleanState:
    A: bool = False
    B: bool = False
    C: bool = False
    D: bool = False
    E: bool = False


omitted_states = set()
CURRENT_DATA = []


######################################################################
# HELPER: Load train dataset
######################################################################
def load_dataset(path="datasets/train.jsonl"):
    data = []
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    data.append(json.loads(line))
    except FileNotFoundError:
        pass
    return data


######################################################################
# HELPER: For each row in CURRENT_DATA, parse its prompt bits and count them up
######################################################################
def compute_pattern_counts():
    """Returns a dict: (a,b,c,d,e) -> how many lines in CURRENT_DATA have that pattern.
    Expects row["prompt"] = '1,0,1,0,1' or similar
    """
    counts = {}
    for row in CURRENT_DATA:
        bits = [bool(int(x)) for x in row["prompt"].split(",")]
        st = tuple(bits)  # e.g. (True,False,True,False,True)
        counts[st] = counts.get(st, 0) + 1
    return counts


######################################################################
# STATS PAGE
######################################################################
def stats_view():
    global DISPLAY_MODE
    # 1) Radio button form to pick display style
    form = Form(
        "Display style:",
        Sl_radio_group(name="mode", value="tf")(
            Sl_radio_button(
                "1/0",
                cls="ml-2",
                value="bits",
            ),
            Sl_radio_button(
                "True/False",
                cls="ml-2",
                value="tf",
            ),
            Sl_radio_button(
                "✅/❌",
                cls="ml-2",
                value="emoji",
            ),
        ),
        hx_post="/stats_mode",
        hx_target="#dataset-stats",
        hx_request="include:#mode",
        hx_swap="innerHTML",
    )

    # 2) Some top-level stats
    total_lines = len(CURRENT_DATA)
    omitted_count = len(omitted_states)
    # Count bits across entire dataset (just an example)
    bit_counts = [0, 0, 0, 0, 0]
    for row in CURRENT_DATA:
        bits = [int(x) for x in row["prompt"].split(",")]
        for i, bval in enumerate(bits):
            bit_counts[i] += bval
    bits_text = ", ".join(f"Bit {i} count: {bit_counts[i]}" for i in range(5))

    summary_div = Div(
        H2("Dataset Summary"),
        Div(f"Number of training lines loaded: {total_lines}"),
        Div(f"Omitted states count: {omitted_count}"),
        Div(bits_text, cls="mt-1"),
    )

    # 3) Table of all 32 possible patterns
    pattern_counts = compute_pattern_counts()
    rows = []
    for i in range(32):
        a = bool((i >> 0) & 1)
        b = bool((i >> 1) & 1)
        c = bool((i >> 2) & 1)
        d = bool((i >> 3) & 1)
        e = bool((i >> 4) & 1)
        st = (a, b, c, d, e)
        omitted = st in omitted_states
        row_label = "Omitted" if omitted else str(pattern_counts.get(st, 0))
        # Convert booleans to user-chosen display style:
        row = Tr(
            Td(bool_str(a, DISPLAY_MODE)),
            Td(bool_str(b, DISPLAY_MODE)),
            Td(bool_str(c, DISPLAY_MODE)),
            Td(bool_str(d, DISPLAY_MODE)),
            Td(bool_str(e, DISPLAY_MODE)),
            Td(row_label),
        )
        rows.append(row)

    table = Table(Thead(Tr(Th("A"), Th("B"), Th("C"), Th("D"), Th("E"), Th("Count/Omitted"))), Tbody(*rows))

    # Wrap everything in a card
    return Sl_card(cls="dataset-stats")(
        Div(Strong("Probability/States Distribution"), slot="header"), Div(form, cls="mb-3"), summary_div, table
    )


######################################################################
# REMOVAL & GENERATION
######################################################################
dataset_state = BooleanState()


def toggles_for_ABCDE(state, route, target):
    # Example using CheckboxX from fasthtml.xtend
    from fasthtml.xtend import CheckboxX

    def row(letter, val):
        return CheckboxX(
            checked=val,
            label=letter,
            id=letter,
            name=letter,
            hx_post=f"{route}/{letter}",
            hx_target=target,
            hx_swap="innerHTML",
        )

    return Div(
        row("A", state.A),
        row("B", state.B),
        row("C", state.C),
        row("D", state.D),
        row("E", state.E),
        cls="flex flex-col space-y-2",
    )


def render_removed_sequences(omitted_states):
    from devtools import debug

    debug(omitted_states)
    if not omitted_states:
        omitted_list = Div("No omitted states yet!")
    else:
        omitted_list = Div(
            *[
                Div(
                    InputGrid(o[0], o[1], o[2], o[3], o[4], size="50"),
                    Sl_button(
                        "Remove",
                        hx_post=f"/remove_omitted/{idx}",
                        hx_target="#dataset-main",
                        hx_swap="innerHTML",
                        cls="mt-2",
                    ),
                    cls="py-1",
                )
                for idx, o in enumerate(omitted_states)
            ],
            cls="mt-4",
        )
    return omitted_list


def remove_sequences_panel(state):
    return Sl_card(
        Div(Strong("Remove sequences"), slot="header"),
        Div(cls="grid grid-cols-1 w-[500px]")(
            Div(
                InputGrid(state.A, state.B, state.C, state.D, state.E, "/dataset_toggle", "#dataset-main"),
                cls="justify-items-center",
            ),
            Sl_button(
                "Add current A/B/C/D/E to Omitted",
                hx_post="/add_omitted",
                hx_target="#dataset-main",
                hx_swap="innerHTML",
                cls="mt-4",
            ),
            Sl_divider(),
            Div(
                render_removed_sequences(omitted_states),
                cls="justify-content-center",
            ),
        ),
    )


# def remove_sequences_panel(state):
#     toggles = toggles_for_ABCDE(state, "/dataset_toggle", "#dataset-main")
#     sorted_omits = sorted(omitted_states)
#     if not sorted_omits:
#         omitted_list = Div("No omitted states yet!")
#     else:
#         omitted_list = Div(
#             *[
#                 Div(
#                     f"({o[0]},{o[1]},{o[2]},{o[3]},{o[4]})",
#                     Sl_button(
#                         "Remove",
#                         hx_post=f"/remove_omitted/{idx}",
#                         hx_target="#dataset-main",
#                         hx_swap="innerHTML",
#                         cls="ml-2",
#                     ),
#                     cls="py-1",
#                 )
#                 for idx, o in enumerate(sorted_omits)
#             ],
#             cls="mt-4",
#         )

#     add_btn = Sl_button(
#         "Add current A/B/C/D/E to Omitted",
#         hx_post="/add_omitted",
#         hx_target="#dataset-main",
#         hx_swap="innerHTML",
#         cls="mt-4",
#     )
#     return Sl_card(Div(Strong("Remove Sequences"), slot="header"), Div(toggles, add_btn, omitted_list, cls="p-3"))


def generate_dataset_panel():
    return Sl_card(cls="rounded-lg shadow-lg ml-4")(
        Div(Strong("Training configurator"), slot="header"),
        Form(id="checkboxes", cls="flex grid grid-cols-2 gap-2 w-[300px] m-4")(
            # Add error div at the top - will be updated via HTMX
            Sl_input(
                value="20000",
                label="Size of dataset",
                name="folder",
                id="folder",
                cls="col-span-2",
                required=True,
            ),
            Sl_divider(cls="col-span-2"),
            Sl_checkbox("Test Set", cls="", name="train_llm"),
            Sl_checkbox("Validation Set", cls="", name="train_rnn"),
            Sl_input(
                value="2000",
                label="Size of test set",
                name="folder",
                id="folder",
                cls="col-span-1",
                required=False,
            ),
            Sl_input(
                value="2000",
                label="Size of validation set",
                name="folder",
                id="folder",
                cls="col-span-1",
                required=False,
            ),
            Div(id="form-errors", cls="col-span-2"),
            Sl_button(
                "Generate dataset",
                hx_post="/generate_dataset",
                hx_target="#dataset-main",
                hx_swap="innerHTML",
                cls="p-3 col-span-2",
            ),
        ),
    )


def render_tab_group(state):
    return Sl_tab_group()(
        Sl_tab("Remove Sequences", slot="nav", panel="remove", active=True),
        Sl_tab("Generate Dataset", slot="nav", panel="generate"),
        Sl_tab("Stats", slot="nav", panel="stats"),
        Sl_tab_panel(remove_sequences_panel(state), name="remove"),
        Sl_tab_panel(generate_dataset_panel(), name="generate"),
        Sl_tab_panel(stats_view(), name="stats"),
    )


######################################################################
# MAIN VIEW
######################################################################
def DatasetView(rt):
    @rt("/dataset_toggle/{var}")
    def post_toggle(var: str):
        if hasattr(dataset_state, var):
            old = getattr(dataset_state, var)
            setattr(dataset_state, var, not old)
        return render_main()

    @rt("/add_omitted")
    def post_add():
        st = (dataset_state.A, dataset_state.B, dataset_state.C, dataset_state.D, dataset_state.E)
        omitted_states.add(st)
        return render_main()

    @rt("/remove_omitted/{idx:int}")
    def post_remove(idx: int):
        sorted_omits = sorted(omitted_states)
        if 0 <= idx < len(sorted_omits):
            omitted_states.remove(sorted_omits[idx])
        return render_main()

    @rt("/generate_dataset")
    def post_generate_dataset(
        train_size: int = 20000,
        genTest: bool = False,
        test_size: int = 2000,
        genVal: bool = False,
        val_size: int = 2000,
    ):
        # Generate user-chosen splits
        omit_list = list(omitted_states)
        call_generate_dataset(train_size, "datasets/train.jsonl", omit_list)
        if genTest:
            call_generate_dataset(test_size, "datasets/test.jsonl", omit_list)
        if genVal:
            call_generate_dataset(val_size, "datasets/validation.jsonl", omit_list)
        # Reload train
        CURRENT_DATA.clear()
        CURRENT_DATA.extend(load_dataset("datasets/train.jsonl"))
        return render_main()

    # Let user pick display mode (tf, bits, emoji)
    @rt("/stats_mode")
    def post_stats_mode(mode: str = "tf"):
        global DISPLAY_MODE
        if mode in ("tf", "bits", "emoji"):
            DISPLAY_MODE = mode
        return stats_view()

    def render_main():
        left_tabs = render_tab_group(dataset_state)
        # Rebuild the right-hand table with CURRENT_DATA
        dataset_table = create_dataset_table(rt)
        return Div(
            Sl_split_panel(position="45")(Div(slot="start")(left_tabs), Div(slot="end")(dataset_table)),
            id="dataset-main",
        )

    # Optionally load a train dataset once on server start:
    # CURRENT_DATA[:] = load_dataset("datasets/train.jsonl")

    return render_main()
