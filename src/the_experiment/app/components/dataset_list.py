"""FrankenUI Tasks Example"""

from fasthtml.common import *
from monsterui.all import *
from fasthtml.svg import *
import json


def NavP(*c, cls=TextFont.muted_sm):
    return P(cls=cls)(*c)


def LAlignedTxtIcon(
    txt,
    icon,
    width=None,
    height=None,
    stroke_width=None,
    cls="space-x-2",
    icon_right=True,
    txt_cls=None,
):
    c = (
        txt
        if isinstance(txt, FT)
        else NavP(txt, cls=ifnone(txt_cls, TextFont.muted_sm)),
        UkIcon(icon=icon, height=height, width=width, stroke_width=stroke_width),
    )
    if not icon_right:
        c = reversed(c)
    return DivLAligned(*c, cls=cls)


def LAlignedIconTxt(
    txt, icon, width=None, height=None, stroke_width=None, cls="space-x-2", txt_cls=None
):
    # Good for navbars
    return LAlignedTxtIcon(
        txt=txt,
        icon=icon,
        width=width,
        stroke_width=stroke_width,
        cls=cls,
        icon_right=False,
        txt_cls=txt_cls,
    )


def SpacedPP(left, right=None):
    return DivFullySpaced(NavP(left), NavP(right) if right else "")


def SpacedPPs(*c):
    return [SpacedPP(*tuplify(o)) for o in c]


with open("data/status_list.json", "r") as f:
    data = json.load(f)
with open("data/statuses.json", "r") as f:
    statuses = json.load(f)


def _create_tbl_data(d):
    return {
        "Done": d["selected"],
        "Task": d["id"],
        "Title": d["title"],
        "Status": d["status"],
        "Priority": d["priority"],
    }


data = [_create_tbl_data(d) for d in data]

priority_dd = [
    {"priority": "low", "count": 36},
    {"priority": "medium", "count": 33},
    {"priority": "high", "count": 31},
]

rows_per_page_dd = [10, 20, 30, 40, 50]

status_dd = [
    {"status": "backlog", "count": 21},
    {"status": "todo", "count": 21},
    {"status": "progress", "count": 20},
    {"status": "done", "count": 19},
    {"status": "cancelled", "count": 19},
]


def create_hotkey_li(hotkey):
    return NavCloseLi(
        A(cls="justify-between")(hotkey[0], Span(hotkey[1], cls=TextFont.muted_sm))
    )


hotkeys_a = (
    ("Profile", "⇧⌘P"),
    ("Billing", "⌘B"),
    ("Settings", "⌘S"),
    ("New Team", ""),
)
hotkeys_b = (("Logout", ""),)

avatar_opts = DropDownNavContainer(
    NavHeaderLi(P("sveltecult"), NavSubtitle("leader@sveltecult.com")),
    NavDividerLi(),
    *map(create_hotkey_li, hotkeys_a),
    NavDividerLi(),
    *map(create_hotkey_li, hotkeys_b),
)


def CreateTaskModal():
    return Modal(
        Div(cls="p-6")(
            ModalTitle("Create Task"),
            P(
                "Fill out the information below to create a new task",
                cls=TextFont.muted_sm,
            ),
            Br(),
            Form(cls="space-y-6")(
                Grid(
                    Div(
                        Select(
                            *map(Option, ("Documentation", "Bug", "Feature")),
                            label="Task Type",
                            id="task_type",
                        )
                    ),
                    Div(
                        Select(
                            *map(
                                Option,
                                ("In Progress", "Backlog", "Todo", "Cancelled", "Done"),
                            ),
                            label="Status",
                            id="task_status",
                        )
                    ),
                    Div(
                        Select(
                            *map(Option, ("Low", "Medium", "High")),
                            label="Priority",
                            id="task_priority",
                        )
                    ),
                ),
                TextArea(
                    label="Title",
                    placeholder="Please describe the task that needs to be completed",
                ),
                DivRAligned(
                    ModalCloseButton("Cancel", cls=ButtonT.ghost),
                    ModalCloseButton("Submit", cls=ButtonT.primary),
                    cls="space-x-5",
                ),
            ),
        ),
        id="TaskForm",
    )


page_heading = DivFullySpaced(cls="space-y-2")(
    Div(cls="space-y-2")(
        H2("Welcome back!"),
        P("Here's a list of your tasks for this month!", cls=TextFont.muted_sm),
    ),
    Div(DiceBearAvatar("sveltcult", 8, 8), avatar_opts),
)

table_controls = (
    Input(cls="w-[250px]", placeholder="Filter task"),
    Button("Status"),
    DropDownNavContainer(
        map(
            NavCloseLi,
            [
                A(DivFullySpaced(P(a["status"]), P(a["count"])), cls=TextT.capitalize)
                for a in status_dd
            ],
        )
    ),
    Button("Priority"),
    DropDownNavContainer(
        map(
            NavCloseLi,
            [
                A(
                    DivFullySpaced(
                        LAlignedIconTxt(a["priority"], icon="check"), a["count"]
                    ),
                    cls=TextT.capitalize,
                )
                for a in priority_dd
            ],
        )
    ),
    Button("View"),
    DropDownNavContainer(
        map(
            NavCloseLi,
            [
                A(LAlignedIconTxt(o, icon="check"))
                for o in ["Title", "Status", "Priority"]
            ],
        )
    ),
    Button(
        "Create Task",
        cls=(ButtonT.primary, TextFont.bold_sm),
        uk_toggle="target: #TaskForm",
    ),
)


def task_dropdown():
    return Div(
        Button(UkIcon("ellipsis")),
        DropDownNavContainer(
            map(
                NavCloseLi,
                [
                    A(
                        "Edit",
                    ),
                    A("Make a copy"),
                    A(
                        "Favorite",
                    ),
                    A(SpacedPP("Delete", "⌘⌫")),
                ],
            )
        ),
    )


def header_render(col):
    cls = "p-2 " + "uk-table-shrink" if col in ("Done", "Actions") else ""
    match col:
        case "Done":
            return Th(CheckboxX(), cls=cls)
        case "Actions":
            return Th("", cls=cls)
        case _:
            return Th(col, cls=cls)


def cell_render(col, val):
    def _Td(*args, cls="", **kwargs):
        return Td(*args, cls=f"p-2 {cls}", **kwargs)

    match col:
        case "Done":
            return _Td(shrink=True)(CheckboxX(selected=val))
        case "Task":
            return _Td(val)
        case "Title":
            return _Td(cls="max-w-[500px] truncate", expand=True)(
                val, cls="font-medium"
            )
        case "Status" | "Priority":
            return _Td(cls="uk-text-nowrap uk-text-capitalize")(Span(val))
        case "Actions":
            return _Td(cls="uk-table-shrink")(task_dropdown())
        case _:
            raise ValueError(f"Unknown column: {col}")


task_columns = ["Done", "Task", "Title", "Status", "Priority", "Actions"]

tasks_table = Div(cls="uk-overflow-auto mt-4 rounded-md border border-border")(
    TableFromDicts(
        header_data=task_columns,
        body_data=data,
        body_cell_render=cell_render,
        header_cell_render=header_render,
        sortable=True,
    )
)


def footer():
    hw_cls = "h-4 w-4"
    return DivFullySpaced(cls="mt-4 px-2 py-2")(
        Div("1 of 100 row(s) selected.", cls="flex-1 text-sm text-muted-foreground"),
        Div(cls="flex flex-none items-center space-x-8")(
            DivCentered("Page 1 of 10", cls="w-[100px] text-sm font-medium"),
            DivLAligned(
                UkIconLink(icon="chevrons-left", button=True),
                UkIconLink(icon="chevron-left", button=True),
                UkIconLink(icon="chevron-right", button=True),
                UkIconLink(icon="chevrons-right", button=True),
            ),
        ),
    )


tasks_ui = Div(
    DivFullySpaced(cls="mt-8")(Div(cls="flex flex-1 gap-4")(table_controls)),
    tasks_table,
    footer(),
)

tasks_homepage = Div(cls="p-8")(page_heading, tasks_ui, CreateTaskModal())
