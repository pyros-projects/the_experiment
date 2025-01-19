from fasthtml.common import *
from monsterui.all import *
from fasthtml.svg import *


def TrainView():
     return Div(cls='flex flex-col',uk_filter="target: .js-filter")(
            Div(cls='flex px-4 py-2 ')(
                H3('Train')))
    