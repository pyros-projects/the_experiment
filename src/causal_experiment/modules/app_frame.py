from fasthtml.common import *
from monsterui.all import *
from fasthtml.svg import *

from causal_experiment.modules.dataset_view import DatasetView
from causal_experiment.modules.test_view import TestView
from causal_experiment.modules.train_view import TrainView


def app_frame(rt):
    sidebar_group1 = (('database', 'Dataset'), ('gauge', 'Train'), ('microscope', 'Test'))

    
    

    def SideBarItem(icon, title): 
        return Li(A(DivLAligned(
            Span(UkIcon(icon)),Span(title)),cls='space-x-2'))
            
    sidebar = Container(NavContainer(
        #NavHeaderLi(H3("Email")),
        *[SideBarItem(i, t) for i, t in sidebar_group1],
        cls='space-y-6'))
    
    return Div(cls='flex divide-x divide-border mt-3')(
        sidebar,
        Grid(DatasetView(rt),TrainView(),TestView(),
             cols=1, gap=0, cls='flex-1 divide-x divide-border'))