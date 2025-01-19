from fasthtml.common import *
from monsterui.all import *
from fasthtml.svg import *

def LAlignedIconTxt(txt, icon, width=None, height=None, stroke_width=None, cls='space-x-2', txt_cls=None):
        return LAlignedTxtIcon(txt=txt, icon=icon, width=width, stroke_width=stroke_width, cls=cls, icon_right=False, txt_cls=txt_cls)