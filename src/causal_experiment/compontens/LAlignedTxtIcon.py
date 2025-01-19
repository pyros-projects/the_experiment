from fasthtml.common import *
from monsterui.all import *
from fasthtml.svg import *

def LAlignedTxtIcon(txt, icon, width=None, height=None, stroke_width=None, cls='space-x-2', icon_right=True, txt_cls=None):
        components = (
            txt if isinstance(txt, FT) else NavP(txt, cls=txt_cls or TextFont.muted_sm),
            UkIcon(icon=icon, height=height, width=width, stroke_width=stroke_width)
        )
        if not icon_right:
            components = reversed(components)
        return DivLAligned(*components, cls=cls)