from fasthtml.common import *
from monsterui.all import *

# def sl_link(*args, **kwargs): return(jsd('@shoelace-style', 'shoelace', 'cdn', *args, prov='npm', **kwargs))
# app = fast_app(debug=True,default_hdrs=True, live=True,pico=False, hdrs=(
#     Meta(charset='UTF-8'),
#     Meta(name='viewport', content='width=device-width, initial-scale=1.0'),
#     sl_link('themes/light.css', typ='css'),
#     sl_link('shoelace-autoloader.js', type='module'),
#     Script(src='https://cdn.tailwindcss.com'),
#     Script(src="https://cdn.plot.ly/plotly-2.24.1.min.js"),
#     Script(src="https://unpkg.com/htmx.org/dist/ext/sse.js"),
#     Theme.slate.headers(highlightjs=True, mode="light",),
# ))

hdrs = (
    Script(src="https://cdn.tailwindcss.com"),
    Link(
        rel="stylesheet",
        href="https://cdn.jsdelivr.net/npm/@shoelace-style/shoelace@2.19.1/cdn/themes/light.css",
    ),
    # Link(rel="stylesheet", media="(prefers-color-scheme:dark)",href="https://cdn.jsdelivr.net/npm/@shoelace-style/shoelace@2.19.1/cdn/themes/dark.css",onload="document.documentElement.classList.add('sl-theme-dark');"),
    Script(
        src="https://cdn.jsdelivr.net/npm/@shoelace-style/shoelace@2.19.1/cdn/shoelace.js",
        type="module",
    ),
    Script(src="https://cdn.plot.ly/plotly-2.24.1.min.js"),
    Theme.slate.headers(highlightjs=True, mode="light"),
    Script(src="https://cdn.plot.ly/plotly-2.24.1.min.js"),
    Script(src="https://unpkg.com/htmx.org/dist/ext/sse.js"),
    Script(src="https://cdnjs.cloudflare.com/ajax/libs/d3/7.8.5/d3.min.js"),
    Script(src="https://cdnjs.cloudflare.com/ajax/libs/d3-sankey/0.12.3/d3-sankey.min.js"),
    # Style('body {\r\n            font-family: Arial, sans-serif;\r\n            margin: 0;\r\n            padding: 20px;\r\n            background: #f5f5f5;\r\n        }\r\n        #container {\r\n            background: white;\r\n            padding: 20px;\r\n            border-radius: 8px;\r\n            box-shadow: 0 2px 4px rgba(0,0,0,0.1);\r\n        }\r\n        .node rect {\r\n            cursor: pointer;\r\n            fill-opacity: .9;\r\n            shape-rendering: crispEdges;\r\n        }\r\n        .node text {\r\n            pointer-events: none;\r\n            font-size: 12px;\r\n        }\r\n        .link {\r\n            fill: none;\r\n            stroke: #000;\r\n            stroke-opacity: .2;\r\n        }\r\n        .link:hover {\r\n            stroke-opacity: .5;\r\n        }\r\n        #fileInput {\r\n            margin-bottom: 20px;\r\n            padding: 10px;\r\n            border: 1px solid #ccc;\r\n            border-radius: 4px;\r\n        }')
)

app = fast_app(hdrs=hdrs, pico=False, default_hdrs=True, live=True, title="The Experiment")
