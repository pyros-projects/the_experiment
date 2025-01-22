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
)

app = fast_app(
    hdrs=hdrs, pico=False, default_hdrs=True, live=True, title="The Experiment"
)
