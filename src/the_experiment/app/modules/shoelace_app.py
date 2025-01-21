from fasthtml.common import (
    fast_app,
    Div,
    P,
    serve,
    H1,
    Button,
    Body,
    A,
    Ul,
    Li,
    Script,
    Link,
    Input,
    NotStr,
    picolink,
    Style,
)
from monsterui.all import *


style = Style("""
            .controls {
                margin: 1rem 0;
                display: flex;
                gap: 1rem;
            }
            .control-group {
                flex: 1;
            }
            .heatmap-container {
                width: 100%;
                height: 800px;
                margin: 20px 0;
            }
            .stats-list {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 1rem;
                margin: 1rem 0;
            }
            .stat-card {
                background: #f8f9fa;
                padding: 1rem;
                border-radius: 4px;
            }
        """)


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
