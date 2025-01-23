import json

import torch
from fasthtml.common import *
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "gpt2"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
current_prompt = ""

app, rt = fast_app()


def get_top_logits(text, k=10):
    inputs = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits[0, -1, :]
        probs = torch.softmax(logits, dim=-1)

    top_k = torch.topk(probs, k)
    tokens = [tokenizer.decode([i]) for i in top_k.indices]
    probabilities = top_k.values.tolist()
    return list(zip(tokens, probabilities))


@rt("/sankey/select")
def post(token: str):
    global current_prompt
    current_prompt += token
    logits = get_top_logits(current_prompt)

    max_width = 300
    table = Table(
        Tr(Th("Token"), Th("Probability"), Th("Bar")),
        *[
            Tr(
                Td(t),
                Td(f"{p:.2%}"),
                Td(Div(style=f"background: #007bff; width: {p * max_width}px; height: 20px;")),
                hx_post=f"/sankey/select",
                hx_vals=json.dumps({"token": t}),
                hx_target="#viz",
                style="cursor: pointer;",
            )
            for t, p in logits
        ],
        style="width: 100%; border-collapse: collapse;",
    )

    return Div(P(f"Current prompt: {current_prompt}"), table)


@rt("/")
def get():
    return Titled(
        "LLM Logit Probabilities",
        Form(
            Input(type="text", id="prompt", name="prompt", placeholder="Enter prompt"),
            Input(type="number", id="top_k", name="top_k", value="10", min="1", max="100"),
            Button("Generate", type="submit"),
            hx_post="/sankey/generate",
            hx_target="#viz",
        ),
        Div(id="viz", style="width: 100%; overflow-x: auto;"),
    )


@rt("/sankey/generate")
def post(prompt: str, top_k: int):
    global current_prompt
    current_prompt = prompt
    logits = get_top_logits(prompt, top_k)

    # Create a simple bar chart visualization
    max_width = 300
    table = Table(
        Tr(Th("Token"), Th("Probability"), Th("Bar")),
        *[
            Tr(
                Td(token),
                Td(f"{prob:.2%}"),
                Td(Div(style=f"background: #007bff; width: {prob * max_width}px; height: 20px;")),
                hx_post=f"/sankey/select",
                hx_vals=json.dumps({"token": token}),
                hx_target="#viz",
                style="cursor: pointer;",
            )
            for token, prob in logits
        ],
        style="width: 100%; border-collapse: collapse;",
    )

    return Div(P(f"Current prompt: {prompt}"), table)


def Tr(*args, **kwargs):
    return ft("tr", *args, **kwargs)


def Th(*args, **kwargs):
    return ft("th", *args, style="padding: 8px; text-align: left; border-bottom: 2px solid #ddd;", **kwargs)


def Td(*args, **kwargs):
    return ft("td", *args, style="padding: 8px; border-bottom: 1px solid #ddd;", **kwargs)


def Table(*args, **kwargs):
    return ft("table", *args, **kwargs)


serve()
