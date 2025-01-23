import json
import plotly.graph_objects as go
from typing import Dict

def generate_sankey_diagram(json_path: str, output_path: str):
    """
    Generate a Sankey diagram from a JSON file and save it as PNG.
    """
    # Load JSON data
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Extract nodes and links
    nodes = data['nodes']
    links = data['links']
    metadata = data.get('metadata', {})
    
    # Prepare Sankey data
    labels = [f"{node['name']} ({node['probability']:.2%})" for node in nodes]
    
    # Create Sankey diagram
    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=labels,
            color="orange"
        ),
        link=dict(
            source=[link['source'] for link in links],
            target=[link['target'] for link in links],
            value=[link['value'] for link in links]
        )
    )])
    
    # Update layout
    title = f"Token Probabilities"
    if 'prompt' in metadata:
        title += f" for prompt: '{metadata['prompt']}'"
        
    fig.update_layout(
        title_text=title,
        font_size=12,
        height=max(600, len(labels) * 30)
    )
    
    # Save to file
    fig.write_image(output_path)

if __name__ == "__main__":
    # Example usage
    generate_sankey_diagram(
        json_path='data/token_probabilities.json',
        output_path='data/sankey_diagram.png'
    )