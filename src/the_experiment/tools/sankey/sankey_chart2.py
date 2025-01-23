import json
import plotly.graph_objects as go
from typing import Dict

def generate_sankey_diagram(json_path: str, output_path: str, probability_threshold: float = 0.05):
    """
    Generate a Sankey diagram from a JSON file and save it as PNG.
    """
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Filter nodes and links based on probability
    nodes = []
    node_mapping = {}  # To map original indices to new ones
    for idx, node in enumerate(data['nodes']):
        if node['probability'] >= probability_threshold:
            node_mapping[idx] = len(nodes)
            nodes.append(node)
    
    # Only keep links between remaining nodes
    links = []
    for link in data['links']:
        if (link['source'] in node_mapping and 
            link['target'] in node_mapping):
            links.append({
                'source': node_mapping[link['source']],
                'target': node_mapping[link['target']],
                'value': link['value']
            })
    
    # Prepare labels
    labels = [f"{node['name']} ({node['probability']:.2%})" for node in nodes]
    
    # Color coding based on probability ranges
    def get_color(prob):
        if prob >= 0.3: return "#1f77b4"  # Dark blue for high prob
        if prob >= 0.1: return "#7fcdbb"  # Medium blue for medium prob
        return "#c7e9b4"  # Light blue for lower prob
    
    node_colors = [get_color(node['probability']) for node in nodes]
    
    # Create Sankey diagram
    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=labels,
            color=node_colors
        ),
        link=dict(
            source=[link['source'] for link in links],
            target=[link['target'] for link in links],
            value=[link['value'] for link in links]
        )
    )])
    
    # Update layout
    title = f"Token Probabilities"
    if 'metadata' in data and 'prompt' in data['metadata']:
        title += f" for prompt: '{data['metadata']['prompt']}'"
        
    fig.update_layout(
        title_text=title,
        font_size=12,
        height=max(600, len(labels) * 30)
    )
    
    fig.write_image(output_path)

if __name__ == "__main__":
    generate_sankey_diagram(
        json_path='data/token_probabilities.json',
        output_path='data/sankey_diagram.png',
        probability_threshold=0.05  # Hide tokens with < 5% probability
    )