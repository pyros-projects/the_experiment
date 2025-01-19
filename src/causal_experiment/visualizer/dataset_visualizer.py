import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
import json
import os

def load_dataset(file_path):
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    return data

def analyze_sequences(data):
    # Extract initial and final sequences
    initial_sequences = [d['prompt'] for d in data]
    final_sequences = [d['completion'].split('\n')[1] for d in data]
    
    # Count frequencies
    initial_freq = Counter(initial_sequences)
    final_freq = Counter(final_sequences)
    
    return initial_freq, final_freq

def analyze_sums(data):
    sums_data = []
    for d in data:
        initial_sum = sum(int(x) for x in d['prompt'].split(','))
        final_sum = sum(int(x) for x in d['completion'].split('\n')[1].split(','))
        sums_data.append((initial_sum, final_sum))
    return Counter(sums_data)

def create_visualizations(data, output_dir='visualizations'):
    os.makedirs(output_dir, exist_ok=True)
    
    # Use a built-in style
    plt.style.use('ggplot')
    
    # 1. Sequence frequency heatmaps
    initial_freq, final_freq = analyze_sequences(data)
    
    # Initial sequences heatmap
    plt.figure(figsize=(12, 8))
    initial_matrix = np.zeros((2, 5))  # 5 variables, 2 possible values
    for seq, freq in initial_freq.items():
        values = [int(x) for x in seq.split(',')]
        for i, v in enumerate(values):
            initial_matrix[v, i] += freq
    
    plt.imshow(initial_matrix, cmap='YlOrRd', aspect='auto')
    plt.colorbar(label='Frequency')
    for i in range(2):
        for j in range(5):
            plt.text(j, i, f'{int(initial_matrix[i,j])}', 
                    ha='center', va='center')
    
    plt.title('Initial State Distribution')
    plt.xlabel('Variable Position (A,B,C,D,E)')
    plt.ylabel('Value (0/1)')
    plt.savefig(f'{output_dir}/initial_state_heatmap.png', bbox_inches='tight', dpi=300)
    plt.close()
    
    # Final sequences heatmap
    plt.figure(figsize=(12, 8))
    final_matrix = np.zeros((2, 4))  # 4 variables in final state
    for seq, freq in final_freq.items():
        values = [int(x) for x in seq.split(',')]
        for i, v in enumerate(values):
            final_matrix[v, i] += freq
    
    plt.imshow(final_matrix, cmap='YlOrRd', aspect='auto')
    plt.colorbar(label='Frequency')
    for i in range(2):
        for j in range(4):
            plt.text(j, i, f'{int(final_matrix[i,j])}', 
                    ha='center', va='center')
    
    plt.title('Final State Distribution')
    plt.xlabel('Variable Position (B,C,D,E)')
    plt.ylabel('Value (0/1)')
    plt.savefig(f'{output_dir}/final_state_heatmap.png', bbox_inches='tight', dpi=300)
    plt.close()
    
    # 2. State transition diagram
    sums_transitions = analyze_sums(data)
    
    plt.figure(figsize=(12, 8))
    transition_matrix = np.zeros((6, 6))  # 0-5 possible sums
    for (initial, final), freq in sums_transitions.items():
        transition_matrix[initial, final] = freq
    
    plt.imshow(transition_matrix, cmap='viridis')
    plt.colorbar(label='Frequency')
    for i in range(6):
        for j in range(6):
            if transition_matrix[i,j] > 0:
                plt.text(j, i, f'{int(transition_matrix[i,j])}', 
                        ha='center', va='center', color='white')
    
    plt.title('Sum Transitions Heatmap')
    plt.xlabel('Final Sum')
    plt.ylabel('Initial Sum')
    plt.savefig(f'{output_dir}/sum_transitions_heatmap.png', bbox_inches='tight', dpi=300)
    plt.close()
    
    # 3. Top sequences bar plots
    plt.figure(figsize=(15, 6))
    
    # Top 10 initial sequences
    top_initial = dict(sorted(initial_freq.items(), key=lambda x: x[1], reverse=True)[:10])
    plt.bar(range(len(top_initial)), list(top_initial.values()))
    plt.xticks(range(len(top_initial)), list(top_initial.keys()), rotation=45)
    plt.title('Top 10 Most Common Initial Sequences')
    plt.xlabel('Sequence')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/top_initial_sequences.png', bbox_inches='tight', dpi=300)
    plt.close()
    
    # Top 10 final sequences
    plt.figure(figsize=(15, 6))
    top_final = dict(sorted(final_freq.items(), key=lambda x: x[1], reverse=True)[:10])
    plt.bar(range(len(top_final)), list(top_final.values()))
    plt.xticks(range(len(top_final)), list(top_final.keys()), rotation=45)
    plt.title('Top 10 Most Common Final Sequences')
    plt.xlabel('Sequence')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/top_final_sequences.png', bbox_inches='tight', dpi=300)
    plt.close()

