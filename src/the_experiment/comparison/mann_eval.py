# file: inference_mann.py

import torch
from transformers import GPT2TokenizerFast
from typing import List, Optional, Tuple
from pathlib import Path
import numpy as np
from loguru import logger

from the_experiment.comparison.mann_lm import MANNLanguageModel

class MANNPredictor:
    def __init__(
        self,
        model_path: str,
        device: Optional[torch.device] = None
    ):
        """Initialize the MANN predictor with a trained model."""
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize tokenizer
        self.tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model
        self.model = MANNLanguageModel(
            vocab_size=self.tokenizer.vocab_size,
            embed_dim=128,
            hidden_dim=256,
            memory_size=128,
            memory_vector_dim=64
        )
        
        # Load trained weights
        state_dict = torch.load(model_path, map_location=self.device)
        if isinstance(state_dict, dict) and 'model_state_dict' in state_dict:
            state_dict = state_dict['model_state_dict']
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()
        
        logger.info(f"Model loaded from {model_path}")
        logger.info(f"Using device: {self.device}")

    def reset_memory(self):
        """Reset the model's memory state."""
        self.model.memory = type(self.model.memory)(
            self.model.memory.memory_size,
            self.model.memory.memory_vector_dim,
            self.model.memory.hidden_dim
        ).to(self.device)

    def predict_next_token(
        self,
        input_text: str,
        hidden_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None
    ) -> Tuple[str, Tuple[torch.Tensor, torch.Tensor]]:
        """Predict the next token given an input text."""
        # Tokenize input
        input_ids = self.tokenizer.encode(input_text, return_tensors='pt').to(self.device)
        
        with torch.no_grad():
            # Get model predictions
            logits, new_hidden = self.model(input_ids, hidden_state)
            next_token_logits = logits[0, -1, :]
            
            # Apply temperature
            if temperature != 1.0:
                next_token_logits = next_token_logits / temperature
            
            # Apply top-k filtering if specified
            if top_k is not None:
                indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                next_token_logits[indices_to_remove] = float('-inf')
            
            # Apply top-p (nucleus) filtering if specified
            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                
                # Remove tokens with cumulative probability above the threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                next_token_logits[indices_to_remove] = float('-inf')
            
            # Sample from the filtered distribution
            probs = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            return self.tokenizer.decode(next_token.item()), new_hidden

    def generate(
        self,
        prompt: str,
        max_length: int = 50,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        stop_tokens: Optional[List[str]] = None,
        reset_memory: bool = True
    ) -> str:
        """Generate text continuation from a prompt."""
        if stop_tokens is None:
            stop_tokens = ['.', '!', '?']
            
        if reset_memory:
            self.reset_memory()
            
        current_text = prompt
        generated_text = ""
        hidden = None
        
        for _ in range(max_length):
            next_token, hidden = self.predict_next_token(
                current_text,
                hidden,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p
            )
            
            generated_text += next_token
            current_text += next_token
            
            # Check for stop tokens
            if any(stop_token in next_token for stop_token in stop_tokens):
                break
                
        return generated_text

    @torch.no_grad()
    def evaluate_perplexity(self, test_text: str, reset_memory: bool = True) -> float:
        """Calculate the perplexity of the model on a test text."""
        if reset_memory:
            self.reset_memory()
            
        # Tokenize input
        input_ids = self.tokenizer.encode(test_text, return_tensors='pt').to(self.device)
        
        # Forward pass
        logits, _ = self.model(input_ids)
        
        # Calculate loss
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = input_ids[..., 1:].contiguous()
        
        loss_fct = torch.nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        
        # Calculate perplexity
        return torch.exp(loss).item()

def main():
    """Example usage of the MANNPredictor."""
    model_path = "out/experiment/mann_lm_best.pt"
    predictor = MANNPredictor(model_path)
    
    # Example predictions
    prompt = "The cat sat on"
    logger.info(f"Prompt: {prompt}")
    
    # Generate with different parameters
    logger.info("Generating with default parameters:")
    output = predictor.generate(prompt)
    logger.info(f"Generated: {output}")
    
    logger.info("\nGenerating with high temperature (more random):")
    output = predictor.generate(prompt, temperature=1.5, top_k=50)
    logger.info(f"Generated: {output}")
    
    logger.info("\nGenerating with low temperature (more focused):")
    output = predictor.generate(prompt, temperature=0.7, top_p=0.9)
    logger.info(f"Generated: {output}")
    
    # Calculate perplexity
    test_text = "The cat sat on the mat."
    perplexity = predictor.evaluate_perplexity(test_text)
    logger.info(f"\nPerplexity on test text: {perplexity:.2f}")

if __name__ == "__main__":
    main()