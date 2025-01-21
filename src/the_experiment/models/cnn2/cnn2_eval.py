# file: inference_cnn.py

import torch
from transformers import GPT2TokenizerFast
from typing import List, Optional
from pathlib import Path
import numpy as np
from loguru import logger

from the_experiment.comparison.cnn_lm import CNNLanguageModel


class CNNPredictor:
    def __init__(self, model_path: str, device: Optional[torch.device] = None):
        """Initialize the CNN predictor with a trained model."""
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # Initialize tokenizer
        self.tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
        self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load model
        self.model = CNNLanguageModel(
            vocab_size=self.tokenizer.vocab_size,
            embed_dim=128,
            hidden_dim=128,
            num_layers=3,
        )

        # Load trained weights
        state_dict = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

        logger.info(f"Model loaded from {model_path}")
        logger.info(f"Using device: {self.device}")

    def predict_next_token(
        self,
        input_text: str,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
    ) -> str:
        """Predict the next token given an input text."""
        # Tokenize input
        input_ids = self.tokenizer.encode(input_text, return_tensors="pt").to(
            self.device
        )

        with torch.no_grad():
            # Get model predictions
            logits, _ = self.model(input_ids)
            next_token_logits = logits[0, -1, :]

            # Apply temperature
            if temperature != 1.0:
                next_token_logits = next_token_logits / temperature

            # Apply top-k filtering if specified
            if top_k is not None:
                indices_to_remove = (
                    next_token_logits
                    < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                )
                next_token_logits[indices_to_remove] = float("-inf")

            # Apply top-p (nucleus) filtering if specified
            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(
                    next_token_logits, descending=True
                )
                cumulative_probs = torch.cumsum(
                    torch.softmax(sorted_logits, dim=-1), dim=-1
                )

                # Remove tokens with cumulative probability above the threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[
                    ..., :-1
                ].clone()
                sorted_indices_to_remove[..., 0] = 0

                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                next_token_logits[indices_to_remove] = float("-inf")

            # Sample from the filtered distribution
            probs = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            return self.tokenizer.decode(next_token.item())

    def generate(
        self,
        prompt: str,
        max_length: int = 50,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        stop_tokens: Optional[List[str]] = None,
    ) -> str:
        """Generate text continuation from a prompt."""
        if stop_tokens is None:
            stop_tokens = [".", "!", "?"]

        current_text = prompt
        generated_text = ""

        for _ in range(max_length):
            next_token = self.predict_next_token(
                current_text, temperature=temperature, top_k=top_k, top_p=top_p
            )

            generated_text += next_token
            current_text += next_token

            # Check for stop tokens
            if any(stop_token in next_token for stop_token in stop_tokens):
                break

        return generated_text

    @torch.no_grad()
    def evaluate_perplexity(self, test_text: str) -> float:
        """Calculate the perplexity of the model on a test text."""
        # Tokenize input
        input_ids = self.tokenizer.encode(test_text, return_tensors="pt").to(
            self.device
        )

        # Forward pass
        logits, _ = self.model(input_ids)

        # Calculate loss
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = input_ids[..., 1:].contiguous()

        loss_fct = torch.nn.CrossEntropyLoss()
        loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
        )

        # Calculate perplexity
        return torch.exp(loss).item()


def main():
    """Example usage of the CNNPredictor."""
    model_path = "out/experiment/cnn_lm_best.pt"
    predictor = CNNPredictor(model_path)

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
