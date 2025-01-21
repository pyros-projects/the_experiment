# file: train_mann.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import GPT2TokenizerFast
import sys
from pathlib import Path
from loguru import logger
import time
from tqdm import tqdm

from the_experiment.models.load_data import MiniworldTextDataset
from the_experiment.models.mann.mann_lm import MANNLanguageModel

# Configure loguru
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True, parents=True)

logger.remove()
logger.add(
    sys.stdout,
    colorize=True,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    level="INFO",
)
logger.add(
    log_dir / "mann_training_{time}.log",
    rotation="100 MB",
    retention="10 days",
    level="DEBUG",
)


def setup_training_environment() -> torch.device:
    """Set up the training environment and return the appropriate device."""
    logger.info("Setting up training environment")

    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
        logger.debug(f"CUDA version: {torch.version.cuda}")
        logger.debug(
            f"Available GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB"
        )
    else:
        device = torch.device("cpu")
        logger.warning("No GPU available, using CPU")

    return device


def validate_model(
    model: nn.Module,
    valid_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    vocab_size: int,
) -> float:
    """Validate the model on the validation dataset."""
    logger.info("Starting validation")
    model.eval()
    val_loss = 0.0
    num_batches = len(valid_loader)
    hidden = None

    try:
        with torch.no_grad():
            for input_ids, attention_mask in tqdm(valid_loader, desc="Validation"):
                input_ids = input_ids.to(device)

                # Forward pass
                logits, hidden = model(input_ids, hidden)

                # Detach hidden state
                if hidden is not None:
                    hidden = tuple(h.detach() for h in hidden)

                # Calculate loss
                shift_logits = logits[:, :-1, :].contiguous().view(-1, vocab_size)
                shift_labels = input_ids[:, 1:].contiguous().view(-1)
                loss = criterion(shift_logits, shift_labels)
                val_loss += loss.item()

        avg_val_loss = val_loss / num_batches
        logger.info(f"Validation Loss: {avg_val_loss:.4f}")
        return avg_val_loss

    except Exception as e:
        logger.error(f"Error during validation: {str(e)}")
        raise
    finally:
        model.train()


def training_mann(folder):
    """Main training function for the Memory Augmented Neural Network."""
    try:
        # Setup
        start_time = time.time()
        device = setup_training_environment()
        out_dir = Path(f"out/{folder}")
        out_dir.mkdir(exist_ok=True, parents=True)

        # Initialize tokenizer
        logger.info("Initializing tokenizer")
        tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
        tokenizer.pad_token = tokenizer.eos_token
        vocab_size = tokenizer.vocab_size

        # Load datasets
        logger.info("Loading datasets")
        train_dataset = MiniworldTextDataset("dataset/train.jsonl", tokenizer)
        valid_dataset = MiniworldTextDataset("dataset/valid.jsonl", tokenizer)

        # Create dataloaders
        batch_size = 16  # Smaller batch size due to memory usage
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=lambda b: (
                torch.stack([x["input_ids"] for x in b]),
                torch.stack([x["attention_mask"] for x in b]),
            ),
        )
        valid_loader = DataLoader(
            valid_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=lambda b: (
                torch.stack([x["input_ids"] for x in b]),
                torch.stack([x["attention_mask"] for x in b]),
            ),
        )

        # Initialize model
        logger.info("Initializing model")
        model = MANNLanguageModel(
            vocab_size=vocab_size,
            embed_dim=128,
            hidden_dim=256,
            memory_size=128,
            memory_vector_dim=64,
            num_layers=1,
        )
        model.to(device)
        model.init_weights()

        # Setup training
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=2, verbose=True
        )
        epochs = 3
        best_val_loss = float("inf")

        # Training loop
        logger.info("Starting training")
        for epoch in range(epochs):
            model.train()
            total_loss = 0.0
            hidden = None

            for batch_idx, (input_ids, attention_mask) in enumerate(
                tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}")
            ):
                try:
                    input_ids = input_ids.to(device)

                    # Forward pass
                    logits, hidden = model(input_ids, hidden)

                    # Detach hidden state
                    if hidden is not None:
                        hidden = tuple(h.detach() for h in hidden)

                    # Calculate loss
                    shift_logits = logits[:, :-1, :].contiguous().view(-1, vocab_size)
                    shift_labels = input_ids[:, 1:].contiguous().view(-1)
                    loss = criterion(shift_logits, shift_labels)

                    # Backward pass
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()

                    total_loss += loss.item()

                    if batch_idx % 100 == 0:
                        logger.debug(f"Batch {batch_idx}, Loss: {loss.item():.4f}")

                except Exception as e:
                    logger.error(f"Error in batch {batch_idx}: {str(e)}")
                    continue

            # Epoch statistics
            avg_loss = total_loss / len(train_loader)
            logger.info(f"Epoch {epoch + 1} - Average Loss: {avg_loss:.4f}")

            # Validation
            val_loss = validate_model(
                model, valid_loader, criterion, device, vocab_size
            )
            scheduler.step(val_loss)

            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "val_loss": val_loss,
                    },
                    f"{out_dir}/mann_lm_best.pt",
                )
                logger.info(f"Saved best model with validation loss: {val_loss:.4f}")

        # Save final model
        torch.save(model.state_dict(), f"{out_dir}/mann_lm_final.pt")

        # Training summary
        total_time = time.time() - start_time
        logger.success(f"Training completed in {total_time:.2f}s")
        logger.success(f"Best validation loss: {best_val_loss:.4f}")

    except Exception as e:
        logger.exception(f"Training failed with error: {str(e)}")
        raise
