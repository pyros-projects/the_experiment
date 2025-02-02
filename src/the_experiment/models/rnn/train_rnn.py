# file: train_rnn.py

import threading
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import GPT2TokenizerFast
import sys
from pathlib import Path
from typing import Tuple, List, Dict
from loguru import logger
import time
from tqdm import tqdm

from the_experiment.models.load_data import MiniworldTextDataset
from the_experiment.models.rnn.rnn_lm import LSTMLanguageModel
from the_experiment.models.rnn.rnn_monitor import RNNTrainingMonitor


# Configure loguru
# Create logs directory if it doesn't exist
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True, parents=True)

logger.remove()  # Remove default handler
logger.add(
    sys.stdout,
    colorize=True,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    level="INFO",
)
logger.add(
    log_dir / "rnn_training_{time}.log",
    rotation="100 MB",
    retention="10 days",
    level="DEBUG",
)


def setup_training_environment() -> torch.device:
    """
    Set up the training environment and return the appropriate device.

    Returns:
        torch.device: The device to be used for training
    """
    logger.info("Setting up training environment")

    # Check CUDA availability
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


def collate_fn(
    batch: List[Dict[str, torch.Tensor]],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Collate function for DataLoader with error handling.

    Args:
        batch: List of dictionaries containing input_ids and attention_mask

    Returns:
        Tuple of input_ids and attention_mask tensors
    """
    try:
        input_ids = torch.stack([item["input_ids"] for item in batch], dim=0)
        attention_mask = torch.stack([item["attention_mask"] for item in batch], dim=0)
        return input_ids, attention_mask
    except Exception as e:
        logger.error(f"Error in collate_fn: {str(e)}")
        logger.debug(f"Batch contents: {batch}")
        raise


def validate_model(
    model: nn.Module,
    valid_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    vocab_size: int,
) -> float:
    """
    Validate the model on the validation dataset.

    Args:
        model: The LSTM model
        valid_loader: DataLoader for validation data
        criterion: Loss function
        device: Device to run validation on
        vocab_size: Size of vocabulary

    Returns:
        float: Average validation loss
    """
    logger.info("Starting validation")
    model.eval()
    val_loss = 0.0
    num_batches = len(valid_loader)

    try:
        with torch.no_grad():
            for input_ids, attention_mask in tqdm(valid_loader, desc="Validation"):
                input_ids = input_ids.to(device)
                logits, hidden = model(input_ids)
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


def training_rnn(folder, callback=None):
    """
    Main training function for the RNN language model.
    """
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
        logger.debug(f"Vocabulary size: {vocab_size}")

        # Load datasets
        logger.info("Loading datasets")
        try:
            train_dataset = MiniworldTextDataset("dataset/train.jsonl", tokenizer)
            valid_dataset = MiniworldTextDataset("dataset/valid.jsonl", tokenizer)
            logger.info(
                f"Loaded {len(train_dataset)} training examples and {len(valid_dataset)} validation examples"
            )
        except Exception as e:
            logger.error(f"Error loading datasets: {str(e)}")
            raise

        # Create dataloaders
        logger.info("Creating DataLoaders")
        train_loader = DataLoader(
            train_dataset, batch_size=16, shuffle=True, collate_fn=collate_fn
        )
        valid_loader = DataLoader(
            valid_dataset, batch_size=16, shuffle=False, collate_fn=collate_fn
        )
        logger.debug(f"Number of training batches: {len(train_loader)}")
        logger.debug(f"Number of validation batches: {len(valid_loader)}")

        # Initialize model
        logger.info("Initializing model")
        model = LSTMLanguageModel(
            vocab_size=vocab_size, embed_dim=128, hidden_dim=128, num_layers=2
        )
        model.to(device)
        logger.debug(
            f"Model parameters: {sum(p.numel() for p in model.parameters()):,}"
        )

        # Setup training
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        epochs = 3
        best_val_loss = float("inf")

        callbacks = [callback] if callback else []
        th = threading.Thread(
            target=background_rnn_thread,
            args=(
                callbacks,
                epochs,
                model,
                train_loader,
                device,
                vocab_size,
                criterion,
                optimizer,
                best_val_loss,
                out_dir,
                valid_loader,
                start_time,
            ),
            daemon=True,
        )
        th.start()

    except Exception as e:
        logger.exception(f"Training failed with error: {str(e)}")
        raise


def background_rnn_thread(
    callbacks,
    epochs,
    model,
    train_loader,
    device,
    vocab_size,
    criterion,
    optimizer,
    best_val_loss,
    out_dir,
    valid_loader,
    start_time,
):
    # Initialize monitor
    total_batches = len(train_loader) * epochs
    monitor = RNNTrainingMonitor(
        callbacks[0] if callbacks else None, total_batches, epochs
    )

    # Signal training start
    monitor.on_train_begin()

    # Training loop
    logger.info("Starting training")
    for epoch in range(epochs):
        epoch_start_time = time.time()
        total_loss = 0.0
        model.train()

        # Signal epoch start
        monitor.on_epoch_begin(epoch)

        # Training epoch
        for batch_idx, (input_ids, attention_mask) in enumerate(
            tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}")
        ):
            try:
                input_ids = input_ids.to(device)
                attention_mask = attention_mask.to(device)

                logits, hidden = model(input_ids)
                shift_logits = logits[:, :-1, :].contiguous().view(-1, vocab_size)
                shift_labels = input_ids[:, 1:].contiguous().view(-1)

                loss = criterion(shift_logits, shift_labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

                # Report batch progress
                monitor.on_batch_end(batch_idx, loss.item())

                if batch_idx % 100 == 0:
                    logger.debug(
                        f"Epoch {epoch + 1}, Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}"
                    )

            except Exception as e:
                logger.error(f"Error in training batch {batch_idx}: {str(e)}")
                continue

        # Epoch statistics
        avg_loss = total_loss / len(train_loader)
        epoch_time = time.time() - epoch_start_time
        logger.info(f"Epoch {epoch + 1}/{epochs} complete:")
        logger.info(f"Average Loss: {avg_loss:.4f}")
        logger.info(f"Epoch Time: {epoch_time:.2f}s")

        # Validation
        val_loss = validate_model(model, valid_loader, criterion, device, vocab_size)
        monitor.on_validation_end(val_loss)

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            logger.info("New best validation loss! Saving model checkpoint...")
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "train_loss": avg_loss,
                    "val_loss": val_loss,
                },
                f"./{out_dir}/rnn_lm_best.pt",
            )

    # Signal training end
    monitor.on_train_end(avg_loss)

    # Save final model and wrap up as before
    logger.info("Saving final model")
    try:
        torch.save(model.state_dict(), f"./{out_dir}/rnn_lm_final.pt")
        logger.success("Model saved successfully")
    except Exception as e:
        logger.error(f"Error saving model: {str(e)}")
        raise

    total_time = time.time() - start_time
    logger.success(f"Training completed in {total_time:.2f}s")
    logger.success(f"Best validation loss: {best_val_loss:.4f}")
