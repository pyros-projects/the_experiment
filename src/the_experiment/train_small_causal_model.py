import json
import torch
import os
from datasets import load_dataset, Dataset
from transformers import (
    GPT2Config,
    GPT2TokenizerFast,
    GPT2LMHeadModel,
    Trainer,
    TrainingArguments,
)
from loguru import logger
import sys
from typing import List, Dict, Any
from pathlib import Path


# Configure loguru
# Create logs directory if it doesn't exist
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True,parents=True)

logger.remove()  # Remove default handler
logger.add(
    sys.stdout,
    colorize=True,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    level="INFO"
)
logger.add(
    log_dir / "training_{time}.log",
    rotation="100 MB",
    retention="10 days",
    level="DEBUG"
)

MAX_LENGTH = 64  # Our inputs are short

def load_jsonl(path: str) -> List[Dict[str, Any]]:
    """
    Load data from a JSONL file with error handling.
    
    Args:
        path: Path to the JSONL file
    
    Returns:
        List of dictionaries containing the data
    """
    logger.info(f"Loading data from {path}")
    if not os.path.exists(path):
        logger.error(f"File not found: {path}")
        raise FileNotFoundError(f"File not found: {path}")
    
    lines = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                try:
                    lines.append(json.loads(line))
                except json.JSONDecodeError as e:
                    logger.warning(f"Error parsing line {line_num} in {path}: {e}")
                    continue
        
        logger.success(f"Successfully loaded {len(lines)} examples from {path}")
        return lines
    except Exception as e:
        logger.error(f"Error reading file {path}: {e}")
        raise
    

def create_hf_dataset(jsonl_data: List[Dict[str, str]]) -> Dataset:
    """
    Converts list of {"prompt": ..., "completion": ...} into a Hugging Face Dataset object.
    
    Args:
        jsonl_data: List of dictionaries containing prompts and completions
    
    Returns:
        Hugging Face Dataset object
    """
    logger.info("Creating Hugging Face dataset")
    texts = []
    for idx, item in enumerate(jsonl_data):
        try:
            if "prompt" not in item or "completion" not in item:
                logger.warning(f"Missing prompt or completion in item {idx}")
                continue
                
            merged_text = item["prompt"] + item["completion"]
            texts.append({"text": merged_text})
        except Exception as e:
            logger.error(f"Error processing item {idx}: {e}")
            continue
    
    logger.success(f"Created dataset with {len(texts)} examples")
    hf_dataset = Dataset.from_list(texts)
    return hf_dataset

def tokenize_function(example: Dict[str, str], tokenizer: GPT2TokenizerFast) -> Dict[str, torch.Tensor]:
    """
    Tokenizes the input text with error handling.
    
    Args:
        example: Dictionary containing the text to tokenize
        tokenizer: GPT2 tokenizer instance
    
    Returns:
        Dictionary containing tokenized inputs
    """
    try:
        result = tokenizer(
            example["text"],
            max_length=MAX_LENGTH,
            padding="max_length",
            truncation=True
        )
        result["labels"] = result["input_ids"].copy()
        return result
    except Exception as e:
        logger.error(f"Error tokenizing example: {e}")
        raise

def setup_training_environment(output_dir: str) -> None:
    """
    Sets up the training environment and checks for CUDA availability.
    
    Args:
        output_dir: Directory for saving model outputs
    """
    logger.info("Setting up training environment")
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    logger.info(f"Created output directory: {output_dir}")
    
    # Check CUDA availability
    if torch.cuda.is_available():
        device = torch.cuda.get_device_name(0)
        logger.info(f"Using GPU: {device}")
    else:
        logger.warning("No GPU available, using CPU")

def training(folder):
    try:
        # Setup training environment
        output_dir = f"./out/{folder}/tiny-gpt2-causal"
        setup_training_environment(output_dir)
        
        # 1. Load data
        logger.info("Starting data loading process")
        train_data = load_jsonl("dataset/train.jsonl")
        valid_data = load_jsonl("dataset/valid.jsonl")
        test_data = load_jsonl("dataset/test.jsonl")

        # 2. Create HF datasets
        logger.info("Creating datasets")
        train_dataset = create_hf_dataset(train_data)
        valid_dataset = create_hf_dataset(valid_data)
        test_dataset = create_hf_dataset(test_data)

        # 3. Initialize tokenizer
        logger.info("Initializing tokenizer")
        tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
        tokenizer.pad_token = tokenizer.eos_token

        # 4. Tokenize data
        logger.info("Tokenizing datasets")
        train_dataset = train_dataset.map(
            lambda ex: tokenize_function(ex, tokenizer),
            batched=False,
            desc="Tokenizing train dataset"
        )
        valid_dataset = valid_dataset.map(
            lambda ex: tokenize_function(ex, tokenizer),
            batched=False,
            desc="Tokenizing validation dataset"
        )
        test_dataset = test_dataset.map(
            lambda ex: tokenize_function(ex, tokenizer),
            batched=False,
            desc="Tokenizing test dataset"
        )

        # Set format
        logger.info("Setting dataset formats")
        for dataset, name in [(train_dataset, "train"),
                            (valid_dataset, "validation"),
                            (test_dataset, "test")]:
            dataset.set_format(type="torch",
                             columns=["input_ids", "attention_mask", "labels"])
            logger.debug(f"Set format for {name} dataset")

        # 5. Define model config
        logger.info("Configuring model")
        config = GPT2Config(
            vocab_size=tokenizer.vocab_size,
            n_embd=128,
            n_layer=4,
            n_head=4,
        )

        # 6. Instantiate model
        logger.info("Instantiating model")
        model = GPT2LMHeadModel(config)
        logger.debug(f"Model parameters: {model.num_parameters():,}")

        # 7. Training Arguments
        logger.info("Setting up training arguments")
        training_args = TrainingArguments(
            output_dir=output_dir,
            overwrite_output_dir=True,
            num_train_epochs=3,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            evaluation_strategy="epoch",
            logging_steps=100,
            save_steps=500,
            save_total_limit=2,
            learning_rate=5e-4,
        )

        # 8. Initialize Trainer
        logger.info("Initializing trainer")
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=valid_dataset,
        )

        # 9. Train
        logger.info("Starting training")
        trainer.train()
        logger.success("Training completed")

        # 10. Evaluate
        logger.info("Starting evaluation")
        eval_result = trainer.evaluate(eval_dataset=valid_dataset)
        perplexity = torch.exp(torch.tensor(eval_result["eval_loss"]))
        logger.info(f"Validation Perplexity: {perplexity:.2f}")

        # 11. Save final model

        logger.info("Saving model and tokenizer")
        final_output_dir = f"{output_dir}/final"
        trainer.save_model(final_output_dir)
        tokenizer.save_pretrained(final_output_dir)
        logger.success(f"Model and tokenizer saved to {final_output_dir}")

    except Exception as e:
        logger.exception(f"Training failed with error: {str(e)}")
        raise

