from dataclasses import dataclass
import os
from pathlib import Path
import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from the_experiment.models.rnn.rnn_lm import LSTMLanguageModel  
from the_experiment.models.cnn.cnn_lm import CNNLanguageModel
from datetime import datetime

DEFAULT_LLM = "tiny-gpt2-causal/final"
DEFAULT_RNN = "rnn_lm_final.pt"
DEFAULT_CNN = "cnn_lm_final.pt"
DEFAULT_OUT = "./out"


@dataclass
class FolderContents:
    def __init__(self, folder=None, has_llm=False, has_rnn=False, has_cnn=False, last_modified=None):
        self.folder = folder
        self.last_modified = last_modified
        self.has_llm = has_llm # if subfolder 'tiny-gpt2-causal' exists
        self.has_rnn = has_rnn # if file 'rnn_lm_final.pt' exists
        self.has_cnn = has_cnn # if file 'cnn_lm_final.pt' exists
        self.text = f"({"LLM" if self.has_llm else ""}{"/" if self.has_llm and self.has_rnn  else ""}{"RNN" if self.has_rnn else ""}{"/" if self.has_rnn and self.has_cnn  else ""}{"CNN" if self.has_cnn else ""}) - {last_modified.strftime('%Y-%m-%d')}"
        

def check_out_folder(directory_path=DEFAULT_OUT) -> list[FolderContents]:
    try:
        # Ensure the provided path is a directory
        if not os.path.isdir(directory_path):
            print(f"The path '{directory_path}' is not a valid directory.")
            return []

        # List all entries in the directory
        entries = os.listdir(directory_path)

        # Filter and list only directories
        folders = [entry for entry in entries if os.path.isdir(os.path.join(directory_path, entry))]

        folder_contents_list = []

        for folder in folders:
            folder_path = os.path.join(directory_path, folder)
            subfolder_exists = os.path.isdir(os.path.join(folder_path, 'tiny-gpt2-causal'))
            rnn_file_exists = os.path.isfile(os.path.join(folder_path, 'rnn_lm_final.pt'))
            cnn_file_exists = os.path.isfile(os.path.join(folder_path, 'cnn_lm_final.pt'))
            last_modified_time = datetime.fromtimestamp(os.path.getmtime(folder_path))

            folder_contents = FolderContents(
                folder=folder,
                has_llm=subfolder_exists,
                has_rnn=rnn_file_exists,
                has_cnn=cnn_file_exists,
                last_modified=last_modified_time
            )

            folder_contents_list.append(folder_contents)
            folder_contents_list.sort(key=lambda x: x.last_modified, reverse=True)
            
        return folder_contents_list

    except Exception as e:
        print(f"An error occurred: {e}")
        return []

output_folders = check_out_folder()


def load_rnn(folder):
    if not folder:
        return None
    model_path = Path(DEFAULT_OUT) / folder / DEFAULT_RNN
    if not model_path.exists():
        return None
    model = LSTMLanguageModel(
        vocab_size=50257,
        embed_dim=128,
        hidden_dim=128,
        num_layers=2
    )
    
    # Load with safety settings
    state_dict = torch.load(
        model_path,
        weights_only=True  # Only load weights
    )
    model.load_state_dict(state_dict)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    return model    

def load_cnn(folder:str):
    if not folder:
        return None
    model_path = Path(DEFAULT_OUT) / folder / DEFAULT_CNN
    if not model_path.exists():
        return None
    
    model = CNNLanguageModel(
        vocab_size=50257,
        embed_dim=128,
        num_filters=128,
        kernel_size=3,
        seq_len=64
    )
    
    state_dict = torch.load(model_path, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    return model

def load_models(folder:str):
    if not folder:
        return None
    loader = load_model(folder)
    if not loader:
        llm_model, tokenizer = None, None
    else:
        llm_model, tokenizer = loader
    rnn_model = load_rnn(folder)
    cnn_model = load_cnn(folder)
    return llm_model,rnn_model,cnn_model, tokenizer


def load_model(folder: str):
    if not folder:
        return None
    
    model_path = Path(DEFAULT_OUT) / folder / DEFAULT_LLM
    if not model_path.exists():
        return None

    tokenizer = GPT2TokenizerFast.from_pretrained(model_path)
    model = GPT2LMHeadModel.from_pretrained(model_path)
    model.eval()
    return model, tokenizer