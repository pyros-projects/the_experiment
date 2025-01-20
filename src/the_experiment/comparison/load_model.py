import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from the_experiment.comparison.rnn_lm import LSTMLanguageModel  
from the_experiment.comparison.cnn_lm import CNNLanguageModel

DEFAULT_PATH = "./out/tiny-gpt2-causal/final"
DEFAULT_RNN = "./out/rnn_lm_final.pt"
DEFAULT_CNN = "./out/cnn_lm_final.pt"


def load_rnn():
    model = LSTMLanguageModel(
        vocab_size=50257,
        embed_dim=128,
        hidden_dim=128,
        num_layers=2
    )
    
    # Load with safety settings
    state_dict = torch.load(
        DEFAULT_RNN,
        weights_only=True  # Only load weights
    )
    model.load_state_dict(state_dict)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    return model    

def load_cnn():

    
    model = CNNLanguageModel(
        vocab_size=50257,
        embed_dim=128,
        num_filters=128,
        kernel_size=3,
        seq_len=64
    )
    
    state_dict = torch.load(DEFAULT_CNN, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    return model

def load_models():
    model_path = DEFAULT_PATH
    tokenizer = GPT2TokenizerFast.from_pretrained(model_path)
    model = GPT2LMHeadModel.from_pretrained(model_path)
    model.eval()

    rnn_model = load_rnn()
    cnn_model = load_cnn()
    return model,rnn_model,cnn_model, tokenizer


def load_model():
    model_path = DEFAULT_PATH
    tokenizer = GPT2TokenizerFast.from_pretrained(model_path)
    model = GPT2LMHeadModel.from_pretrained(model_path)
    model.eval()
    return model, tokenizer