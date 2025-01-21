import torch

from the_experiment.rules.rules import bool2int
from devtools import debug

from the_experiment.models.load_model import check_out_folder, load_models


class ModelEvaluator:
    def __init__(self):
        self.folder_contents = check_out_folder()
        self.active_folder = None

        if len(self.folder_contents) > 0:
            self.active_folder = self.folder_contents[0].folder
            self.reload_models(self.folder_contents[0].folder)

    def reload_models(self, folder):
        self.folder_contents = check_out_folder()
        if folder not in [f.folder for f in self.folder_contents]:
            self.active_folder = self.folder_contents[0].folder
        else:
            self.active_folder = folder
        loader = load_models(folder)
        if not loader:
            self.model = None
            self.rnn_model = None
            self.cnn_model = None
            self.tokenizer = None
        else:
            self.model, self.rnn_model, self.cnn_model, self.tokenizer = loader

    def eval_model(self, prompt_text: str) -> str:
        try:
            if self.model is None:
                return None

            input_ids = self.tokenizer.encode(prompt_text, return_tensors="pt")

            with torch.no_grad():
                output = self.model.generate(
                    input_ids, max_length=64, num_return_sequences=1, do_sample=False
                )

            output = self.tokenizer.decode(output[0])
            return str(output)
        except Exception:
            return None

    def eval_cnn(self, prompt_text: str) -> str:
        try:
            if self.cnn_model is None:
                return None
            model = self.cnn_model
            with torch.no_grad():
                # Tokenize input
                inputs = self.tokenizer(prompt_text, return_tensors="pt")
                input_ids = inputs["input_ids"].to(
                    torch.device("cuda" if torch.cuda.is_available() else "cpu")
                )

                logits = model(input_ids)

                # Get next token predictions for each position
                predicted_tokens = torch.argmax(logits[0], dim=-1)

                # Decode the predicted sequence and clean it up
                predicted_text = self.tokenizer.decode(predicted_tokens)

                return predicted_text
        except Exception as e:
            debug(e)
            return None

    def eval_rnn(self, prompt_text: str) -> str:
        try:
            if self.rnn_model is None or prompt_text == "":
                return ""
            if self.tokenizer is None:
                return ""
            model = self.rnn_model
            with torch.no_grad():
                # Tokenize input
                inputs = self.tokenizer(prompt_text, return_tensors="pt")
                if torch.cuda.is_available():
                    inputs = {k: v.cuda() for k, v in inputs.items()}

                # Generate output token by token
                max_length = 100
                generated_ids = []
                # Initialize hidden state
                batch_size = 1
                num_layers = 2
                hidden_dim = 128
                h0 = torch.zeros(num_layers, batch_size, hidden_dim)
                c0 = torch.zeros(num_layers, batch_size, hidden_dim)
                if torch.cuda.is_available():
                    h0 = h0.cuda()
                    c0 = c0.cuda()
                hidden = (h0, c0)

                # Start with input sequence
                current_ids = inputs["input_ids"]

                for _ in range(max_length):
                    # Get logits and hidden state
                    logits, hidden = model(current_ids, hidden)

                    # Get next token probabilities from the last position
                    next_token_logits = logits[:, -1, :]

                    # Sample next token
                    next_token = torch.argmax(next_token_logits, dim=-1)
                    generated_ids.append(next_token.item())

                    # Break if we hit the end token
                    if next_token.item() == self.tokenizer.eos_token_id:
                        break

                    # Update input for next iteration - ensure correct dimensions
                    current_ids = next_token.view(1, 1)

                # Decode the generated sequence
                output_text = self.tokenizer.decode(generated_ids)
                return output_text
        except Exception:
            return None

    def eval_model_bool(self, a, b, c, d, e) -> str:
        a = str(bool2int(a))
        b = str(bool2int(b))
        c = str(bool2int(c))
        d = str(bool2int(d))
        e = str(bool2int(e))
        prompt = f"{a},{b},{c},{d},{e}"
        return self.eval_model(prompt)

    def eval_rnn_bool(self, a, b, c, d, e) -> str:
        a = str(bool2int(a))
        b = str(bool2int(b))
        c = str(bool2int(c))
        d = str(bool2int(d))
        e = str(bool2int(e))
        prompt = f"{a},{b},{c},{d},{e}"
        return self.eval_rnn(prompt)

    def eval_cnn_bool(self, a, b, c, d, e) -> str:
        a = str(bool2int(a))
        b = str(bool2int(b))
        c = str(bool2int(c))
        d = str(bool2int(d))
        e = str(bool2int(e))
        prompt = f"{a},{b},{c},{d},{e}"
        return self.eval_cnn(prompt)

    def eval_model_int(self, a, b, c, d, e) -> str:
        a = str(a)
        b = str(b)
        c = str(c)
        d = str(d)
        e = str(e)
        prompt = f"{a},{b},{c},{d},{e}"
        return self.eval_model(prompt)


MODEL_EVALUATOR: ModelEvaluator = ModelEvaluator()
