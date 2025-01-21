class RNNTrainingMonitor:
    """Monitor class that mimics the TrainerCallback interface for RNN training"""

    def __init__(self, callback, total_batches, epochs):
        self.callback = callback
        self.total_batches = total_batches
        self.epochs = epochs
        self.current_epoch = 0
        self.global_step = 0

        # Initialize state object to mimic Trainer state
        self.state = type(
            "State", (), {"epoch": 0, "global_step": 0, "log_history": []}
        )()

        # Initialize args object to mimic Trainer args
        self.args = type(
            "Args",
            (),
            {
                "num_train_epochs": epochs,
                "train_batch_size": 16,
                "gradient_accumulation_steps": 1,
            },
        )()

    def on_train_begin(self):
        """Called when training starts"""
        if self.callback:
            self.callback.on_init_end(self.args, self.state, None)
            self.callback.on_train_begin(self.args, self.state, None)

    def on_epoch_begin(self, epoch):
        """Called at the beginning of each epoch"""
        self.current_epoch = epoch
        self.state.epoch = epoch

    def on_batch_end(self, batch_idx, loss):
        """Called after each batch"""
        self.global_step += 1
        self.state.global_step = self.global_step

        if self.callback:
            # Create logs similar to what Trainer would create
            logs = {"loss": loss, "epoch": self.current_epoch, "step": self.global_step}
            self.state.log_history.append(logs)
            self.callback.on_log(self.args, self.state, None, logs)

    def on_validation_end(self, val_loss):
        """Called after validation"""
        if self.callback:
            metrics = {"eval_loss": val_loss}
            self.callback.on_evaluate(self.args, self.state, None, metrics)

    def on_train_end(self, final_loss):
        """Called when training ends"""
        if self.callback:
            if not self.state.log_history:
                self.state.log_history.append({"loss": final_loss})
            self.callback.on_train_end(self.args, self.state, None)
