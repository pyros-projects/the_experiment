from loguru import logger
from devtools import debug
class CNNTrainingMonitor:
    """Monitor class that mimics the TrainerCallback interface for CNN training"""
    def __init__(self, callback, total_batches, epochs):
        logger.info(f"Initializing CNNTrainingMonitor with {total_batches} batches, {epochs} epochs")
        self.callback = callback
        self.total_batches = total_batches
        self.epochs = epochs
        self.current_epoch = 0
        self.global_step = 0
        self.losses = []  # Track losses for charting
        
        # Initialize state object to mimic Trainer state
        self.state = type('State', (), {
            'epoch': 0,
            'global_step': 0,
            'log_history': [],
            'max_steps': total_batches
        })()
        
        # Initialize args object to mimic Trainer args - FIXED: Now storing in self.args
        self.args = type('Args', (), {
            'num_train_epochs': epochs,
            'train_batch_size': 16,
            'gradient_accumulation_steps': 1,
            'max_steps': total_batches
        })()
        
        debug({
            "monitor_init": {
                "total_batches": total_batches,
                "epochs": epochs,
                "callback_present": callback is not None
            }
        })
        
    def on_train_begin(self):
        """Called when training starts"""
        if self.callback:
            logger.info(f"Starting training with callback: {self.callback.model_name}")
            self.callback.total_batches = self.total_batches
            debug({
                "train_begin": {
                    "total_batches": self.total_batches,
                    "model_name": self.callback.model_name
                }
            })
            self.callback.on_init_end(self.args, self.state, None)
            self.callback.on_train_begin(self.args, self.state, None)
    
    def on_epoch_begin(self, epoch):
        """Called at the beginning of each epoch"""
        logger.info(f"Beginning epoch {epoch}")
        self.current_epoch = epoch
        self.state.epoch = epoch
    
    def on_batch_end(self, batch_idx, loss):
        """Called after each batch"""
        self.global_step += 1
        self.state.global_step = self.global_step
        self.losses.append(loss)  # Track loss for charting
        
        if batch_idx % 50 == 0:  # Log more frequently
            logger.debug(f"Batch {batch_idx}/{self.total_batches}, Loss: {loss:.4f}")
        
        if self.callback:
            logs = {
                'loss': loss,
                'learning_rate': 0.001,
                'epoch': self.current_epoch,
                'step': self.global_step
            }
            self.state.log_history.append(logs)
            
            debug({
                "batch_stats": {
                    "model": self.callback.model_name,
                    "batch": batch_idx,
                    "global_step": self.global_step,
                    "total_batches": self.total_batches,
                    "loss": loss,
                    "loss_history_len": len(self.losses)
                }
            })
            
            self.callback.on_log(self.args, self.state, None, logs)
            
    def on_validation_end(self, val_loss):
        """Called after validation"""
        if self.callback:
            metrics = {
                'eval_loss': val_loss,
                'losses': self.losses  # Include loss history
            }
            debug({
                "validation": {
                    "model": self.callback.model_name,
                    "val_loss": val_loss,
                    "current_epoch": self.current_epoch,
                    "step": self.global_step
                }
            })
            self.callback.on_evaluate(self.args, self.state, None, metrics)