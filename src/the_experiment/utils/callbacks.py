
from collections import deque
import threading
from transformers import TrainerCallback
from typing import Dict
import asyncio
from the_experiment.components.training_monitor import monitor


class TrainingLogQueue:
    def __init__(self):
        self.lock = threading.Lock()
        self.queue = deque()

    def put(self, **item):
        with self.lock:
            self.queue.append(item)

    def get(self):
        """Pops a single item, or None if empty."""
        with self.lock:
            if len(self.queue) > 0:
                return self.queue.popleft()
            return None
        
llm_log_queue = TrainingLogQueue()


class ProgressCallback(TrainerCallback):
    def init(self, model_name):
        self.log_progress_fn = llm_log_queue.put
        self.model_name = "GPT2"
        num_batches_per_epoch = 20000 //16
        self.total_batches = num_batches_per_epoch * 3
        self.setup(20000,16,3,"GPT2")

    def setup(self, num_train_examples, train_batch_size,num_train_epochs, model_name):
        """Called when training starts"""
        num_batches_per_epoch = num_train_examples //train_batch_size
        self.total_batches = num_batches_per_epoch * num_train_epochs
        
        llm_log_queue.put(
            model_name="GPT2",
            epoch=0,
            batch=0,
            loss=0.0,
            total_batches=3750,
            status="Starting training..."
        )

    def on_log(self, args, state, control, logs: Dict[str, float], **kwargs):
        """Called on each log (determined by logging_steps)"""
        if 'loss' in logs:
            current_step = state.global_step
            current_epoch = state.epoch

            llm_log_queue.put(
                model_name="GPT2",
                epoch=current_epoch,
                batch=current_step,
                loss=logs['loss'],
                total_batches=3750,
                status="Training"
            )

    def on_evaluate(self, args, state, control, metrics, **kwargs):
        """Called after each evaluation"""
        if 'eval_loss' in metrics:
            llm_log_queue.put(
                model_name="GPT2",
                epoch=state.epoch,
                batch=state.global_step,
                loss=metrics['eval_loss'],
                total_batches=3750,
                status=f"Validation Loss: {metrics['eval_loss']:.4f}"
            )

    def on_train_end(self, args, state, control, **kwargs):
        """Called when training ends"""
        llm_log_queue.put(
            model_name="GPT2",
            epoch=state.epoch,
            batch=3750,
            loss=state.log_history[-1].get('loss', 0),
            total_batches=3750,
            status="Training completed!"
        )

# class ProgressCallback(TrainerCallback):
#     def __init__(self, model_name):
#         self.model_name = model_name
#         self.setup(20000,16,3,"GPT2")
        
#     def setup(self, num_train_examples, train_batch_size,num_train_epochs, model_name):
#         """Called when training starts"""
#         num_batches_per_epoch = num_train_examples //train_batch_size
#         self.total_batches = num_batches_per_epoch * num_train_epochs
        
#         asyncio.create_task(monitor.log_progress(
#             model_name=self.model_name,
#             epoch=0,
#             batch=0,
#             loss=0.0,
#             total_batches=self.total_batches,
#             status="Starting training..."
#         ))
    
#     def on_log(self, args, state, control, logs: Dict[str, float], **kwargs):
#         """Called on each log (determined by logging_steps)"""
#         if 'loss' in logs:
#             current_step = state.global_step
#             current_epoch = state.epoch
            
#             asyncio.create_task(monitor.log_progress(
#                 model_name=self.model_name,
#                 epoch=current_epoch,
#                 batch=current_step,
#                 loss=logs['loss'],
#                 total_batches=self.total_batches,
#                 status="Training"
#             ))
            
#     def on_evaluate(self, args, state, control, metrics, **kwargs):
#         """Called after each evaluation"""
#         if 'eval_loss' in metrics:
#             asyncio.create_task(monitor.log_progress(
#                 model_name=self.model_name,
#                 epoch=state.epoch,
#                 batch=state.global_step,
#                 loss=metrics['eval_loss'],
#                 total_batches=self.total_batches,
#                 status=f"Validation Loss: {metrics['eval_loss']:.4f}"
#             ))
    
#     def on_train_end(self, args, state, control, **kwargs):
#         """Called when training ends"""
#         asyncio.create_task(monitor.log_progress(
#             model_name=self.model_name,
#             epoch=state.epoch,
#             batch=self.total_batches,
#             loss=state.log_history[-1].get('loss', 0),
#             total_batches=self.total_batches,
#             status="Training completed!"
#         ))

# class ProgressCallback(TrainerCallback):
#     def __init__(self, model_name):
#         self.model_name = model_name
#         self.setup(20000,16,3,"GPT2")
        
#     def setup(self, num_train_examples, train_batch_size,num_train_epochs, model_name):
#         """Called when training starts"""
#         num_batches_per_epoch = num_train_examples //train_batch_size
#         self.total_batches = num_batches_per_epoch * num_train_epochs
        
#         asyncio.create_task(monitor.log_progress(
#             model_name=self.model_name,
#             epoch=0,
#             batch=0,
#             loss=0.0,
#             total_batches=self.total_batches,
#             status="Starting training..."
#         ))
    
#     def on_log(self, args, state, control, logs: Dict[str, float], **kwargs):
#         """Called on each log (determined by logging_steps)"""
#         if 'loss' in logs:
#             current_step = state.global_step
#             current_epoch = state.epoch
            
#             asyncio.create_task(monitor.log_progress(
#                 model_name=self.model_name,
#                 epoch=current_epoch,
#                 batch=current_step,
#                 loss=logs['loss'],
#                 total_batches=self.total_batches,
#                 status="Training"
#             ))
            
#     def on_evaluate(self, args, state, control, metrics, **kwargs):
#         """Called after each evaluation"""
#         if 'eval_loss' in metrics:
#             asyncio.create_task(monitor.log_progress(
#                 model_name=self.model_name,
#                 epoch=state.epoch,
#                 batch=state.global_step,
#                 loss=metrics['eval_loss'],
#                 total_batches=self.total_batches,
#                 status=f"Validation Loss: {metrics['eval_loss']:.4f}"
#             ))
    
#     def on_train_end(self, args, state, control, **kwargs):
#         """Called when training ends"""
#         asyncio.create_task(monitor.log_progress(
#             model_name=self.model_name,
#             epoch=state.epoch,
#             batch=self.total_batches,
#             loss=state.log_history[-1].get('loss', 0),
#             total_batches=self.total_batches,
#             status="Training completed!"
#         ))