from dataclasses import dataclass
from fasthtml.common import *
from fasthtml.components import (
    Sl_checkbox,
    Sl_button,
    Sl_split_panel,
    Sl_input,
    Sl_card,
)
from monsterui import *
from monsterui.all import *
from starlette.background import BackgroundTask
from devtools import debug
from the_experiment.models.cnn.train_cnn import training_cnn
from the_experiment.models.gpt2.train_small_causal_model import training
from loguru import logger
from the_experiment.app.components.training_monitor import (
    monitor,
    training_stats_component,
)
from transformers import TrainerCallback
from devtools import debug
import asyncio

from the_experiment.models.model_eval import MODEL_EVALUATOR
from the_experiment.models.rnn.train_rnn import training_rnn


@dataclass
class TrainingForm:
    folder: str
    train_llm: bool = False
    train_rnn: bool = False
    train_cnn: bool = False

    def __post_init__(self):
        if not self.folder or self.folder.isspace():
            raise ValueError("Output folder cannot be empty")


class StatsCallback(TrainerCallback):
    """Callback to monitor training progress"""
    def __init__(self, model_name):
        logger.info(f"Initializing StatsCallback for model: {model_name}")
        self.model_name = model_name
        self.total_batches = 0
        self.start_time = None
        self.losses = []
        
    def on_init_end(self, args, state, control, **kwargs):
        """Called when trainer initialization ends"""
        if hasattr(args, 'max_steps') and args.max_steps > 0:
            self.total_batches = args.max_steps
        else:
            self.total_batches = (
                args.num_train_epochs 
                * getattr(args, 'train_batch_size', 16)
            )
            
        debug({
            "callback_init": {
                "model": self.model_name,
                "total_batches_calculated": self.total_batches,
                "args_available": [attr for attr in dir(args) if not attr.startswith('_')]
            }
        })
        
        # Send initial stats
        monitor.push_stats(
            model_name=self.model_name,
            epoch=0,
            batch=0,
            loss=0.0,
            total_batches=self.total_batches,
            status="Initializing...",
            losses=[]  # Initialize empty losses array
        )
        return control

    def on_train_begin(self, args, state, control, **kwargs):
        monitor.push_stats(
            model_name=self.model_name,
            epoch=0,
            batch=0,
            loss=0.0,
            total_batches=self.total_batches,
            status="Training started",
            losses=self.losses
        )
        return control
        
    def on_log(self, args, state, control, logs, **kwargs):
        if 'loss' in logs:
            loss = logs['loss']
            self.losses.append(loss)
            
            if len(self.losses) % 50 == 0:  # Debug every 50 losses
                debug({
                    "stats_update": {
                        "model": self.model_name,
                        "current_loss": loss,
                        "loss_history_len": len(self.losses),
                        "batch": getattr(state, 'global_step', 0),
                        "total_batches": self.total_batches
                    }
                })
            
            monitor.push_stats(
                model_name=self.model_name,
                epoch=getattr(state, 'epoch', 0),
                batch=getattr(state, 'global_step', 0),
                loss=loss,
                total_batches=self.total_batches,
                status="Training in progress",
                losses=self.losses
            )
        return control

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics:
            eval_loss = metrics.get('eval_loss', 0.0)
            debug({
                "evaluation": {
                    "model": self.model_name,
                    "eval_loss": eval_loss,
                    "loss_history_len": len(self.losses),
                    "batch": getattr(state, 'global_step', 0)
                }
            })
            
            monitor.push_stats(
                model_name=self.model_name,
                epoch=getattr(state, 'epoch', 0),
                batch=getattr(state, 'global_step', 0),
                loss=eval_loss,
                total_batches=self.total_batches,
                status=f"Validation Loss: {eval_loss:.4f}",
                losses=self.losses
            )
        return control

    def on_train_end(self, args, state, control, **kwargs):
        final_loss = self.losses[-1] if self.losses else 0
        debug({
            "training_end": {
                "model": self.model_name,
                "final_loss": final_loss,
                "total_losses_recorded": len(self.losses),
                "total_batches_completed": getattr(state, 'global_step', 0)
            }
        })
        
        monitor.push_stats(
            model_name=self.model_name,
            epoch=getattr(state, 'epoch', 0),
            batch=self.total_batches,
            loss=final_loss,
            total_batches=self.total_batches,
            status="Training completed!",
            losses=self.losses
        )

def checkboxes():
    return Sl_card(cls="rounded-lg shadow-lg ml-4")(
        Div(Strong("Training configurator"), slot="header"),
        Form(id="checkboxes", cls="flex grid grid-cols-3 gap-4 w-auto m-4")(
            # Add error div at the top - will be updated via HTMX
            Sl_input(
                label="Name for training run",
                name="folder",
                id="folder",
                cls="col-span-3",
                required=True,
            ),
            Sl_checkbox("Train LLM", cls="m-4", name="train_llm"),
            Sl_checkbox("Train RNN", cls="m-4", name="train_rnn"),
            Sl_checkbox("Train CNN", cls="m-4", name="train_cnn"),
            Div(id="form-errors", cls="col-span-3"),
            Sl_button(
                "Train selected",
                hx_post="/train",
                hx_target="#form-errors",  # Target the error div
                hx_request="include:#train_llm,#train_rnn,#train_cnn,#folder",
                cls="col-span-3",
            ),
            Sl_button(
                "Train all",
                hx_post="/train_all",
                hx_target="#form-errors",  # Target the error div
                hx_request="include:#folder",
                cls="col-span-3",
            ),
        ),
    )


def TrainView(rt):
    @rt("/training-stats")
    async def get():
        return await monitor.get_stats_stream()

    @rt("/train")
    async def post(req):
        try:
            # Parse form data into our dataclass
            form_data = await req.form()
            training_form_data = TrainingForm(
                folder=form_data.get("folder"),
                train_llm=form_data.get("train_llm") == "on",
                train_rnn=form_data.get("train_rnn") == "on",
                train_cnn=form_data.get("train_cnn") == "on",
            )
            debug(training_form_data)
            # If validation passes, proceed with training
            debug(training_form_data.folder)
            debug(training_form_data.train_llm)
            debug(training_form_data.train_rnn)
            debug(training_form_data.train_cnn)

            tasks = []

            if training_form_data.train_llm:
                callback = StatsCallback("LLM")
                # Make sure training() returns a coroutine
                if asyncio.iscoroutinefunction(training):
                    task = asyncio.create_task(
                        training(training_form_data.folder, callback=callback)
                    )
                else:
                    # If training is not async, wrap it in run_in_threadpool
                    task = asyncio.create_task(
                        run_in_threadpool(
                            training, training_form_data.folder, callback=callback
                        )
                    )
                tasks.append(task)

            if training_form_data.train_rnn:
                callback = StatsCallback("RNN")
                if asyncio.iscoroutinefunction(training_rnn):
                    task = asyncio.create_task(
                        training_rnn(training_form_data.folder, callback=callback)
                    )
                else:
                    task = asyncio.create_task(
                        run_in_threadpool(
                            training_rnn, training_form_data.folder, callback=callback
                        )
                    )
                tasks.append(task)

            if training_form_data.train_cnn:
                callback = StatsCallback("CNN")
                if asyncio.iscoroutinefunction(training_cnn):
                    task = asyncio.create_task(
                        training_cnn(training_form_data.folder, callback=callback)
                    )
                else:
                    task = asyncio.create_task(
                        run_in_threadpool(
                            training_cnn, training_form_data.folder, callback=callback
                        )
                    )
                tasks.append(task)

            debug(tasks)
            await asyncio.gather(*tasks)

            active_folder = MODEL_EVALUATOR.active_folder
            MODEL_EVALUATOR.reload_models(active_folder)

            return Div(cls="text-green-500")("Training started successfully!")

        except ValueError as e:
            # Return validation error
            return Div(cls="text-red-500 p-2 border border-red-500 rounded")(str(e))

    return Sl_split_panel(position="30")(
        Div(Container(checkboxes()), slot="start"),
        Div(slot="end")(
            Div(cls="w-[100%] items-center")(
                Div(
                    H1("Training Monitor", cls="text-2xl font-bold text-center"),
                    training_stats_component(),
                    Script("""
                        document.addEventListener('DOMContentLoaded', function() {
                            console.log('Setting up SSE connection...');
                            const evtSource = new EventSource('/training-stats');
                            
                            evtSource.onopen = function() {
                                console.log('SSE connection opened');
                            };
                            
                            evtSource.onmessage = function(event) {
                                console.log('SSE message received:', event.data);
                                const stats = JSON.parse(event.data);
                                updateStatus(stats);
                                updateProgress(stats);
                                updateStats(stats);
                                updateChart(stats.losses);
                            };
                            
                            evtSource.onerror = function(err) {
                                console.error('SSE Error:', err);
                            };
                        });
                    """),
                )
            ),
        ),
    )
