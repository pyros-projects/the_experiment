from dataclasses import dataclass
from fasthtml.common import *
from fasthtml.components import Sl_tab_group, Sl_tab, Sl_checkbox, Sl_button, Sl_split_panel, Sl_input
from monsterui import *

from the_experiment.models.cnn.train_cnn import training_cnn
from the_experiment.models.gpt2.train_small_causal_model import training

from the_experiment.app.components.training_monitor import monitor, training_stats_component
from transformers import TrainerCallback
from devtools import debug
import asyncio
from starlette.background import BackgroundTasks

from the_experiment.models.model_eval import MODEL_EVALUATOR
from the_experiment.models.rnn.train_rnn import training_rnn

@dataclass
class TrainState:
    LLM: bool = False
    RNN: bool = False
    CNN: bool = False
    MANN: bool = False
    CNN2: bool = False
    folder: str = "Hello"
    
train_state = TrainState()

@patch
def __ft__(self:TrainState):
    return Div(
        AX(Sl_checkbox("Train LLM",cls="m-4",id="train_llm",checked=self.LLM),"/set/LLM","checkboxes",hx_swap="outerHTML"),
        AX(Sl_checkbox("Train RNN",cls="m-4",id="train_rnn", checked=self.RNN),"/set/RNN","checkboxes",hx_swap="outerHTML"),
        AX(Sl_checkbox("Train CNN",cls="m-4",id="train_cnn", checked=self.CNN),"/set/CNN","checkboxes",hx_swap="outerHTML"),
        AX(Sl_checkbox("Train MANN",cls="m-4",id="train_mann", checked=self.MANN),"/set/MANN","checkboxes",hx_swap="outerHTML"),
        AX(Sl_checkbox("Train CNN2",cls="m-4",id="train_cnn2", checked=self.CNN2),"/set/CNN2","checkboxes",hx_swap="outerHTML"),
        Sl_input(self.folder,name="folder",id="folder",cls="m-4"),
        id="train_state"
    )

class StatsCallback(TrainerCallback):
    from devtools import debug
    """Callback to monitor training progress"""
    def __init__(self, model_name):
        self.model_name = model_name
        self.total_batches = 0
        
    def on_init_end(self, args, state, control, **kwargs):
        """Called when trainer initialization ends"""
        debug(state)
        debug(args)
        state.global_step
        self.total_batches = (
            args.num_train_epochs
            * (20000 // (args.train_batch_size * args.gradient_accumulation_steps))
        )
        monitor.push_stats(
            model_name=self.model_name,
            epoch=0,
            batch=0,
            loss=0.0,
            total_batches=self.total_batches,
            status="Initializing..."
        )
        return control

    def on_train_begin(self, args, state, control, **kwargs):
        monitor.push_stats(
            model_name=self.model_name,
            epoch=0,
            batch=0,
            loss=0.0,
            total_batches=self.total_batches,
            status="Training started"
        )
        return control
        
    def on_log(self, args, state, control, logs, **kwargs):
        if 'loss' in logs:
            monitor.push_stats(
                model_name=self.model_name,
                epoch=state.epoch,
                batch=state.global_step,
                loss=logs['loss'],
                total_batches=self.total_batches,
                status="Training in progress"
            )
        return control

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics:
            monitor.push_stats(
                model_name=self.model_name,
                epoch=state.epoch,
                batch=state.global_step,
                loss=metrics.get('eval_loss', 0.0),
                total_batches=self.total_batches,
                status=f"Validation Loss: {metrics.get('eval_loss', 0.0):.4f}"
            )
        return control

    def on_train_end(self, args, state, control, **kwargs):
        monitor.push_stats(
            model_name=self.model_name,
            epoch=state.epoch,
            batch=self.total_batches,
            loss=state.log_history[-1].get('loss', 0) if state.log_history else 0,
            total_batches=self.total_batches,
            status="Training completed!"
        )
        return control

def checkboxes():
    return Form(id="checkboxes",cls="flex grid grid-cols-1 gap-4 w-[500px] m-4")(
        train_state,
       
        Sl_button("Train selected",hx_post='/train',hx_request="include:#folder",hx_swap='none'),
        Sl_button("Train all",hx_post='/train_all',hx_request="include:#folder",hx_swap='none')
    )

def TrainView(rt):
    @rt("/set/{var}")
    def get(var: str):
        current_folder = getattr(train_state, "folder")
        current = getattr(train_state, var)
        setattr(train_state, var, not current)
        setattr(train_state, "folder", current_folder)
        return checkboxes()
    
    @rt("/training-stats")
    async def get():
        return await monitor.get_stats_stream()

    @rt("/train")
    async def post(folder: str):
        debug(folder)
        tasks = []
        
        if train_state.LLM:
            callback = StatsCallback("LLM")
            # Make sure training() returns a coroutine 
            if asyncio.iscoroutinefunction(training):
                task = asyncio.create_task(training(folder, callback=callback))
            else:
                # If training is not async, wrap it in run_in_threadpool
                task = asyncio.create_task(
                    run_in_threadpool(training, folder, callback=callback)
                )
            tasks.append(task)
            
        if train_state.RNN:
            callback = StatsCallback("RNN")
            if asyncio.iscoroutinefunction(training_rnn):
                task = asyncio.create_task(training_rnn(folder, callback=callback))
            else:
                task = asyncio.create_task(
                    run_in_threadpool(training_rnn, folder, callback=callback)
                )
            tasks.append(task)
            
        if train_state.CNN:
            callback = StatsCallback("CNN")
            if asyncio.iscoroutinefunction(training_cnn):
                task = asyncio.create_task(training_cnn(folder, callback=callback))
            else:
                task = asyncio.create_task(
                    run_in_threadpool(training_cnn, folder, callback=callback)
                )
            tasks.append(task)
            
        await asyncio.gather(*tasks)
        
        active_folder = MODEL_EVALUATOR.active_folder
        MODEL_EVALUATOR.reload_models(active_folder)

    return Sl_split_panel(position="30")(
        Div(fill_form(checkboxes(), train_state), slot="start"),
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
                    """)
                )
            ),
     
        )
    )
        
    