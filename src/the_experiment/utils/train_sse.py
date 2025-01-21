# file: train_sse.py
import asyncio
import json
import time
from fasthtml.common import *
from the_experiment.utils.callbacks import llm_log_queue
from sse_starlette.sse import EventSourceResponse
from devtools import debug

shutdown_event = signal_shutdown()

async def sse_training_logs():
    """
    Async generator that yields SSE messages from the `llm_log_queue`.
    """
    while not shutdown_event.is_set():
        # check for next item
        item = llm_log_queue.get()
        if item is None:
            # no logs right now; short sleep
            await asyncio.sleep(0.5)
            continue
        # If there's an item, yield an SSE message
        data_str = json.dumps(item)
        debug(f"Sending SSE message: {data_str}")
        yield sse_message(data_str)
        # small delay to avoid spamming
        await asyncio.sleep(0.05)

def train_llm_sse(request):
    """
    Return an event-stream that pushes the training logs from the queue.
    """
    async def send_stats(stats):
        yield f"data: {stats.to_json()}\n\n"
    
    async def stream():
        queue = asyncio.Queue()
        
        async def callback(stats):
            await queue.put(stats)
        
        monitor.subscribe(callback)
        
        try:
            while True:
                stats = await queue.get()
                yield f"data: {stats.to_json()}\n\n"
        finally:
            monitor.unsubscribe(callback)
    return EventStream(sse_training_logs())
