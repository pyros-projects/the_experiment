from dataclasses import dataclass, field
from typing import List, Optional, Set, Dict
import json
from datetime import datetime
from fasthtml.common import *
import asyncio
from queue import Queue
from threading import Lock

@dataclass
class TrainingStats:
    """Stores training statistics for live updates"""
    model_name: str = ""
    epoch: float = 0
    batch: int = 0
    loss: float = 0.0
    total_batches: int = 0
    status: str = ""
    start_time: Optional[datetime] = None
    losses: List[float] = field(default_factory=list)
    
    def __post_init__(self):
        if self.start_time is None:
            self.start_time = datetime.now()
        
    def to_json(self):
        return json.dumps({
            "model_name": self.model_name,
            "epoch": round(self.epoch, 2),
            "batch": self.batch,
            "loss": round(self.loss, 4),
            "total_batches": self.total_batches,
            "status": self.status,
            "elapsed_time": str(datetime.now() - self.start_time).split('.')[0],
            "losses": [round(x, 4) for x in self.losses],
            "progress": round((self.batch / self.total_batches * 100) if self.total_batches > 0 else 0, 1)
        })

class TrainingMonitor:
    def __init__(self):
        self._stats = {}  # Dict to store stats for each model
        self._queues: Set[asyncio.Queue] = set()
        self._lock = Lock()
        self._sync_queue = Queue()  # Synchronous queue for updates
        
    def push_stats(self, **kwargs):
        """Synchronous method to update stats"""
        with self._lock:
            model_name = kwargs.get('model_name', 'default')
            if model_name not in self._stats:
                self._stats[model_name] = TrainingStats(model_name=model_name)
            
            stats = self._stats[model_name]
            
            # Update stats
            for k, v in kwargs.items():
                if hasattr(stats, k):
                    setattr(stats, k, v)
            
            # Add loss to history if it changed
            if 'loss' in kwargs:
                stats.losses.append(kwargs['loss'])
            
            # Put stats in sync queue for async processing
            self._sync_queue.put(stats)
    
    def subscribe(self, queue: asyncio.Queue):
        """Add a subscriber queue"""
        self._queues.add(queue)
    
    def unsubscribe(self, queue: asyncio.Queue):
        """Remove a subscriber queue"""
        self._queues.discard(queue)
    
    async def get_stats_stream(self):
        """Creates an SSE stream for stats updates"""
        async def stream():
            queue = asyncio.Queue()
            self.subscribe(queue)
            
            try:
                # Send initial stats
                for stats in self._stats.values():
                    yield f"data: {stats.to_json()}\n\n"
                
                while True:
                    # Check sync queue for updates
                    while not self._sync_queue.empty():
                        stats = self._sync_queue.get_nowait()
                        for q in list(self._queues):
                            try:
                                await q.put(stats)
                            except:
                                self._queues.discard(q)
                    
                    # Get updates from async queue
                    try:
                        stats = await asyncio.wait_for(queue.get(), timeout=0.1)
                        yield f"data: {stats.to_json()}\n\n"
                    except asyncio.TimeoutError:
                        continue  # No updates, check sync queue again
            finally:
                self.unsubscribe(queue)
        
        return EventStream(stream())

# Global monitor instance
monitor = TrainingMonitor()

def training_stats_component():
    """Returns the HTML component for displaying training stats"""
    return Card(cls="p-4 w-full max-w-[1200px] mx-auto")(
    #H2("Training Progress", cls="text-xl font-bold mb-4"),
    Div(
        # Status at the top
        Div(id="training-status", cls="text-lg font-medium mb-2 text-center"),
        
        # Progress bar
        Div(id="training-progress", cls="mb-4"),
        
        # Two-column layout for stats and chart
        Div(cls="flex gap-6")(
            # Stats grid on the left
            Div(
                id="training-stats",
                cls="flex flex-col gap-4 w-[200px]"  # Fixed width for the stats column
            ),
            
            # Chart container on the right
            Div(cls="flex-grow h-full min-h-[500px]")(
                Canvas(id="loss-graph", cls="w-full h-full bg-gray-100 rounded")
            )
        ),
        cls="space-y-4"
    ),
        Script(src="https://cdn.jsdelivr.net/npm/chart.js"),
        Script("""
            let lossChart;

            function initChart() {
                if (lossChart) {
                    lossChart.destroy();
                }
                const ctx = document.getElementById('loss-graph').getContext('2d');
                lossChart = new Chart(ctx, {
                    type: 'line',
                    data: {
                        labels: [],
                        datasets: [{
                            label: 'Training Loss',
                            data: [],
                            borderColor: 'rgb(75, 192, 192)',
                            backgroundColor: 'rgba(75, 192, 192, 0.1)',
                            tension: 0.3,
                            fill: true
                        }]
                    },
                    options: {
                        responsive: true,
                        maintainAspectRatio: false,
                        animation: {
                            duration: 150
                        },
                        scales: {
                            y: {
                                beginAtZero: true,
                                title: {
                                    display: true,
                                    text: 'Loss'
                                }
                            },
                            x: {
                                title: {
                                    display: true,
                                    text: 'Steps'
                                }
                            }
                        },
                        plugins: {
                            legend: {
                                display: true,
                                position: 'top'
                            }
                        }
                    }
                });
            }

            function updateStatus(stats) {
                let status = `${stats.model_name}: ${stats.status}`;
                if (stats.status.includes('Validation')) {
                    status = `<div class="font-bold text-blue-600">${status}</div>`;
                }
                document.getElementById('training-status').innerHTML = status;
            }

            function updateProgress(stats) {
                document.getElementById('training-progress').innerHTML = `
                    <div class="w-full bg-gray-200 rounded-full h-2.5">
                        <div class="bg-blue-600 h-2.5 rounded-full transition-all" 
                             style="width: ${stats.progress}%"></div>
                    </div>
                    <div class="text-center mt-2 text-sm text-gray-600">
                        Progress: ${stats.progress}% (${stats.batch}/${stats.total_batches} batches)
                    </div>
                `;
            }

            function updateStats(stats) {
                document.getElementById('training-stats').innerHTML = `
                    <div class="stat-box">
                        <div class="font-bold text-sm text-gray-600">Current Loss</div>
                        <div class="text-lg">${stats.loss.toFixed(4)}</div>
                    </div>
                    <div class="stat-box">
                        <div class="font-bold text-sm text-gray-600">Current Epoch</div>
                        <div class="text-lg">${stats.epoch}</div>
                    </div>
                    <div class="stat-box">
                        <div class="font-bold text-sm text-gray-600">Elapsed Time</div>
                        <div class="text-lg">${stats.elapsed_time}</div>
                    </div>
                    <div class="stat-box">
                        <div class="font-bold text-sm text-gray-600">Batch Progress</div>
                        <div class="text-lg">${stats.batch} / ${stats.total_batches}</div>
                    </div>
                `;
            }

            function updateChart(losses) {
                if (!lossChart) {
                    console.warn('Chart not initialized, initializing now...');
                    initChart();
                }
                try {
                    if (Array.isArray(losses)) {
                        const currentLoss = losses[losses.length - 1]; // Get the latest loss
                        const suggestedMax = currentLoss * 4;
                        
                        lossChart.data.labels = [...Array(losses.length).keys()];
                        lossChart.data.datasets[0].data = losses;
                        lossChart.options.scales.y.max = suggestedMax;
                        lossChart.update('none'); // Disable animation for smoother updates
                    } else {
                        console.warn('Losses data is not an array:', losses);
                    }
                } catch (error) {
                    console.error('Error updating chart:', error);
                }
            }

            // Initialize chart and set up SSE connection when DOM loads
            document.addEventListener('DOMContentLoaded', function() {
                console.log('Setting up SSE connection...');
                initChart();
                
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
                    if (lossChart) {
                        updateChart(stats.losses);
                    } else {
                        console.warn('Chart not initialized yet');
                        initChart();
                        updateChart(stats.losses);
                    }
                };
                
                evtSource.onerror = function(err) {
                    console.error('SSE Error:', err);
                };
            });
        """),
        Style("""
            .stat-box {
                padding: 1rem;
                background-color: #f3f4f6;
                border-radius: 0.5rem;
                box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.05);
            }
        """)
    )