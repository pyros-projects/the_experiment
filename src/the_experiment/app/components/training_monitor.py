from dataclasses import dataclass, field
from typing import List, Optional, Set
import json
from datetime import datetime
from fasthtml.common import *
import asyncio
from queue import Queue
from threading import Lock

@dataclass
class ChartConfig:
    """Configuration for a training chart"""
    model_name: str
    class_name: str  # For color and style
    color_main: str
    color_bg: str

CHART_CONFIGS = {
    'LLM': ChartConfig('LLM', 'bg-blue-600', 'rgb(59, 130, 246)', 'rgba(59, 130, 246, 0.1)'),
    'RNN': ChartConfig('RNN', 'bg-green-600', 'rgb(16, 185, 129)', 'rgba(16, 185, 129, 0.1)'),
    'CNN': ChartConfig('CNN', 'bg-red-600', 'rgb(239, 68, 68)', 'rgba(239, 68, 68, 0.1)')
}



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
        return json.dumps(
            {
                "model_name": self.model_name,
                "epoch": round(self.epoch, 2),
                "batch": self.batch,
                "loss": round(self.loss, 4),
                "total_batches": self.total_batches,
                "status": self.status,
                "elapsed_time": str(datetime.now() - self.start_time).split(".")[0],
                "losses": [round(x, 4) for x in self.losses],
                "progress": round(
                    (self.batch / self.total_batches * 100)
                    if self.total_batches > 0
                    else 0,
                    1,
                ),
            }
        )


class TrainingMonitor:
    def __init__(self):
        self._stats = {}  # Dict to store stats for each model
        self._queues: Set[asyncio.Queue] = set()
        self._lock = Lock()
        self._sync_queue = Queue()  # Synchronous queue for updates

    def push_stats(self, **kwargs):
        """Synchronous method to update stats"""
        with self._lock:
            model_name = kwargs.get("model_name", "default")
            if model_name not in self._stats:
                self._stats[model_name] = TrainingStats(model_name=model_name)

            stats = self._stats[model_name]

            # Update stats
            for k, v in kwargs.items():
                if hasattr(stats, k):
                    setattr(stats, k, v)

            # Add loss to history if it changed
            if "loss" in kwargs:
                stats.losses.append(kwargs["loss"])

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
    return Card(cls="p-2 w-full")(
        Div(id="charts-container", cls="grid gap-4"),
        Script(src="https://cdn.jsdelivr.net/npm/chart.js"),
        Script("""
            const charts = {};
            const chartData = {};

            function createChartContainer(modelName) {
                return `
                    <div class="chart-wrapper p-3 bg-white rounded-lg shadow-lg transition-all duration-300" 
                        data-model="${modelName}" 
                        onclick="toggleChartFocus('${modelName}')">
                        <div class="flex justify-between items-center mb-2">
                            <div class="font-bold text-lg">${modelName}</div>
                            <div id="training-status-${modelName}" class="text-sm"></div>
                        </div>
                        
                        <div id="training-progress-${modelName}" class="mb-2"></div>
                        
                        <div class="flex gap-4">
                            <div id="training-stats-${modelName}" 
                                class="flex flex-col gap-2 w-[160px]">
                            </div>
                            
                            <div class="flex-grow chart-height">
                                <canvas id="loss-graph-${modelName}" 
                                    class="w-full h-full">
                                </canvas>
                            </div>
                        </div>
                    </div>
                `;
            }

            let focusedModel = null;
               
            function toggleChartFocus(modelName) {
                const container = document.getElementById('charts-container');
                if (!container) return;

                // If clicking the already focused model, unfocus it
                if (focusedModel === modelName) {
                    focusedModel = null;
                } else {
                    focusedModel = modelName;
                }
                
                updateLayout();
                
                // Trigger a resize event to make charts redraw
                window.dispatchEvent(new Event('resize'));
            }
               
            function initChart(modelName) {
                if (charts[modelName]) {
                    charts[modelName].destroy();
                }
                
                const canvas = document.getElementById(`loss-graph-${modelName}`);
                if (!canvas) return;
                
                const ctx = canvas.getContext('2d');
                charts[modelName] = new Chart(ctx, {
                    type: 'line',
                    data: {
                        labels: [],
                        datasets: [{
                            label: 'Loss',
                            data: [],
                            borderColor: modelName === 'LLM' ? 'rgb(59, 130, 246)' :
                                       modelName === 'RNN' ? 'rgb(16, 185, 129)' :
                                       'rgb(239, 68, 68)',
                            backgroundColor: modelName === 'LLM' ? 'rgba(59, 130, 246, 0.1)' :
                                           modelName === 'RNN' ? 'rgba(16, 185, 129, 0.1)' :
                                           'rgba(239, 68, 68, 0.1)',
                            tension: 0.3,
                            fill: true
                        }]
                    },
                    options: {
                        responsive: true,
                        maintainAspectRatio: false,
                        animation: { duration: 150 },
                        scales: {
                            y: {
                                beginAtZero: true,
                                title: { display: false },
                                grid: { display: true, color: 'rgba(0,0,0,0.05)' }
                            },
                            x: {
                                title: { display: false },
                                grid: { display: false }
                            }
                        },
                        plugins: {
                            legend: { display: false }
                        }
                    }
                });
            }

            function updateStatus(stats) {
                const statusEl = document.getElementById(`training-status-${stats.model_name}`);
                if (!statusEl) return;
                
                let status = stats.status;
                if (stats.status.includes('Validation')) {
                    status = `<span class="font-bold text-blue-600">${status}</span>`;
                }
                statusEl.innerHTML = status;
            }

            function updateProgress(stats) {
                const progressEl = document.getElementById(`training-progress-${stats.model_name}`);
                if (!progressEl) return;
                
                const color = stats.model_name === 'LLM' ? 'bg-blue-600' :
                             stats.model_name === 'RNN' ? 'bg-green-600' :
                             'bg-red-600';
                             
                progressEl.innerHTML = `
                    <div class="w-full bg-gray-100 rounded-full h-1.5">
                        <div class="${color} h-1.5 rounded-full transition-all" 
                             style="width: ${stats.progress}%"></div>
                    </div>
                    <div class="text-center mt-1 text-xs text-gray-600">
                        ${stats.batch}/${stats.total_batches} batches (${stats.progress}%)
                    </div>
                `;
            }

            function updateStats(stats) {
                const statsEl = document.getElementById(`training-stats-${stats.model_name}`);
                if (!statsEl) return;
                
                statsEl.innerHTML = `
                    <div class="stat-box">
                        <div class="text-xs text-gray-600">Loss</div>
                        <div class="font-bold">${stats.loss.toFixed(4)}</div>
                    </div>
                    <div class="stat-box">
                        <div class="text-xs text-gray-600">Epoch</div>
                        <div class="font-bold">${stats.epoch}</div>
                    </div>
                    <div class="stat-box">
                        <div class="text-xs text-gray-600">Time</div>
                        <div class="font-bold">${stats.elapsed_time}</div>
                    </div>
                `;
            }

            function updateChart(modelName, losses) {
                const chart = charts[modelName];
                if (!chart || !Array.isArray(losses)) return;
                
                const currentLoss = losses[losses.length - 1];
                const suggestedMax = currentLoss * 2;
                
                chart.data.labels = [...Array(losses.length).keys()];
                chart.data.datasets[0].data = losses;
                chart.options.scales.y.max = suggestedMax;
                chart.update('none');
            }

            function updateLayout() {
                const activeModels = Object.keys(charts).length;
                const container = document.getElementById('charts-container');
                if (!container) return;

                if (activeModels === 1) {
                    // Single chart layout
                    container.className = 'grid gap-4 grid-cols-1';
                    const wrapper = container.querySelector('.chart-wrapper');
                    if (wrapper) {
                        wrapper.className = 'chart-wrapper p-3 bg-white rounded-lg shadow-lg col-span-full w-full transition-all duration-300';
                        const chartContainer = wrapper.querySelector('.flex-grow');
                        if (chartContainer) {
                            chartContainer.className = 'flex-grow h-[400px]';
                        }
                    }
                } else if (activeModels > 1 && focusedModel) {
                    // Focused layout
                    container.className = 'grid gap-4 grid-cols-4';
                    
                    container.querySelectorAll('.chart-wrapper').forEach(wrapper => {
                        const modelName = wrapper.getAttribute('data-model');
                        if (modelName === focusedModel) {
                            wrapper.className = 'chart-wrapper p-3 bg-white rounded-lg shadow-lg col-span-3 cursor-pointer transition-all duration-300 hover:shadow-xl';
                            const chartContainer = wrapper.querySelector('.flex-grow');
                            if (chartContainer) {
                                chartContainer.className = 'flex-grow h-[400px]';
                            }
                        } else {
                            wrapper.className = 'chart-wrapper p-3 bg-white rounded-lg shadow-lg col-span-1 cursor-pointer transition-all duration-300 hover:shadow-xl';
                            const chartContainer = wrapper.querySelector('.flex-grow');
                            if (chartContainer) {
                                chartContainer.className = 'flex-grow h-[200px]';
                            }
                        }
                    });
                } else {
                    // Default grid layout
                    container.className = `grid gap-4 ${
                        activeModels === 2 ? 'grid-cols-2' : 'grid-cols-3'
                    }`;
                    container.querySelectorAll('.chart-wrapper').forEach(wrapper => {
                        wrapper.className = 'chart-wrapper p-3 bg-white rounded-lg shadow-lg cursor-pointer transition-all duration-300 hover:shadow-xl';
                        const chartContainer = wrapper.querySelector('.flex-grow');
                        if (chartContainer) {
                            chartContainer.className = 'flex-grow h-[220px]';
                        }
                    });
                }
                
                // Update all charts to fit their new containers
                Object.values(charts).forEach(chart => chart.resize());
            }

            // Clean up existing SSE connection if it exists
            function cleanupMonitor() {
                if (window.trainingEventSource) {
                    window.trainingEventSource.close();
                    window.trainingEventSource = null;
                }
                
                Object.values(charts).forEach(chart => {
                    if (chart) chart.destroy();
                });
                Object.keys(charts).forEach(key => delete charts[key]);
            }

            function initializeTrainingMonitor() {
                console.log('Initializing training monitor...');
                
                cleanupMonitor();
                
                window.trainingEventSource = new EventSource('/training-stats');
                
                window.trainingEventSource.onopen = function() {
                    console.log('SSE connection opened');
                };
                
                window.trainingEventSource.onmessage = function(event) {
                    console.log('SSE message received:', event.data);
                    const stats = JSON.parse(event.data);
                    const modelName = stats.model_name;
                    
                    if (!document.getElementById(`loss-graph-${modelName}`)) {
                        const container = document.getElementById('charts-container');
                        if (container) {
                            container.insertAdjacentHTML('beforeend', createChartContainer(modelName));
                            initChart(modelName);
                            updateLayout();
                        }
                    }
                    
                    updateStatus(stats);
                    updateProgress(stats);
                    updateStats(stats);
                    updateChart(modelName, stats.losses);
                };
                
                window.trainingEventSource.onerror = function(err) {
                    console.error('SSE Error:', err);
                    cleanupMonitor();
                };
            }

            // Listen for HTMX page swaps
            document.addEventListener('htmx:afterSwap', function(evt) {
                if (evt.detail.target.id === 'main-area') {
                    const hasTrainingMonitor = document.getElementById('charts-container');
                    if (hasTrainingMonitor) {
                        console.log('Training view loaded, initializing monitor...');
                        initializeTrainingMonitor();
                    } else {
                        console.log('Not training view, cleaning up monitor...');
                        cleanupMonitor();
                    }
                }
            });

            // Initial setup
            if (document.getElementById('charts-container')) {
                initializeTrainingMonitor();
            }
        """),
        Style("""
            .stat-box {
                padding: 0.5rem;
                background-color: #f8fafc;
                border-radius: 0.375rem;
            }
            .chart-wrapper {
                background: white;
                border-radius: 0.5rem;
                overflow: hidden;
                cursor: pointer;
                transition: all 0.3s ease-in-out;
            }
            .chart-wrapper:hover {
                transform: translateY(-2px);
            }
            .chart-height {
                transition: height 0.3s ease-in-out;
            }
        """)
    )