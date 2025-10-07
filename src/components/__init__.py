"""
Components module
"""

from .canvas import render_drawing_canvas, process_canvas_data
from .config import render_sidebar_config
from .results import (
    render_progress_section,
    render_api_calls_log,
    render_fitting_comparison,
    render_pareto_frontier,
    render_score_history
)

__all__ = [
    'render_drawing_canvas',
    'process_canvas_data',
    'render_sidebar_config',
    'render_progress_section',
    'render_api_calls_log',
    'render_fitting_comparison',
    'render_pareto_frontier',
    'render_score_history',
]
