import logging

from .to_arviz import results_to_arviz
from .from_arviz import get_spline_model, get_weights, get_periodogram
from .compare_results import compare_results

logging.getLogger("arviz").setLevel(logging.ERROR)
