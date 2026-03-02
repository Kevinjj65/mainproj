from flask import Blueprint

bp = Blueprint("api", __name__)

# Import routes to register them with the blueprint
# These must be imported after 'bp' is defined to avoid circular imports
from . import (
    llm_routes,
    translate_routes,
    rag_routes,
    inference_routes,
    benchmark_routes,
    system_routes
)