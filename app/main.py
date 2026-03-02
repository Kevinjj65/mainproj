from flask import Flask
from flask_cors import CORS
from app.config import ensure_dirs
# Changed from app.api.routes to just app.api
from app.api import bp as api_bp 

def create_app():
    ensure_dirs()
    app = Flask(__name__)
    CORS(app, resources={r"*": {"origins": "*"}})
    
    # This registers all the routes linked in app/api/__init__.py
    app.register_blueprint(api_bp) 
    
    return app