# aurevIa_timepiece/run_app.py
import os
import sys

# Add the src directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))

# Import the main function from the app
from app.app import main

if __name__ == "__main__":
    main()
