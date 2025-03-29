#!/usr/bin/env python3

"""
Simple runner script for the AI Image Analysis app.
This allows users to run the app from the project root.
"""

from src.hackaton.app import run_app

if __name__ == "__main__":
    print("Starting AI Image Analysis tool...")
    print("Loading models - this may take a few moments on first run.")
    print("Access the web interface at http://localhost:5000 once server starts.")
    run_app()
