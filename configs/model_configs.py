"""
Central configuration for all Generative AI models used in the project.
This allows for easy updates and consistency.
"""

# --- Model for Generating Query Restatements ---
RESTATE_QUERIES_MODEL = "gemini-2.5-flash"

# --- Models for the Optimization Loop (we will use these later) ---
PROMPT_GENERATION_MODEL = "gemini-2.5-pro"

# --- Model for evaluating a prompt against our audio test suite. ---
TARGET_MODEL_FOR_EVAL = "gemini-live-2.5-flash"
