"""
Central configuration for all Generative AI models used in the project.
This allows for easy updates and consistency.
"""

# --- Model for Generating Query Restatements ---
# We use a powerful model for this creative task to get high-quality, diverse restatements.
# Note: "Gemini 2.5" is not a public model name as of late 2024.
# We are using the latest available powerful model. Replace with "gemini-2.5-..." if it becomes available.
RESTATE_QUERIES_MODEL = "gemini-2.5-flash"

RESTATE_QUERIES_CONFIG = {
    "temperature": 0.7,
    "top_p": 0.95,
}

# --- Models for the Optimization Loop (we will use these later) ---

# Model for generating new prompt suggestions in the APE loop.
# This needs to be a highly capable reasoning model.
PROMPT_GENERATION_MODEL = "gemini-2.5-pro"
PROMPT_GENERATION_CONFIG = {
    "temperature": 0.8,
}

# Model for evaluating a prompt against our audio test suite.
# This is the "target" model we are trying to optimize the prompt for.
# We use Flash for its speed and cost-effectiveness in the evaluation loop.
TARGET_MODEL_FOR_EVAL = "gemini-live-2.5-flash-preview-native-audio"
# TARGET_MODEL_FOR_EVAL = "gemini-live-2.5-flash"
TARGET_MODEL_CONFIG = {
    "temperature": 0.0,  # We want deterministic behavior during evaluation
}

# Add other configurations as needed