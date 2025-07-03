import asyncio
import os
import sys
import shutil
import logging

# Import the core logic from our data_generation package
from data_generation.query_restater import generate_restated_queries
from data_generation.audio_generator import generate_audio_files

# --- Configuration ---
AUDIO_SUITE_DIR = "audio_test_suite"
FINAL_MAPPING_FILE = os.path.join(AUDIO_SUITE_DIR, "audio_mapping.json")

# --- Logging Configuration ---
# Create separate handlers for console (INFO) and file (DEBUG)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

file_handler = logging.FileHandler('test_preparation.log')
file_handler.setLevel(logging.DEBUG)

# Configure the root logger with both handlers
logging.basicConfig(
    level=logging.DEBUG,  # Set root logger to DEBUG to capture everything
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[console_handler, file_handler]
)
logger = logging.getLogger(__name__)

# Suppress verbose logging from external libraries
logging.getLogger('google_genai.live').setLevel(logging.WARNING)
logging.getLogger('google_genai').setLevel(logging.WARNING)

async def main():
    """
    Orchestrates the two-step process of preparing the audio test suite.
    1. Generates text restatements for base queries.
    2. Generates audio files for all text queries.
    """
    logger.info("--- Automated Voice Test Suite Preparation ---")
    logger.info("This script will generate a comprehensive set of audio files for testing.")

    # Safety check to prevent accidental (and costly) regeneration.
    if os.path.exists(FINAL_MAPPING_FILE):
        logger.warning(f"Test suite assets ('{FINAL_MAPPING_FILE}' and '{AUDIO_SUITE_DIR}/') seem to exist already.")
        response = input("Do you want to DELETE them and regenerate the entire suite? (y/n): ").strip().lower()
        if response != 'y':
            logger.info("Operation cancelled. Using existing test suite.")
            sys.exit(0)
        logger.info("Deleting existing assets...")
        if os.path.exists(AUDIO_SUITE_DIR):
            shutil.rmtree(AUDIO_SUITE_DIR)
        # No need to remove FINAL_MAPPING_FILE separately since it's inside AUDIO_SUITE_DIR

    try:
        # --- Step 1: Generate Text Restatements ---
        logger.info("[Step 1/2] Generating text restatements using the 'Restater' model...")
        await generate_restated_queries()
        logger.info("✅ Text restatements generated successfully.")

        # --- Step 2: Generate Audio Files ---
        logger.info("[Step 2/2] Generating audio files using Google Cloud Text-to-Speech...")
        logger.info("(This may take a few minutes depending on the number of queries.)")
        await generate_audio_files()
        logger.info(f"✅ Audio files generated in '{AUDIO_SUITE_DIR}/' directory.")
        logger.info(f"✅ Master mapping file '{FINAL_MAPPING_FILE}' created.")

        # --- Final Verification ---
        if os.path.exists(FINAL_MAPPING_FILE):
            logger.info("--- Test Suite Preparation Complete! ---")
            logger.info("You are now ready to run the prompt optimization process with '02_run_optimization.py'.")
        else:
            logger.error("A critical error occurred. The final mapping file was not created.")
            sys.exit(1)

    except Exception as e:
        logger.error(f"An error occurred during test suite preparation: {e}")
        logger.error("Please check the following:")
        logger.error("1. Your cloud authentication (`gcloud auth application-default login`).")
        logger.error("2. Your environment variables (`GOOGLE_CLOUD_PROJECT`, `GOOGLE_CLOUD_LOCATION`).")
        logger.error("3. That the necessary Google Cloud APIs (Vertex AI, Text-to-Speech) are enabled.")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())