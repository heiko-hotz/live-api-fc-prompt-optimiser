import asyncio
import os
import sys
import shutil

# Import the core logic from our data_generation package
from data_generation.query_restater import generate_restated_queries
from data_generation.audio_generator import generate_audio_files

# --- Configuration ---
AUDIO_SUITE_DIR = "audio_test_suite"
FINAL_MAPPING_FILE = os.path.join(AUDIO_SUITE_DIR, "audio_mapping.json")

async def main():
    """
    Orchestrates the two-step process of preparing the audio test suite.
    1. Generates text restatements for base queries.
    2. Generates audio files for all text queries.
    """
    print("--- Automated Voice Test Suite Preparation ---")
    print("This script will generate a comprehensive set of audio files for testing.")

    # Safety check to prevent accidental (and costly) regeneration.
    if os.path.exists(FINAL_MAPPING_FILE):
        print(f"\n[WARNING] Test suite assets ('{FINAL_MAPPING_FILE}' and '{AUDIO_SUITE_DIR}/') seem to exist already.")
        response = input("Do you want to DELETE them and regenerate the entire suite? (y/n): ").strip().lower()
        if response != 'y':
            print("\nOperation cancelled. Using existing test suite.")
            sys.exit(0)
        print("Deleting existing assets...")
        if os.path.exists(AUDIO_SUITE_DIR):
            shutil.rmtree(AUDIO_SUITE_DIR)
        os.remove(FINAL_MAPPING_FILE)

    try:
        # --- Step 1: Generate Text Restatements ---
        print("\n[Step 1/2] Generating text restatements using the 'Restater' model...")
        await generate_restated_queries()
        print("✅ Text restatements generated successfully.")

        # --- Step 2: Generate Audio Files ---
        print("\n[Step 2/2] Generating audio files using Google Cloud Text-to-Speech...")
        print("(This may take a few minutes depending on the number of queries.)")
        await generate_audio_files()
        print(f"✅ Audio files generated in '{AUDIO_SUITE_DIR}/' directory.")
        print(f"✅ Master mapping file '{FINAL_MAPPING_FILE}' created.")

        # --- Final Verification ---
        if os.path.exists(FINAL_MAPPING_FILE):
            print("\n--- Test Suite Preparation Complete! ---")
            print("You are now ready to run the prompt optimization process with '02_run_optimization.py'.")
        else:
            print("\n[ERROR] A critical error occurred. The final mapping file was not created.")
            sys.exit(1)

    except Exception as e:
        print(f"\n[FATAL ERROR] An error occurred during test suite preparation: {e}")
        print("Please check the following:")
        print("1. Your cloud authentication (`gcloud auth application-default login`).")
        print("2. Your environment variables (`GOOGLE_CLOUD_PROJECT`, `GOOGLE_CLOUD_LOCATION`).")
        print("3. That the necessary Google Cloud APIs (Vertex AI, Text-to-Speech) are enabled.")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())