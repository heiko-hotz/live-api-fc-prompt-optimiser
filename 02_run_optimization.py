#!/usr/bin/env python3
"""
Main script to run the prompt optimization process.

This script uses the Automatic Prompt Engineering (APE) approach to iteratively
improve system prompts for the Cymbal voice assistant's function calling accuracy.

Usage:
    python 02_run_optimization.py

Requirements:
    - Run 01_prepare_test_suite.py first to generate the audio test suite
    - Set GOOGLE_CLOUD_PROJECT and GOOGLE_CLOUD_LOCATION environment variables
    - Authenticate with Google Cloud: gcloud auth application-default login
"""

import asyncio
import logging
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Configure logging for the main script
logging.basicConfig(
    level=logging.INFO,  # Changed to DEBUG to see detailed output
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('optimization.log')
    ]
)
logger = logging.getLogger(__name__)

def check_prerequisites():
    """Check that all prerequisites are met before starting optimization."""
    
    # Check for test suite
    audio_mapping_path = "audio_test_suite/audio_mapping.json"
    if not Path(audio_mapping_path).exists():
        logger.error(f"Audio test suite not found at: {audio_mapping_path}")
        logger.error("Please run 'python 01_prepare_test_suite.py' first to generate the test assets.")
        return False
    
    # Check environment variables
    if not os.environ.get("GOOGLE_CLOUD_PROJECT"):
        logger.error("GOOGLE_CLOUD_PROJECT environment variable not set")
        logger.error("Please set it with: export GOOGLE_CLOUD_PROJECT=your-project-id")
        return False
    
    if not os.environ.get("GOOGLE_CLOUD_LOCATION"):
        logger.error("GOOGLE_CLOUD_LOCATION environment variable not set")
        logger.error("Please set it with: export GOOGLE_CLOUD_LOCATION=us-central1")
        return False
    
    return True

async def main():
    """Main optimization execution function."""
    
    logger.info("="*80)
    logger.info("CYMBAL VOICE ASSISTANT PROMPT OPTIMIZATION")
    logger.info("="*80)
    logger.info("This will run an Automatic Prompt Engineering (APE) optimization")
    logger.info("to improve the function calling accuracy of the Cymbal voice assistant.")
    logger.info("-"*80)
    
    # Check prerequisites
    if not check_prerequisites():
        logger.error("Prerequisites not met. Exiting.")
        sys.exit(1)
    
    # Load the comprehensive starting prompt from the initial system instruction file
    initial_instruction_path = "initial-system-instruction.txt"
    try:
        with open(initial_instruction_path, 'r', encoding='utf-8') as f:
            starting_prompt = f.read().strip()
        logger.info(f"Loaded comprehensive starting prompt from: {initial_instruction_path}")
    except FileNotFoundError:
        logger.error(f"Initial system instruction file not found: {initial_instruction_path}")
        logger.error("Please ensure the file exists or create it with your comprehensive system prompt.")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error loading initial system instruction: {e}")
        sys.exit(1)
    
    # Configuration
    audio_mapping_path = "audio_test_suite/audio_mapping.json"
    num_iterations = 10  # Number of optimization iterations
    max_concurrent_tests = 2  # Batch size for evaluation
    early_stopping_threshold = 1.0  # Stop if accuracy exceeds 90%
    
    logger.info(f"Configuration:")
    logger.info(f"  - Audio mapping: {audio_mapping_path}")
    logger.info(f"  - Optimization iterations: {num_iterations}")
    logger.info(f"  - Max concurrent tests: {max_concurrent_tests}")
    logger.info(f"  - Early stopping threshold: {early_stopping_threshold:.2%}")
    logger.info(f"  - Starting prompt length: {len(starting_prompt)} characters")
    logger.info("-"*80)
    
    try:
        # Import here to avoid import errors if prerequisites aren't met
        from optimization.prompt_optimizer import optimize_prompt
        
        # Run the optimization
        logger.info("Starting optimization process...")
        best_prompt, best_score = await optimize_prompt(
            starting_prompt=starting_prompt,
            audio_mapping_path=audio_mapping_path,
            num_iterations=num_iterations,
            max_concurrent_tests=max_concurrent_tests,
            early_stopping_threshold=early_stopping_threshold
        )
        
        # Final results
        logger.info("="*80)
        logger.info("🎉 OPTIMIZATION COMPLETED SUCCESSFULLY! 🎉")
        logger.info("="*80)
        logger.info(f"Best accuracy achieved: {best_score:.2%}")
        logger.info(f"Improvement from baseline: Coming in next evaluation...")
        logger.info("\nOptimized prompt:")
        logger.info("="*40)
        logger.info(best_prompt)
        logger.info("="*40)
        logger.info("\nAll detailed results have been saved in the runs/ directory.")
        logger.info("Check the latest 'optimization_*' folder for complete results.")
        
    except KeyboardInterrupt:
        logger.info("\n\nOptimization interrupted by user. Partial results may be available in runs/ directory.")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Optimization failed with error: {e}")
        logger.error("Please check your configuration and try again.")
        raise

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)
