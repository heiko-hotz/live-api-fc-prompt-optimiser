import os
import json
import asyncio
import logging
import sys
from google import genai
from typing import List, Dict
from dotenv import load_dotenv

# Add the parent directory to Python path to find the configs module
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

load_dotenv()

# Import model configurations
from configs.model_configs import RESTATE_QUERIES_MODEL, RESTATE_QUERIES_CONFIG

# Configuration for file paths
INPUT_FILE = "configs/input_queries.json"
OUTPUT_FILE = "data_generation/output_queries.json" # This is a temporary intermediate file
NUM_RESTATEMENTS = 5
MAX_CONCURRENT_REQUESTS = 5

# Initialize the Gemini client (assuming environment is configured)
# Note: This might be better as a passed-in object in a larger system,
# but for this script, initializing here is fine.
client = genai.Client(
    vertexai=True,
    project=os.environ.get("GOOGLE_CLOUD_PROJECT"),
    location=os.environ.get("GOOGLE_CLOUD_LOCATION"),
)

# --- Logging Configuration ---
logger = logging.getLogger(__name__)

# Suppress verbose logging from external libraries
logging.getLogger('google_genai.live').setLevel(logging.WARNING)
logging.getLogger('google_genai').setLevel(logging.WARNING)

async def _restate_single_query(query: str) -> List[str]:
    """Helper function to restate one query using Gemini."""
    prompt = f"""Please restate the following user query for an AI voice assistant in {NUM_RESTATEMENTS} different ways.
    The goal is to create a diverse set of test cases.

    Guidelines:
    - The core intent must remain identical.
    - Use a mix of tones: direct, casual, polite, and formal.
    - Vary the sentence structure and vocabulary.
    - Return ONLY the restatements, each on a new line. Do not include numbering, bullets, or any other text.

    Original Query: "{query}"
    """
    try:
        response = await client.aio.models.generate_content(
            model=RESTATE_QUERIES_MODEL,
            contents=prompt,
            # generation_config=RESTATE_QUERIES_CONFIG
        )
        restatements = [line.strip() for line in response.text.split('\n') if line.strip()]
        return restatements[:NUM_RESTATEMENTS]
    except Exception as e:
        logger.error(f"Error generating restatements for query '{query}': {e}")
        # Return a fallback result if the API fails
        return [
            f"Could you tell me {query.lower()}",
            f"I'd like to know {query.lower()}",
            f"Please help me with: {query.lower()}",
            f"Can you provide information about {query.lower()}",
            f"I need to find out {query.lower()}"
        ]

async def _process_batch(query_batch: List[Dict]) -> List[Dict]:
    """Processes a batch of queries concurrently."""
    tasks = [asyncio.create_task(_restate_single_query(q['query'])) for q in query_batch]
    all_restatements = await asyncio.gather(*tasks)

    results = []
    for query_obj, restatements in zip(query_batch, all_restatements):
        result = {
            "original_query": query_obj['query'],
            "trigger_function": query_obj.get('trigger_function', False),
            "restatements": restatements
        }
        if result['trigger_function']:
            result['function_name'] = query_obj.get('function_name')
            result['function_args'] = query_obj.get('function_args')
        results.append(result)
    return results

async def generate_restated_queries():
    """
    Step 1: Generates diverse restatements for each base query.
    """
    try:
        with open(INPUT_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        results = []
        restater = QueryRestater()
        
        for i, entry in enumerate(data['queries']):
            query = entry['original_query']
            logger.info(f"Processing query {i+1}/{len(data['queries'])}: '{query}'")
            
            restatements = await restater.generate_restatements(query)
            
            results.append({
                "query_id": i + 1,
                "original_query": query,
                "expected_function": entry.get("expected_function"),
                "restatements": restatements
            })
        
        # Save to intermediate output file
        with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Generated restatements saved to: {OUTPUT_FILE}")
        
    except FileNotFoundError:
        logger.error(f"Input file '{INPUT_FILE}' not found.")
        raise
    except Exception as e:
        logger.error(f"Error in generate_restated_queries: {e}")
        raise