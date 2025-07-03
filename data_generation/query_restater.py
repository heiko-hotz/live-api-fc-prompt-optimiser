import os
import json
import asyncio
from google import genai
from typing import List, Dict
from dotenv import load_dotenv

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

async def _restate_single_query(query: str) -> List[str]:
    """Helper function to restate one query using Gemini."""
    prompt = f"""Please restate the following user query for a financial voice assistant in {NUM_RESTATEMENTS} different ways.
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
        print(f"Error generating restatements for query '{query}': {e}")
        return []

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
    """Processes queries from the input file and saves restatements to an intermediate file."""
    try:
        with open(INPUT_FILE, 'r') as f:
            input_data = json.load(f)

        queries = input_data['queries']
        all_results = []
        for i in range(0, len(queries), MAX_CONCURRENT_REQUESTS):
            batch = queries[i:i + MAX_CONCURRENT_REQUESTS]
            batch_results = await _process_batch(batch)
            all_results.extend(batch_results)

        with open(OUTPUT_FILE, 'w') as f:
            json.dump({"queries": all_results}, f, indent=2)

    except FileNotFoundError:
        print(f"Error: Input file '{INPUT_FILE}' not found.")
        raise
    except Exception as e:
        print(f"Error in generate_restated_queries: {e}")
        raise