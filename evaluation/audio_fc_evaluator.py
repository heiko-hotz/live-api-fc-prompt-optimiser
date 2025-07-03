import asyncio
import json
import os
import io
import sys
import librosa
import soundfile as sf
import numpy as np
import csv
import logging
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime
from dotenv import load_dotenv

# Add the parent directory to Python path to find the configs module
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Load environment variables from .env file
load_dotenv()

from google import genai
from google.genai.types import (
    LiveConnectConfig, SpeechConfig, VoiceConfig, PrebuiltVoiceConfig,
    Blob, FunctionResponse
)

# Import model configurations from our central config file
from configs.model_configs import TARGET_MODEL_FOR_EVAL, TARGET_MODEL_CONFIG

# --- Logging Configuration ---
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Suppress verbose logging from external libraries
logging.getLogger('google_genai.live').setLevel(logging.WARNING)
logging.getLogger('google_genai').setLevel(logging.WARNING)

# --- Tool Schemas (preserved from original project) ---
GET_INFORMATION_SCHEMA = {
    "name": "get_information",
    "description": "Retrieves information or answers general questions and knowledge queries.",
    "parameters": {"type": "OBJECT", "properties": {"query": {"type": "STRING", "description": "The user's question or information request."}}, "required": ["query"]}
}
ESCALATE_TO_SUPPORT_SCHEMA = {
    "name": "escalate_to_support",
    "description": "Escalates the conversation to human support when required.",
    "parameters": {"type": "OBJECT", "properties": {"reason": {"type": "STRING", "description": "Must be one of 'human-request' or 'vulnerable-user'."}}, "required": ["reason"]}
}
TOOL_SCHEMAS = [GET_INFORMATION_SCHEMA, ESCALATE_TO_SUPPORT_SCHEMA]

# --- Constants ---
INPUT_RATE = 16000   # Required for Gemini
TEST_TIMEOUT = 20.0  # Timeout for individual tests

class AudioFunctionCallEvaluator:
    """
    Evaluates a system prompt against a suite of audio test cases
    to determine its function-calling accuracy.
    """
    def __init__(self, audio_mapping_path: str, max_concurrent_tests: int = 10):
        self.test_cases = self._load_and_flatten_test_cases(audio_mapping_path)
        self.max_concurrent_tests = max_concurrent_tests
        self.test_counter = 0
        self.total_tests = len(self.test_cases)
        
        # Initialize client with proper configuration
        project_id = os.environ.get("GOOGLE_CLOUD_PROJECT")
        location = os.environ.get("GOOGLE_CLOUD_LOCATION")
        
        if not project_id or not location:
            raise ValueError("GOOGLE_CLOUD_PROJECT and GOOGLE_CLOUD_LOCATION environment variables must be set")
            
        self.client = genai.Client(vertexai=True, project=project_id, location=location)
        logger.debug(f"Evaluator initialized with {len(self.test_cases)} test cases.")

    def _load_and_flatten_test_cases(self, path: str) -> List[Dict[str, Any]]:
        """Loads the audio mapping and flattens it into a simple list of tests."""
        try:
            with open(path, 'r') as f:
                data = json.load(f)
        except FileNotFoundError:
            logger.error(f"Audio mapping file not found at: {path}")
            raise
        
        flat_list = []
        for query_data in data["audio_mappings"]:
            original_query = query_data["original_query"]
            audio_files = query_data["audio_files"]
            
            # Process original file
            original_file = audio_files["original"]
            flat_list.append({
                "file_path": original_file["path"], 
                "expected_function": original_file["expected_function"],
                "query": original_query,
                "file_type": "original",
                "voice": original_file.get("voice", "unknown"),
                "restatement_text": None
            })
            
            # Process restatement files
            for i, restatement in enumerate(audio_files["restatements"]):
                flat_list.append({
                    "file_path": restatement["path"], 
                    "expected_function": restatement["expected_function"],
                    "query": original_query,
                    "file_type": f"restatement_{i+1}",
                    "voice": restatement.get("voice", "unknown"),
                    "restatement_text": restatement.get("text", "")
                })
        
        return flat_list

    def _load_audio_bytes(self, input_file: str) -> bytes:
        """Load and convert audio file to required format."""
        if not Path(input_file).exists():
            raise FileNotFoundError(f"Audio file not found: {input_file}")
            
        y, _ = librosa.load(input_file, sr=INPUT_RATE)
        buffer = io.BytesIO()
        sf.write(buffer, y, INPUT_RATE, format="RAW", subtype="PCM_16")
        buffer.seek(0)
        return buffer.read()

    def _compare_function_calls(self, actual_calls: List[Dict], expected_call: Dict) -> Dict:
        """Compares actual vs. expected function calls and returns a result dict."""
        if expected_call is None:
            # No function call expected
            if len(actual_calls) == 0:
                return {"status": "PASS", "message": "No function call expected and none made"}
            else:
                return {
                    "status": "FAIL", 
                    "message": f"No function call expected but {len(actual_calls)} made",
                    "actual_calls": actual_calls
                }
        else:
            # Function call expected
            if len(actual_calls) == 0:
                return {
                    "status": "FAIL",
                    "message": "Function call expected but none made",
                    "expected": expected_call
                }
            elif len(actual_calls) > 1:
                return {
                    "status": "FAIL",
                    "message": f"Expected 1 function call but {len(actual_calls)} made",
                    "expected": expected_call,
                    "actual_calls": actual_calls
                }
            else:
                actual_call = actual_calls[0]
                if actual_call["name"] == expected_call["name"]:
                    return {
                        "status": "PASS",
                        "message": "Correct function called",
                        "expected": expected_call,
                        "actual": actual_call
                    }
                else:
                    return {
                        "status": "FAIL",
                        "message": f"Wrong function called. Expected {expected_call['name']}, got {actual_call['name']}",
                        "expected": expected_call,
                        "actual": actual_call
                    }

    async def _process_single_test(self, test_case: Dict, system_prompt: str, semaphore: asyncio.Semaphore) -> Dict:
        """Processes a single audio file and returns the evaluation result."""
        async with semaphore:
            # Increment and log progress
            self.test_counter += 1
            progress = f"[{self.test_counter}/{self.total_tests}]"
            
            try:
                logger.debug(f"{progress} Testing: {test_case['file_path']}")
                
                # Apply timeout to the entire test processing
                result = await asyncio.wait_for(
                    self._process_test_with_api(test_case, system_prompt),
                    timeout=TEST_TIMEOUT
                )
                
                # Log result with color-coded status
                status = result["comparison"]["status"]
                if status == "PASS":
                    logger.info(f"{progress} ✅ PASS: {test_case['file_path']} - {result['comparison']['message']}")
                else:
                    logger.info(f"{progress} ❌ FAIL: {test_case['file_path']} - {result['comparison']['message']}")
                
                return result
                
            except asyncio.TimeoutError:
                logger.error(f"{progress} ⏰ TIMEOUT: {test_case['file_path']} - Test exceeded {TEST_TIMEOUT} second timeout")
                return {
                    **test_case, 
                    "comparison": {
                        "status": "TIMEOUT", 
                        "message": f"Test exceeded {TEST_TIMEOUT} second timeout - likely hanging session"
                    }
                }
            except Exception as e:
                logger.error(f"{progress} 💥 ERROR: {test_case['file_path']} - {str(e)}")
                return {**test_case, "comparison": {"status": "ERROR", "message": str(e)}}

    async def _process_test_with_api(self, test_case: Dict, system_prompt: str) -> Dict:
        """Process a single test case with the Gemini Live API."""
        config = LiveConnectConfig(
            system_instruction=system_prompt,
            tools=[{"function_declarations": TOOL_SCHEMAS}]
        )
        
        actual_calls_made = []
        
        async with self.client.aio.live.connect(model=TARGET_MODEL_FOR_EVAL, config=config) as session:
            # Load and send audio
            audio_bytes = self._load_audio_bytes(test_case["file_path"])
            await session.send_realtime_input(
                media=Blob(data=audio_bytes, mime_type=f"audio/pcm;rate={INPUT_RATE}")
            )
            
            # Process responses
            async for response in session.receive():
                if response.tool_call:
                    # Handle function calls
                    for fc in response.tool_call.function_calls:
                        actual_calls_made.append({"name": fc.name, "args": dict(fc.args)})
                        
                elif (response.server_content and 
                      response.server_content.model_turn and 
                      response.server_content.model_turn.parts):
                    # Handle other response content (audio, text, etc.)
                    for part in response.server_content.model_turn.parts:
                        if part.inline_data:
                            # We could save audio here if needed, but for evaluation we just need function calls
                            pass
        
        # Compare results
        comparison = self._compare_function_calls(actual_calls_made, test_case["expected_function"])
        return {**test_case, "comparison": comparison}

    def _save_results_as_csv(self, test_results: Dict, csv_file: str):
        """Save test results as a CSV file for human readability."""
        fieldnames = [
            'identifier',
            'query',
            'voice',
            'expected_function_call',
            'actual_function_call',
            'actual_FC_parameters',
            'test_status'
        ]
        
        with open(csv_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            
            # Track query numbers for identifier generation
            query_tracker = {}
            query_counter = 1
            
            for result in test_results['results']:
                comparison = result.get('comparison', {})
                
                # Generate identifier
                original_query = result.get('query', '')
                if original_query not in query_tracker:
                    query_tracker[original_query] = query_counter
                    query_counter += 1
                
                query_num = query_tracker[original_query]
                file_type = result.get('file_type', '')
                
                if file_type == 'original':
                    identifier = f"query_{query_num}_restate_0"
                elif file_type.startswith('restatement_'):
                    restate_num = file_type.split('_')[1]
                    identifier = f"query_{query_num}_restate_{restate_num}"
                else:
                    identifier = f"query_{query_num}_unknown"
                
                # Determine the actual query text (original or restatement)
                if result.get('restatement_text'):
                    query_text = result.get('restatement_text', '')
                else:
                    query_text = result.get('query', '')
                
                # Extract expected function name
                expected_func_name = ""
                if 'expected' in comparison:
                    expected_func_name = comparison['expected'].get('name', '')
                elif result.get('expected_function'):
                    expected_func_name = result['expected_function'].get('name', '')
                
                # Extract actual function details
                actual_func_name = ""
                actual_func_args = ""
                if 'actual' in comparison:
                    actual_func_name = comparison['actual'].get('name', '')
                    actual_func_args = json.dumps(comparison['actual'].get('args', {}))
                elif 'actual_calls' in comparison and comparison['actual_calls']:
                    # Handle case where multiple calls were made
                    calls = comparison['actual_calls']
                    actual_func_name = '; '.join([call.get('name', '') for call in calls])
                    actual_func_args = '; '.join([json.dumps(call.get('args', {})) for call in calls])
                
                row = {
                    'identifier': identifier,
                    'query': query_text,
                    'voice': result.get('voice', ''),
                    'expected_function_call': expected_func_name,
                    'actual_function_call': actual_func_name,
                    'actual_FC_parameters': actual_func_args,
                    'test_status': comparison.get('status', '')
                }
                writer.writerow(row)

    async def evaluate_prompt(self, system_prompt: str, save_detailed_results: bool = False, output_dir: str = None) -> (float, Dict):
        """The main public method. Evaluates a system prompt and returns its performance score."""
        start_time = datetime.now()
        
        # Reset counter for this evaluation
        self.test_counter = 0
        
        logger.info("="*60)
        logger.info("STARTING FUNCTION CALL EVALUATION")
        logger.info("="*60)
        logger.debug(f"Model: {TARGET_MODEL_FOR_EVAL}")
        logger.debug(f"Total tests: {self.total_tests}")
        logger.debug(f"Max concurrent tests: {self.max_concurrent_tests}")
        logger.debug(f"Test timeout: {TEST_TIMEOUT}s per test")
        logger.info("-"*60)
        
        semaphore = asyncio.Semaphore(self.max_concurrent_tests)
        tasks = [self._process_single_test(tc, system_prompt, semaphore) for tc in self.test_cases]
        
        # Execute all tests concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        end_time = datetime.now()
        execution_time = (end_time - start_time).total_seconds()

        # Process results and handle exceptions
        processed_results = []
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Task failed with exception: {result}")
                processed_results.append({
                    "comparison": {"status": "ERROR", "message": f"Task exception: {result}"}
                })
            else:
                processed_results.append(result)

        # Count results by status
        passed_count = sum(1 for res in processed_results if res.get("comparison", {}).get("status") == "PASS")
        failed_count = sum(1 for res in processed_results if res.get("comparison", {}).get("status") == "FAIL")
        timeout_count = sum(1 for res in processed_results if res.get("comparison", {}).get("status") == "TIMEOUT")
        error_count = sum(1 for res in processed_results if res.get("comparison", {}).get("status") == "ERROR")
        total_tests = len(self.test_cases)
        
        accuracy = passed_count / total_tests if total_tests > 0 else 0.0

        detailed_results = {
            "test_run_timestamp": start_time.isoformat(),
            "model_name": TARGET_MODEL_FOR_EVAL,
            "max_concurrent_tests": self.max_concurrent_tests,
            "prompt_text": system_prompt,
            "score": accuracy,
            "total_tests": total_tests,
            "passed_tests": passed_count,
            "failed_tests": failed_count,
            "timeout_tests": timeout_count,
            "error_tests": error_count,
            "execution_time_seconds": execution_time,
            "average_time_per_test": execution_time / total_tests if total_tests > 0 else 0,
            "summary": {
                "total_tests": total_tests, 
                "passed": passed_count, 
                "failed": failed_count,
                "timeout": timeout_count,
                "error": error_count
            },
            "results": processed_results
        }
        
        # Save detailed results if requested
        if save_detailed_results and output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            # Save JSON
            json_file = output_path / "evaluation_results.json"
            with open(json_file, 'w') as f:
                json.dump(detailed_results, f, indent=2)
            
            # Save CSV
            csv_file = output_path / "evaluation_results.csv"
            self._save_results_as_csv(detailed_results, str(csv_file))
            
            logger.debug(f"Detailed results saved to: {json_file}")
            logger.debug(f"CSV results saved to: {csv_file}")
        
        # Print comprehensive summary
        logger.info("="*60)
        logger.info("EVALUATION COMPLETE")
        logger.info("="*60)
        logger.info(f"Overall Accuracy Score: {accuracy:.2%}")
        logger.info(f"Success rate: {passed_count}/{total_tests} tests passed ({accuracy:.1%})")
        logger.debug(f"Total execution time: {execution_time:.2f} seconds")
        logger.debug(f"Average time per test: {detailed_results['average_time_per_test']:.2f} seconds")
        
        if timeout_count > 0:
            logger.warning(f"⏰ Timeout tests: {timeout_count}")
        if error_count > 0:
            logger.error(f"💥 Error tests: {error_count}")
        
        # Show detailed breakdown of failures
        failed_tests = [r for r in processed_results if r.get('comparison', {}).get('status') == 'FAIL']
        timeout_tests = [r for r in processed_results if r.get('comparison', {}).get('status') == 'TIMEOUT']
        error_tests = [r for r in processed_results if r.get('comparison', {}).get('status') == 'ERROR']
        
        if failed_tests:
            logger.info(f"\n❌ FAILED TESTS ({len(failed_tests)}):")
            for test in failed_tests:
                logger.info(f"  - {test.get('file_path', 'unknown')}: {test['comparison']['message']}")
        
        if timeout_tests:
            logger.debug(f"\n⏰ TIMEOUT TESTS ({len(timeout_tests)}):")
            for test in timeout_tests:
                logger.debug(f"  - {test.get('file_path', 'unknown')}: {test['comparison']['message']}")
        
        if error_tests:
            logger.debug(f"\n💥 ERROR TESTS ({len(error_tests)}):")
            for test in error_tests:
                logger.debug(f"  - {test.get('file_path', 'unknown')}: {test['comparison']['message']}")
        
        logger.info("="*60)
        
        return accuracy, detailed_results

# ==============================================================================
# --- STANDALONE TEST BLOCK ---
# ==============================================================================
if __name__ == '__main__':
    
    async def run_standalone_test():
        """Function to execute a single evaluation test."""
        print("--- Running AudioFunctionCallEvaluator in Standalone Test Mode ---")

        # 1. Check if the required test suite assets exist
        audio_mapping_path = os.path.join("audio_test_suite", "audio_mapping.json")
        if not os.path.exists(audio_mapping_path):
            print(f"\n[ERROR] Required test file not found: {audio_mapping_path}")
            print("Please run 'python 01_prepare_test_suite.py' first to generate the test assets.")
            sys.exit(1)
            
        # 2. Define a simple, human-written baseline prompt to test
        baseline_prompt = """
You are a helpful AI voice assistant.
Your main job is to understand the user's intent and route their request to the correct function.
- For general questions about topics, information requests, or knowledge queries, use the `get_information` function.
- If the user explicitly asks to speak to a human, get help from a person, or requests human assistance, use the `escalate_to_support` function with the reason 'human-request'.
- If the user sounds distressed, anxious, mentions feeling overwhelmed, or describes a difficult situation, use the `escalate_to_support` function with the reason 'vulnerable-user'.
        """.strip()

        # 3. Initialize and run the evaluator
        try:
            print(f"\nInitializing evaluator with test suite from: {audio_mapping_path}")
            evaluator = AudioFunctionCallEvaluator(
                audio_mapping_path=audio_mapping_path, 
                max_concurrent_tests=6  # Set your desired batch size here
            )
            
            print("\nEvaluating the baseline prompt...")
            print("-" * 50)
            print(baseline_prompt)
            print("-" * 50)
            
            # Create runs directory for detailed results
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            runs_dir = Path("runs")
            runs_dir.mkdir(exist_ok=True)
            run_dir = runs_dir / f"standalone_eval_{timestamp}"
            
            score, details = await evaluator.evaluate_prompt(
                baseline_prompt, 
                save_detailed_results=True, 
                output_dir=str(run_dir)
            )
            
            # 4. Print results and save them for inspection
            print("\n--- Standalone Evaluation Results ---")
            print(f"Overall Accuracy Score: {score:.2%}")
            print(f"Success rate: {details['passed_tests']}/{details['total_tests']} tests passed")
            
            # Print failed tests
            failed_tests = [r for r in details['results'] if r['comparison']['status'] == 'FAIL']
            timeout_tests = [r for r in details['results'] if r['comparison']['status'] == 'TIMEOUT']
            error_tests = [r for r in details['results'] if r['comparison']['status'] == 'ERROR']
            
            if failed_tests:
                print(f"\nFAILED TESTS ({len(failed_tests)}):")
                for test in failed_tests:
                    print(f"  - {test['file_path']}: {test['comparison']['message']}")
            
            if timeout_tests:
                print(f"\nTIMEOUT TESTS ({len(timeout_tests)}):")
                for test in timeout_tests:
                    print(f"  - {test['file_path']}: {test['comparison']['message']}")
            
            if error_tests:
                print(f"\nERROR TESTS ({len(error_tests)}):")
                for test in error_tests:
                    print(f"  - {test['file_path']}: {test['comparison']['message']}")
            
            # Also save a simple results file in the root for quick inspection
            output_file = "standalone_eval_results.json"
            with open(output_file, 'w') as f:
                json.dump(details, f, indent=2)
            print(f"\nDetailed results also saved to: {output_file}")

        except Exception as e:
            print(f"\n[FATAL ERROR] An unexpected error occurred during the standalone test: {e}")
            print("Please check your cloud authentication and API permissions.")
            sys.exit(1)

    # Run the async test function
    # Ensure you have set GOOGLE_CLOUD_PROJECT and authenticated with `gcloud auth application-default login`
    asyncio.run(run_standalone_test())