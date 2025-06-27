import asyncio
import os
import re
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Tuple, Dict, Any
import aiofiles
import backoff

# Add parent directory to path for imports when running standalone
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from google import genai
# Note: Other imports moved to where they're needed to avoid import issues in standalone mode

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Enable DEBUG logging specifically for this module
# logger.setLevel(logging.DEBUG)

class PromptOptimizer:
    """
    Automatic Prompt Engineering (APE) optimizer that iteratively improves 
    system prompts based on evaluation feedback.
    """
    
    def __init__(self, 
                 num_iterations: int, 
                 starting_prompt: str, 
                 evaluator, # Changed from AudioFunctionCallEvaluator to avoid import issues
                 metaprompt_path: str = "optimization/metaprompt_template.txt"):
        """
        Initialize the prompt optimizer.
        
        Args:
            num_iterations: Number of optimization iterations to run
            starting_prompt: Initial prompt to start optimization from
            evaluator: AudioFunctionCallEvaluator instance for prompt evaluation
            metaprompt_path: Path to the metaprompt template file
        """
        self.num_iterations = num_iterations
        self.starting_prompt = starting_prompt
        self.evaluator = evaluator
        self.metaprompt_path = metaprompt_path

        # Validate environment variables
        project_id = os.environ.get("GOOGLE_CLOUD_PROJECT")
        location = os.environ.get("GOOGLE_CLOUD_LOCATION")
        
        if not project_id or not location:
            raise ValueError("GOOGLE_CLOUD_PROJECT and GOOGLE_CLOUD_LOCATION environment variables must be set")
        
        # Setup run directory
        self.run_folder = self._create_run_folder()
        self.prompt_history_file = os.path.join(self.run_folder, 'prompt_history.txt')
        self.best_prompt = {"text": starting_prompt, "score": 0.0}
        
        # Import config when needed
        from configs.model_configs import PROMPT_GENERATION_MODEL
        
        logger.info(f"Optimizer initialized. Results will be saved in: {self.run_folder}")
        logger.info(f"Using generation model: {PROMPT_GENERATION_MODEL}")
        logger.info(f"Running {num_iterations} optimization iterations")

    def _create_run_folder(self) -> str:
        """Create a timestamped run folder for this optimization session."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_folder = os.path.join("runs", f'optimization_{timestamp}')
        os.makedirs(run_folder, exist_ok=True)
        return run_folder

    def _read_and_sort_history(self) -> str:
        """
        Reads the prompt history file, sorts it by accuracy, and returns a formatted string.
        
        Returns:
            Formatted string of prompts sorted by accuracy (best first)
        """
        try:
            with open(self.prompt_history_file, 'r') as f:
                content = f.read()
            
            # Extract prompt-score pairs using regex
            pattern = re.compile(
                r'<PROMPT>\n<PROMPT_TEXT>\n(.*?)\n</PROMPT_TEXT>\n<ACCURACY>\n(.*?)\n</ACCURACY>\n</PROMPT>', 
                re.DOTALL
            )
            matches = pattern.findall(content)
            
            if not matches:
                return "No history yet."
            
            # Sort by accuracy (descending)
            sorted_prompts = sorted(matches, key=lambda x: float(x[1]), reverse=True)
            
            # Reformat for metaprompt
            sorted_prompts_string = ""
            for prompt, accuracy in sorted_prompts:
                sorted_prompts_string += (
                    f"<PROMPT>\n<PROMPT_TEXT>\n{prompt.strip()}\n</PROMPT_TEXT>\n"
                    f"<ACCURACY>\n{float(accuracy):.2%}\n</ACCURACY>\n</PROMPT>\n\n"
                )
            return sorted_prompts_string
            
        except FileNotFoundError:
            return "No history yet."

    def _update_metaprompt(self) -> str:
        """
        Formats the metaprompt with the sorted history of prompts and their scores.
        
        Returns:
            Complete metaprompt ready for generation
        """
        sorted_history = self._read_and_sort_history()
        
        with open(self.metaprompt_path, 'r') as f:
            metaprompt_template = f.read()
        
        # Use replace() instead of format() to avoid issues with curly braces in prompt history
        return metaprompt_template.replace('{prompt_scores}', sorted_history)

    @backoff.on_exception(backoff.expo, Exception, max_tries=3, base=2)
    async def _generate_new_prompt(self, metaprompt: str) -> str:
        """
        Calls the generation model to get a new prompt candidate.
        
        Args:
            metaprompt: The complete metaprompt including history
            
        Returns:
            Generated prompt text
            
        Raises:
            ValueError: If no valid prompt is extracted from response
        """
        try:
            # Import config when needed
            from configs.model_configs import PROMPT_GENERATION_MODEL
            
            # Initialize client properly for Vertex AI (following query_restater.py pattern)
            project_id = os.environ.get("GOOGLE_CLOUD_PROJECT")
            location = os.environ.get("GOOGLE_CLOUD_LOCATION")
            
            logger.info("="*80)
            logger.info("📝 GENERATING NEW PROMPT WITH OPTIMIZATION MODEL")
            logger.info("="*80)
            logger.info(f"Model: {PROMPT_GENERATION_MODEL}")
            logger.info(f"Project: {project_id}, Location: {location}")
            logger.info(f"Metaprompt length: {len(metaprompt)} characters")
            logger.info("-"*80)
            
            logger.info("📋 METAPROMPT BEING SENT TO MODEL:")
            logger.info("-"*80)
            logger.info(metaprompt)
            logger.info("-"*80)
            
            client = genai.Client(
                vertexai=True,
                project=project_id,
                location=location,
            )
            
            # Make async call using the correct API pattern
            logger.info("⏳ Making API call to optimization model...")
            response = await client.aio.models.generate_content(
                model=PROMPT_GENERATION_MODEL,
                contents=metaprompt,
                # Note: generation_config commented out as in query_restater.py to avoid issues
                # generation_config=PROMPT_GENERATION_CONFIG
            )
            
            logger.info("✅ API call completed successfully")
            
            # Extract full response
            response_text = response.text if hasattr(response, 'text') else str(response)
            
            logger.info("="*80)
            logger.info("🤖 MODEL'S COMPLETE RESPONSE:")
            logger.info("="*80)
            logger.info(response_text)
            logger.info("="*80)
            
            # Extract prompt from response using regex
            match = re.search(r'\[\[(.*?)\]\]', response_text, re.DOTALL)
            if match:
                generated_prompt = match.group(1).strip()
                
                logger.info("🎯 EXTRACTED NEW PROMPT:")
                logger.info("-"*80)
                logger.info(generated_prompt)
                logger.info("-"*80)
                logger.info(f"✅ Generated prompt length: {len(generated_prompt)} characters")
                
                return generated_prompt
            else:
                logger.warning("⚠️  Could not extract a new prompt from the model's response. Retrying.")
                logger.info("❌ RESPONSE PARSING FAILED - No [[...]] format found")
                raise ValueError("No prompt found in [[...]] format")
                
        except Exception as e:
            logger.error(f"💥 Error generating new prompt: {e}")
            logger.error(f"Error type: {type(e).__name__}")
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
            raise

    async def _save_iteration_results(self, iteration: int, prompt: str, score: float, details: Dict[Any, Any]):
        """
        Save detailed results for a specific iteration.
        
        Args:
            iteration: Iteration number
            prompt: The prompt that was evaluated
            score: Accuracy score achieved
            details: Detailed evaluation results
        """
        iteration_folder = os.path.join(self.run_folder, f'iteration_{iteration}')
        os.makedirs(iteration_folder, exist_ok=True)
        
        # Save evaluation details
        with open(os.path.join(iteration_folder, 'evaluation_details.json'), 'w') as f:
            json.dump(details, f, indent=2)
        
        # Save prompt text
        with open(os.path.join(iteration_folder, 'prompt.txt'), 'w') as f:
            f.write(prompt)
        
        # Save summary
        summary = {
            "iteration": iteration,
            "score": score,
            "timestamp": datetime.now().isoformat(),
            "is_best": score > self.best_prompt["score"]
        }
        with open(os.path.join(iteration_folder, 'summary.json'), 'w') as f:
            json.dump(summary, f, indent=2)

    async def run(self) -> Tuple[str, float]:
        """
        Executes the main optimization loop.
        
        Returns:
            Tuple of (best_prompt_text, best_score)
        """
        logger.info("="*80)
        logger.info("STARTING PROMPT OPTIMIZATION")
        logger.info("="*80)
        logger.info(f"Initial prompt length: {len(self.starting_prompt)} characters")
        logger.info(f"Optimization iterations: {self.num_iterations}")
        logger.info(f"Results directory: {self.run_folder}")
        logger.info("-"*80)
        
        for i in range(self.num_iterations + 1):  # +1 to include the initial prompt
            logger.info("\n" + "="*60)
            logger.info(f"ITERATION {i}/{self.num_iterations}")
            logger.info("="*60)
            
            current_prompt = ""
            
            if i == 0:
                logger.info("Evaluating starting prompt...")
                current_prompt = self.starting_prompt
            else:
                logger.info("Generating new prompt candidate...")
                try:
                    metaprompt = self._update_metaprompt()
                    current_prompt = await self._generate_new_prompt(metaprompt)
                    logger.info(f"Generated prompt length: {len(current_prompt)} characters")
                except Exception as e:
                    logger.error(f"Failed to generate new prompt after retries: {e}. Skipping iteration.")
                    continue

            # Evaluate the current prompt
            logger.info(f"Evaluating prompt for iteration {i}...")
            logger.info("-"*40)
            
            try:
                # Create iteration-specific output directory for detailed results
                iteration_folder = os.path.join(self.run_folder, f'iteration_{i}')
                
                score, details = await self.evaluator.evaluate_prompt(
                    current_prompt, 
                    save_detailed_results=True, 
                    output_dir=iteration_folder
                )
                
                logger.info("-"*40)
                logger.info(f"Iteration {i} completed - Score: {score:.2%}")
                
                # Save iteration results
                await self._save_iteration_results(i, current_prompt, score, details)

                # Update history file
                async with aiofiles.open(self.prompt_history_file, 'a') as f:
                    await f.write(
                        f"<PROMPT>\n<PROMPT_TEXT>\n{current_prompt}\n</PROMPT_TEXT>\n"
                        f"<ACCURACY>\n{score}\n</ACCURACY>\n</PROMPT>\n\n"
                    )

                # Check if this is a new best
                if score > self.best_prompt["score"]:
                    improvement = score - self.best_prompt["score"]
                    logger.info(f"🎉 NEW BEST SCORE! {score:.2%} (improved by {improvement:.2%})")
                    self.best_prompt["score"] = score
                    self.best_prompt["text"] = current_prompt
                    
                    # Save best prompt separately
                    with open(os.path.join(self.run_folder, 'best_prompt.txt'), 'w') as f:
                        f.write(current_prompt)
                    with open(os.path.join(self.run_folder, 'best_prompt_info.json'), 'w') as f:
                        json.dump({
                            "score": score,
                            "iteration": i,
                            "timestamp": datetime.now().isoformat()
                        }, f, indent=2)
                else:
                    logger.info(f"Score {score:.2%} did not improve on best {self.best_prompt['score']:.2%}")
                    
            except Exception as e:
                logger.error(f"Error evaluating prompt in iteration {i}: {e}")
                continue

        # Final summary
        logger.info("\n" + "="*80)
        logger.info("OPTIMIZATION COMPLETE")
        logger.info("="*80)
        logger.info(f"Best prompt found with accuracy: {self.best_prompt['score']:.2%}")
        logger.info(f"All results saved in: {self.run_folder}")
        logger.info("\nBest prompt text:")
        logger.info("-"*40)
        logger.info(self.best_prompt['text'])
        logger.info("-"*40)
        
        return self.best_prompt['text'], self.best_prompt['score']

# Convenience function for easy usage
async def optimize_prompt(
    starting_prompt: str,
    audio_mapping_path: str,
    num_iterations: int = 5,
    max_concurrent_tests: int = 6
) -> Tuple[str, float]:
    """
    Convenience function to run prompt optimization.
    
    Args:
        starting_prompt: Initial prompt to optimize
        audio_mapping_path: Path to audio test mapping file
        num_iterations: Number of optimization iterations
        max_concurrent_tests: Max concurrent evaluations
        
    Returns:
        Tuple of (best_prompt, best_score)
    """
    # Import here to avoid import issues
    from evaluation.audio_fc_evaluator import AudioFunctionCallEvaluator
    
    evaluator = AudioFunctionCallEvaluator(
        audio_mapping_path=audio_mapping_path,
        max_concurrent_tests=max_concurrent_tests
    )
    
    optimizer = PromptOptimizer(
        num_iterations=num_iterations,
        starting_prompt=starting_prompt,
        evaluator=evaluator
    )
    
    return await optimizer.run()

# ==============================================================================
# --- STANDALONE DEBUG BLOCK ---
# ==============================================================================
if __name__ == '__main__':
    
    async def debug_prompt_generation():
        """Debug function to test prompt generation in isolation."""
        print("--- Debugging Prompt Generation ---")
        
        # Add parent directory to path for imports
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        
        # Create a minimal test setup
        starting_prompt = """
You are a helpful voice assistant for a service called Cymbal.
Your main job is to understand the user's intent and route their request to the correct tool.
- For general questions about Cymbal products or services, use the `get_chatbot_response` tool.
- If the user explicitly asks to speak to a human or a live agent, use the `escalate_to_human_agent` tool with the reason 'live-agent-request'.
- If the user sounds distressed, anxious, or mentions being in a vulnerable situation, use the `escalate_to_human_agent` tool with the reason 'vulnerable-user'.
        """.strip()
        
        # Create a dummy evaluator (we won't use it, just need it for initialization)
        audio_mapping_path = "audio_test_suite/audio_mapping.json"
        if not os.path.exists(audio_mapping_path):
            print(f"[ERROR] Test file not found: {audio_mapping_path}")
            print("Please run 'python 01_prepare_test_suite.py' first.")
            return
        
        try:
            # Import here to avoid issues when running standalone
            from evaluation.audio_fc_evaluator import AudioFunctionCallEvaluator
            evaluator = AudioFunctionCallEvaluator(audio_mapping_path, max_concurrent_tests=1)
            
            # Create optimizer
            optimizer = PromptOptimizer(
                num_iterations=1,
                starting_prompt=starting_prompt,
                evaluator=evaluator
            )
            
            print("\n1. Testing metaprompt generation...")
            metaprompt = optimizer._update_metaprompt()
            print(f"   ✅ Metaprompt created successfully ({len(metaprompt)} characters)")
            print(f"   Preview: {metaprompt[:300]}...")
            
            print("\n2. Testing API call for prompt generation...")
            try:
                generated_prompt = await optimizer._generate_new_prompt(metaprompt)
                print(f"   ✅ Prompt generated successfully!")
                print(f"   Generated prompt length: {len(generated_prompt)} characters")
                print(f"   Generated prompt preview: {generated_prompt[:200]}...")
            except Exception as e:
                print(f"   ❌ API call failed: {e}")
                print(f"   Error type: {type(e).__name__}")
                import traceback
                print(f"   Full traceback: {traceback.format_exc()}")
                
        except Exception as e:
            print(f"[FATAL ERROR] {e}")
            import traceback
            print(f"Full traceback: {traceback.format_exc()}")
    
    # Run the debug function
    asyncio.run(debug_prompt_generation())
