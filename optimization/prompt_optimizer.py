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
                 metaprompt_path: str = "optimization/metaprompt_template.txt",
                 early_stopping_threshold: float = None):
        """
        Initialize the prompt optimizer.
        
        Args:
            num_iterations: Number of optimization iterations to run
            starting_prompt: Initial prompt to start optimization from
            evaluator: AudioFunctionCallEvaluator instance for prompt evaluation
            metaprompt_path: Path to the metaprompt template file
            early_stopping_threshold: If set, stop optimization when accuracy exceeds this threshold (0.0-1.0)
        """
        self.num_iterations = num_iterations
        self.starting_prompt = starting_prompt
        self.evaluator = evaluator
        self.metaprompt_path = metaprompt_path
        self.early_stopping_threshold = early_stopping_threshold

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
        if early_stopping_threshold is not None:
            logger.info(f"Early stopping enabled: will stop if accuracy exceeds {early_stopping_threshold:.2%}")

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
            
            # Try enhanced format first
            enhanced_pattern = re.compile(
                r'<PROMPT>\n<PROMPT_TEXT>\n(.*?)\n</PROMPT_TEXT>\n'
                r'<OVERALL_ACCURACY>\n(.*?)\n</OVERALL_ACCURACY>\n'
                r'<QUERY_BREAKDOWN>\n(.*?)\n</QUERY_BREAKDOWN>\n'
                r'<FAILING_EXAMPLES>\n(.*?)\n</FAILING_EXAMPLES>\n</PROMPT>', 
                re.DOTALL
            )
            enhanced_matches = enhanced_pattern.findall(content)
            
            if enhanced_matches:
                # Sort by overall accuracy (descending)
                sorted_prompts = sorted(enhanced_matches, key=lambda x: float(x[1]), reverse=True)
                
                # Format enhanced data for metaprompt
                formatted_history = ""
                for prompt, accuracy, breakdown, examples in sorted_prompts:
                    formatted_history += (
                        f"<PROMPT>\n<PROMPT_TEXT>\n{prompt.strip()}\n</PROMPT_TEXT>\n"
                        f"<OVERALL_ACCURACY>\n{float(accuracy):.2%}\n</OVERALL_ACCURACY>\n"
                        f"<QUERY_PERFORMANCE>\n{breakdown.strip()}\n</QUERY_PERFORMANCE>\n"
                        f"<CRITICAL_FAILURES>\n{examples.strip()}\n</CRITICAL_FAILURES>\n"
                        f"</PROMPT>\n\n"
                    )
                return formatted_history
            
            # Fallback to old format for backward compatibility
            old_pattern = re.compile(
                r'<PROMPT>\n<PROMPT_TEXT>\n(.*?)\n</PROMPT_TEXT>\n<ACCURACY>\n(.*?)\n</ACCURACY>\n</PROMPT>', 
                re.DOTALL
            )
            old_matches = old_pattern.findall(content)
            
            if old_matches:
                # Sort by accuracy (descending)
                sorted_prompts = sorted(old_matches, key=lambda x: float(x[1]), reverse=True)
                
                # Reformat for metaprompt (old format)
                sorted_prompts_string = ""
                for prompt, accuracy in sorted_prompts:
                    sorted_prompts_string += (
                        f"<PROMPT>\n<PROMPT_TEXT>\n{prompt.strip()}\n</PROMPT_TEXT>\n"
                        f"<OVERALL_ACCURACY>\n{float(accuracy):.2%}\n</OVERALL_ACCURACY>\n"
                        f"<QUERY_PERFORMANCE>\nNo query-level data available (old format)\n</QUERY_PERFORMANCE>\n"
                        f"<CRITICAL_FAILURES>\nNo failure data available (old format)\n</CRITICAL_FAILURES>\n"
                        f"</PROMPT>\n\n"
                    )
                return sorted_prompts_string
            
            return "No history yet."
            
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

    def _calculate_query_breakdown(self, details: Dict) -> str:
        """Extract query-level performance from evaluation details."""
        # Group results by original query
        query_results = {}
        for result in details['results']:
            query = result['query']
            if query not in query_results:
                query_results[query] = {'passed': 0, 'total': 0}
            
            query_results[query]['total'] += 1
            if result['comparison']['status'] == 'PASS':
                query_results[query]['passed'] += 1
        
        # Format breakdown
        breakdown_lines = []
        for query, stats in query_results.items():
            accuracy = stats['passed'] / stats['total'] if stats['total'] > 0 else 0
            status = ""
            if accuracy < 0.6:
                status = " - CRITICAL"
            elif accuracy < 0.8:
                status = " - WEAK"
            
            breakdown_lines.append(
                f"{query}: {stats['passed']}/{stats['total']} ({accuracy:.0%}){status}"
            )
        
        return "\n".join(breakdown_lines)

    def _extract_failing_examples(self, details: Dict) -> str:
        """Extract specific failing examples for the metaprompt."""
        failing_examples = []
        
        for result in details['results']:
            if result['comparison']['status'] == 'FAIL':
                query_text = result.get('restatement_text') or result['query']
                expected = result['comparison'].get('expected', {})
                actual = result['comparison'].get('actual', {})
                
                expected_desc = "None"
                if expected:
                    expected_desc = f"{expected.get('name', 'None')}"
                    if expected.get('args'):
                        # Format args nicely
                        args_str = ", ".join([f"{k}='{v}'" for k, v in expected['args'].items()])
                        expected_desc += f"({args_str})"
                
                actual_desc = actual.get('name', 'None') if actual else 'None'
                
                failing_examples.append(
                    f'"{query_text}" → Expected: {expected_desc}, Got: {actual_desc}'
                )
        
        # Return top 3 most informative failures
        return "\n".join(failing_examples[:3]) if failing_examples else "No specific failures to report"

    def _parse_query_breakdown_for_history(self, breakdown_text: str) -> Dict[str, Dict[str, any]]:
        """Parse query breakdown text into structured data for score history."""
        queries = {}
        
        for line in breakdown_text.strip().split('\n'):
            if not line.strip():
                continue
                
            # Pattern: "Query text: passed/total (percentage%) [- STATUS]"
            match = re.match(r'^(.*?): (\d+)/(\d+) \((\d+)%\)(.*)$', line.strip())
            if match:
                query_text = match.group(1).strip()
                passed = int(match.group(2))
                total = int(match.group(3))
                percentage = int(match.group(4))
                status_part = match.group(5).strip()
                
                # Extract status if present
                status = None
                if status_part.startswith(' - '):
                    status = status_part[3:]  # Remove " - " prefix
                
                queries[query_text] = {
                    'passed': passed,
                    'total': total,
                    'percentage': percentage,
                    'status': status
                }
        
        return queries

    def _update_score_history(self, iteration: int, overall_accuracy: float, query_breakdown: str):
        """Update the score history summary file with latest results."""
        score_history_file = os.path.join(self.run_folder, 'score_history_summary.txt')
        
        # Parse current query breakdown
        query_data = self._parse_query_breakdown_for_history(query_breakdown)
        
        # Read existing query data from prompt_history.txt to get complete historical data
        all_results = []
        try:
            with open(self.prompt_history_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Parse all historical results from prompt_history.txt
            pattern = re.compile(
                r'<PROMPT>\n<PROMPT_TEXT>\n(.*?)\n</PROMPT_TEXT>\n'
                r'<OVERALL_ACCURACY>\n(.*?)\n</OVERALL_ACCURACY>\n'
                r'<QUERY_BREAKDOWN>\n(.*?)\n</QUERY_BREAKDOWN>\n'
                r'<FAILING_EXAMPLES>\n(.*?)\n</FAILING_EXAMPLES>\n</PROMPT>', 
                re.DOTALL
            )
            
            matches = pattern.findall(content)
            for i, (prompt_text, accuracy_str, breakdown_text, examples_text) in enumerate(matches):
                try:
                    overall_acc = float(accuracy_str.strip())
                    query_breakdown_data = self._parse_query_breakdown_for_history(breakdown_text)
                    
                    all_results.append({
                        'iteration': i,
                        'overall_accuracy': overall_acc,
                        'query_breakdown': query_breakdown_data
                    })
                except (ValueError, AttributeError):
                    continue
                    
        except FileNotFoundError:
            # If no history file yet, just use current result
            all_results = [{
                'iteration': iteration,
                'overall_accuracy': overall_accuracy,
                'query_breakdown': query_data
            }]
        
        if not all_results:
            return
        
        # Get all unique queries across all iterations
        all_queries = set()
        for result in all_results:
            if result.get('query_breakdown'):
                all_queries.update(result['query_breakdown'].keys())
        all_queries = sorted(all_queries)
        
        # Generate summary
        summary_lines = [
            "="*80,
            "OPTIMIZATION SCORE HISTORY (LIVE UPDATES)",
            "="*80,
            f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Completed iterations: {len(all_results)}",
            ""
        ]
        
        # Overall progress
        initial_score = all_results[0]['overall_accuracy']
        current_score = all_results[-1]['overall_accuracy']
        best_score = max(r['overall_accuracy'] for r in all_results)
        best_iteration = next(i for i, r in enumerate(all_results) if r['overall_accuracy'] == best_score)
        
        summary_lines.extend([
            "OVERALL PROGRESS:",
            "-" * 40,
            f"Initial accuracy (Iteration 0):  {initial_score:.1%}",
            f"Current accuracy (Iteration {len(all_results)-1}):  {current_score:.1%}",
            f"Best accuracy achieved:          {best_score:.1%} (Iteration {best_iteration})",
            f"Total improvement so far:        {current_score - initial_score:+.1%}",
            ""
        ])
        
        # Query-level performance across all iterations
        if all_queries:
            summary_lines.extend([
                "QUERY PERFORMANCE ACROSS ITERATIONS:",
                "-" * 80
            ])
            
            # Create header
            header = f"{'Query':<35} {'Iter':<4}"
            for result in all_results:
                header += f" {result['iteration']:<4}"
            summary_lines.append(header)
            
            # Add separator
            separator = "-" * len(header)
            summary_lines.append(separator)
            
            # Add data rows for each query
            for query in all_queries:
                # Truncate long query names
                query_display = (query[:32] + "...") if len(query) > 35 else query
                row = f"{query_display:<35} {'%':<4}"
                
                for result in all_results:
                    if result.get('query_breakdown') and query in result['query_breakdown']:
                        query_info = result['query_breakdown'][query]
                        percentage = query_info['percentage']
                        
                        # Add status indicator
                        if query_info['status'] == 'CRITICAL':
                            indicator = "⚠"
                        elif query_info['status'] == 'WEAK':
                            indicator = "⚡"
                        else:
                            indicator = ""
                        
                        cell = f"{percentage:2d}{indicator}"
                        row += f" {cell:<4}"
                    else:
                        row += f" {'--':<4}"
                
                summary_lines.append(row)
            
            summary_lines.extend([
                "",
                "LEGEND:",
                "⚠  CRITICAL (< 60% accuracy)",
                "⚡ WEAK (60-79% accuracy)",
                "-- No data available",
                ""
            ])
        
        # Overall accuracy progression
        summary_lines.extend([
            "OVERALL ACCURACY PROGRESSION:",
            "-" * 40,
            f"{'Iter':<4} {'Overall':<8} {'Status':<15}"
        ])
        
        summary_lines.append("-" * 30)
        
        for result in all_results:
            status = "✅ Complete"
            if result['overall_accuracy'] == best_score:
                status = "🏆 Best so far"
            elif result['overall_accuracy'] < 0.6:
                status = "⚠️  Critical"
            
            row = f"{result['iteration']:<4} {result['overall_accuracy']:<8.1%} {status:<15}"
            summary_lines.append(row)
        
        # Current iteration detailed breakdown
        current_result = all_results[-1]
        if current_result.get('query_breakdown'):
            summary_lines.extend([
                "",
                f"CURRENT ITERATION ({current_result['iteration']}) DETAILED BREAKDOWN:",
                "-" * 50
            ])
            
            for query, data in current_result['query_breakdown'].items():
                status_indicator = ""
                if data['status'] == 'CRITICAL':
                    status_indicator = " ⚠️"
                elif data['status'] == 'WEAK':
                    status_indicator = " ⚡"
                
                summary_lines.append(
                    f"{query}: {data['passed']}/{data['total']} ({data['percentage']}%){status_indicator}"
                )
        
        # Critical issues tracking
        summary_lines.extend([
            "",
            "CRITICAL ISSUES OVER TIME:",
            "-" * 40
        ])
        
        for result in all_results:
            if result.get('query_breakdown'):
                critical_queries = [
                    query for query, data in result['query_breakdown'].items()
                    if data['status'] == 'CRITICAL'
                ]
                
                if critical_queries:
                    # Truncate long query names for this summary
                    critical_short = [q[:30] + "..." if len(q) > 33 else q for q in critical_queries]
                    summary_lines.append(f"Iteration {result['iteration']}: {', '.join(critical_short)}")
                else:
                    summary_lines.append(f"Iteration {result['iteration']}: No critical issues ✅")
        
        # Write updated summary
        with open(score_history_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(summary_lines))
        
        logger.info(f"Updated score history: {score_history_file}")

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

                # Calculate query-level breakdown and failing examples
                query_breakdown = self._calculate_query_breakdown(details)
                failing_examples = self._extract_failing_examples(details)
                
                # Update history file with enhanced format
                async with aiofiles.open(self.prompt_history_file, 'a') as f:
                    await f.write(
                        f"<PROMPT>\n<PROMPT_TEXT>\n{current_prompt}\n</PROMPT_TEXT>\n"
                        f"<OVERALL_ACCURACY>\n{score}\n</OVERALL_ACCURACY>\n"
                        f"<QUERY_BREAKDOWN>\n{query_breakdown}\n</QUERY_BREAKDOWN>\n"
                        f"<FAILING_EXAMPLES>\n{failing_examples}\n</FAILING_EXAMPLES>\n"
                        f"</PROMPT>\n\n"
                    )
                
                # Update score history summary
                self._update_score_history(i, score, query_breakdown)

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
                     
                # Check for early stopping
                if self.early_stopping_threshold is not None and score >= self.early_stopping_threshold:
                    logger.info("="*60)
                    logger.info("🚀 EARLY STOPPING TRIGGERED!")
                    logger.info("="*60)
                    logger.info(f"Accuracy threshold {self.early_stopping_threshold:.2%} reached: {score:.2%}")
                    logger.info(f"Stopping optimization at iteration {i}")
                    logger.info("="*60)
                    break
                
            except Exception as e:
                logger.error(f"Error evaluating prompt in iteration {i}: {e}")
                continue

        # Final summary
        logger.info("\n" + "="*80)
        logger.info("OPTIMIZATION COMPLETE")
        logger.info("="*80)
        logger.info(f"Best prompt found with accuracy: {self.best_prompt['score']:.2%}")
        if self.early_stopping_threshold is not None and self.best_prompt['score'] >= self.early_stopping_threshold:
            logger.info(f"🎯 Early stopping threshold of {self.early_stopping_threshold:.2%} was reached!")
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
    max_concurrent_tests: int = 6,
    early_stopping_threshold: float = None
) -> Tuple[str, float]:
    """
    Convenience function to run prompt optimization.
    
    Args:
        starting_prompt: Initial prompt to optimize
        audio_mapping_path: Path to audio test mapping file
        num_iterations: Number of optimization iterations
        max_concurrent_tests: Max concurrent evaluations
        early_stopping_threshold: If set, stop optimization when accuracy exceeds this threshold (0.0-1.0)
        
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
        evaluator=evaluator,
        early_stopping_threshold=early_stopping_threshold
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
