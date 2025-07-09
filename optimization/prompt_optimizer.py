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

# Configure logging with whitelist approach
# Root logger at INFO by default (keeps external libraries quiet)
logging.basicConfig(
    level=logging.INFO,  # Root logger at INFO to suppress external library debug noise
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Explicitly enable DEBUG logging only for our application modules
logging.getLogger('optimization').setLevel(logging.DEBUG)

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
        logger.debug(f"Using generation model: {PROMPT_GENERATION_MODEL}")
        logger.debug(f"Running {num_iterations} optimization iterations")
        if early_stopping_threshold is not None:
            logger.debug(f"Early stopping enabled: will stop if accuracy exceeds {early_stopping_threshold:.2%}")

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
                r'<PROMPT>\n<ITERATION>\n(.*?)\n</ITERATION>\n'
                r'<PROMPT_TEXT>\n(.*?)\n</PROMPT_TEXT>\n'
                r'<OVERALL_ACCURACY>\n(.*?)\n</OVERALL_ACCURACY>\n'
                r'<QUERY_BREAKDOWN>\n(.*?)\n</QUERY_BREAKDOWN>\n'
                r'<FAILING_EXAMPLES>\n(.*?)\n</FAILING_EXAMPLES>\n</PROMPT>', 
                re.DOTALL
            )
            enhanced_matches = enhanced_pattern.findall(content)
            
            if enhanced_matches:
                # Sort by overall accuracy (ascending) - iteration info is captured but not used for sorting
                sorted_prompts = sorted(enhanced_matches, key=lambda x: float(x[2]), reverse=False)
                
                # Format enhanced data for metaprompt (iteration info is not included)
                formatted_history = ""
                for iteration, prompt, accuracy, breakdown, examples in sorted_prompts:
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
            
            logger.debug("="*80)
            logger.debug("📝 GENERATING NEW PROMPT WITH OPTIMIZATION MODEL")
            logger.debug("="*80)
            logger.debug(f"Model: {PROMPT_GENERATION_MODEL}")
            logger.debug(f"Project: {project_id}, Location: {location}")
            logger.debug(f"Metaprompt length: {len(metaprompt)} characters")
            logger.debug("-"*80)
            
            logger.debug("📋 METAPROMPT BEING SENT TO MODEL:")
            logger.debug("-"*80)
            logger.debug(metaprompt)
            logger.debug("-"*80)
            
            client = genai.Client(
                vertexai=True,
                project=project_id,
                location=location,
            )
            
            # Make async call using the correct API pattern
            logger.debug("⏳ Making API call to optimization model...")
            response = await client.aio.models.generate_content(
                model=PROMPT_GENERATION_MODEL,
                contents=metaprompt,
                # Note: generation_config commented out as in query_restater.py to avoid issues
                # generation_config=PROMPT_GENERATION_CONFIG
            )
            
            logger.debug("✅ API call completed successfully")
            
            # Extract full response
            response_text = response.text if hasattr(response, 'text') else str(response)
            
            logger.debug("="*80)
            logger.debug("🤖 MODEL'S COMPLETE RESPONSE:")
            logger.debug("="*80)
            logger.debug(response_text)
            logger.debug("="*80)
            
            # Extract prompt from response using regex
            match = re.search(r'\[\[(.*?)\]\]', response_text, re.DOTALL)
            if match:
                generated_prompt = match.group(1).strip()
                
                logger.debug("🎯 EXTRACTED NEW PROMPT:")
                logger.debug("-"*80)
                logger.debug(generated_prompt)
                logger.debug("-"*80)
                logger.info(f"✅ Generated prompt length: {len(generated_prompt)} characters")
                
                return generated_prompt
            else:
                logger.warning("⚠️  Could not extract a new prompt from the model's response. Retrying.")
                logger.debug("❌ RESPONSE PARSING FAILED - No [[...]] format found")
                raise ValueError("No prompt found in [[...]] format")
                
        except Exception as e:
            logger.error(f"💥 Error generating new prompt: {e}")
            logger.debug(f"Error type: {type(e).__name__}")
            import traceback
            logger.debug(f"Full traceback: {traceback.format_exc()}")
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

    def _parse_existing_query_breakdown(self, content: str, all_results: list):
        """
        Parse the existing query breakdown table from score history content
        and add query_breakdown data to all_results.
        """
        import re
        
        # Find the query performance table section
        table_match = re.search(r'QUERY PERFORMANCE ACROSS ITERATIONS:.*?\n(.*?)(?=\n\n|\Z)', content, re.DOTALL)
        if not table_match:
            return
        
        table_content = table_match.group(1)
        lines = table_content.strip().split('\n')
        
        # Find the header line to extract iteration numbers
        header_line = None
        for line in lines:
            if 'Iter' in line and 'Query' in line:
                header_line = line
                break
        
        if not header_line:
            return
        
        # Parse iteration numbers from header
        # Format: "ID       Query                               Iter 0    1   "
        iteration_matches = re.findall(r'(\d+)', header_line)
        if not iteration_matches:
            return
        
        iteration_numbers = [int(x) for x in iteration_matches]
        
        # Parse each query row
        for line in lines:
            if line.startswith('query_'):
                # Format: "query_1  What's the weather like today?      %    --   100 "
                parts = line.split()
                if len(parts) < 4:
                    continue
                
                query_id = parts[0]
                
                # Extract the query text (everything between query_id and the % marker)
                # Find the position of the % marker
                percent_pos = line.find('%')
                if percent_pos == -1:
                    continue
                
                # Extract query text between query_id and %
                query_text_start = len(query_id)
                query_text = line[query_text_start:percent_pos].strip()
                
                # Extract the performance values after the % marker
                values_part = line[percent_pos + 1:].strip()
                values = values_part.split()
                
                # Match values to iterations
                for i, value in enumerate(values):
                    if i < len(iteration_numbers):
                        iteration_num = iteration_numbers[i]
                        
                        # Find the corresponding result in all_results
                        for result in all_results:
                            if result['iteration'] == iteration_num:
                                if 'query_breakdown' not in result:
                                    result['query_breakdown'] = {}
                                
                                # Only add if it's not "--" (missing data)
                                if value != '--':
                                    try:
                                        percentage = int(value)
                                        result['query_breakdown'][query_text] = {
                                            'percentage': percentage
                                        }
                                    except ValueError:
                                        # Skip invalid values
                                        pass

    def _update_score_history(self, iteration: int, overall_accuracy: float, query_breakdown: str):
        """
        Updates the score history summary file with latest results.
        Maintains a CSV-like format showing performance across all iterations.
        """
        score_history_file = os.path.join(self.run_folder, 'score_history_summary.txt')
        
        # Load existing results to maintain history
        all_results = []
        if os.path.exists(score_history_file):
            with open(score_history_file, 'r', encoding='utf-8') as f:
                content = f.read()
                # Parse existing iterations from the content
                import re
                # Match both "INITIAL" and "ITER X" patterns
                matches = re.findall(r'  (INITIAL|ITER (\d+))\s*:\s*([\d.]+)%', content)
                for match in matches:
                    if match[0] == 'INITIAL':
                        iteration_num = 0
                    else:
                        iteration_num = int(match[1])
                    
                    all_results.append({
                        'iteration': iteration_num,
                        'score': float(match[2]) / 100.0
                    })
                
                # Parse existing query breakdown table if it exists
                self._parse_existing_query_breakdown(content, all_results)
        
        # Add current result
        current_result = {
            'iteration': iteration,
            'score': overall_accuracy,
            'query_breakdown': self._parse_query_breakdown_for_history(query_breakdown)
        }
        
        # Update or add current iteration
        existing_idx = None
        for i, result in enumerate(all_results):
            if result['iteration'] == iteration:
                existing_idx = i
                break
        
        if existing_idx is not None:
            all_results[existing_idx] = current_result
        else:
            all_results.append(current_result)
        
        # Sort by iteration
        all_results.sort(key=lambda x: x['iteration'])
        
        # Collect all unique queries for the breakdown table with proper matching
        all_queries = []
        query_mapping = {}  # Maps normalized query -> canonical query text
        
        for result in all_results:
            if result.get('query_breakdown'):
                for query in result['query_breakdown'].keys():
                    # Normalize query for matching (remove ellipsis, extra spaces)
                    normalized = query.strip().rstrip('...').strip()
                    
                    # Check if this query matches any existing query
                    matched = False
                    for existing_norm, existing_canonical in query_mapping.items():
                        if (normalized == existing_norm or 
                            normalized.startswith(existing_norm) or 
                            existing_norm.startswith(normalized)):
                            # Use the longer, more complete version as canonical
                            if len(query) > len(existing_canonical):
                                query_mapping[existing_norm] = query
                                # Update in all_queries list
                                idx = all_queries.index(existing_canonical)
                                all_queries[idx] = query
                            matched = True
                            break
                    
                    if not matched:
                        query_mapping[normalized] = query
                        all_queries.append(query)
        
        # Generate summary content
        summary_lines = [
            "OPTIMIZATION SCORE HISTORY",
            "=" * 80,
            "",
            "OVERALL ACCURACY BY ITERATION:",
            "-" * 40
        ]
        
        for result in all_results:
            iteration_label = "INITIAL" if result['iteration'] == 0 else f"ITER {result['iteration']}"
            summary_lines.append(f"  {iteration_label:<8}: {result['score']:.1%}")
        
        summary_lines.extend([
            "",
            f"BEST SCORE: {max(r['score'] for r in all_results):.1%}",
            ""
        ])
        
        # Query-level performance across all iterations (without legend)
        if all_queries:
            summary_lines.extend([
                "QUERY PERFORMANCE ACROSS ITERATIONS:",
                "-" * 80
            ])
            
            # Create header with ID column
            header = f"{'ID':<8} {'Query':<35} {'Iter':<4}"
            for result in all_results:
                header += f" {result['iteration']:<4}"
            summary_lines.append(header)
            
            # Add separator
            separator = "-" * len(header)
            summary_lines.append(separator)
            
            # Add data rows for each query (with ID column)
            for query_index, query in enumerate(all_queries, 1):
                # Generate query ID (query_1, query_2, etc.)
                query_id = f"query_{query_index}"
                
                # Truncate long query names
                query_display = (query[:32] + "...") if len(query) > 35 else query
                row = f"{query_id:<8} {query_display:<35} {'%':<4}"
                
                for result in all_results:
                    if result.get('query_breakdown'):
                        # Find matching query using normalization
                        matched_query = None
                        query_normalized = query.strip().rstrip('...').strip()
                        
                        for result_query in result['query_breakdown'].keys():
                            result_normalized = result_query.strip().rstrip('...').strip()
                            if (query_normalized == result_normalized or 
                                query_normalized.startswith(result_normalized) or 
                                result_normalized.startswith(query_normalized)):
                                matched_query = result_query
                                break
                        
                        if matched_query:
                            query_info = result['query_breakdown'][matched_query]
                            percentage = query_info['percentage']
                            
                            # Remove status indicators - just show percentage
                            cell = f"{percentage:2d}"
                            row += f" {cell:<4}"
                        else:
                            row += f" {'--':<4}"
                    else:
                        row += f" {'--':<4}"
                
                summary_lines.append(row)
        
        # Write updated summary
        with open(score_history_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(summary_lines))
        
        logger.debug(f"Updated score history: {score_history_file}")

    async def run(self) -> Tuple[str, float]:
        """
        Executes the main optimization loop.
        
        Returns:
            Tuple of (best_prompt_text, best_score)
        """
        logger.info("="*80)
        logger.info("STARTING PROMPT OPTIMIZATION")
        logger.info("="*80)
        logger.debug(f"Initial prompt length: {len(self.starting_prompt)} characters")
        logger.debug(f"Optimization iterations: {self.num_iterations}")
        logger.debug(f"Results directory: {self.run_folder}")
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
                    logger.debug(f"Generated prompt length: {len(current_prompt)} characters")
                except Exception as e:
                    logger.error(f"Failed to generate new prompt after retries: {e}. Skipping iteration.")
                    continue

            # Evaluate the current prompt
            logger.info(f"Evaluating prompt for iteration {i}...")
            logger.debug("-"*40)
            
            try:
                # Create iteration-specific output directory for detailed results
                iteration_folder = os.path.join(self.run_folder, f'iteration_{i}')
                
                score, details = await self.evaluator.evaluate_prompt(
                    current_prompt, 
                    save_detailed_results=True, 
                    output_dir=iteration_folder
                )
                
                logger.debug("-"*40)
                logger.info(f"Iteration {i} completed - Score: {score:.2%}")
                
                # Save iteration results
                await self._save_iteration_results(i, current_prompt, score, details)

                # Calculate query-level breakdown and failing examples
                query_breakdown = self._calculate_query_breakdown(details)
                failing_examples = self._extract_failing_examples(details)
                
                # Update history file with enhanced format
                iteration_label = "INITIAL PROMPT" if i == 0 else f"ITERATION {i}"
                async with aiofiles.open(self.prompt_history_file, 'a') as f:
                    await f.write(
                        f"<PROMPT>\n<ITERATION>\n{iteration_label}\n</ITERATION>\n"
                        f"<PROMPT_TEXT>\n{current_prompt}\n</PROMPT_TEXT>\n"
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
        logger.debug("\nBest prompt text:")
        logger.debug("-"*40)
        logger.debug(self.best_prompt['text'])
        logger.debug("-"*40)
        
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
You are a helpful AI voice assistant.
Your main job is to understand the user's intent and route their request to the correct function.
- For general questions about topics, information requests, or knowledge queries, use the `get_information` function.
- If the user explicitly asks to speak to a human, get help from a person, or requests human assistance, use the `escalate_to_support` function with the reason 'human-request'.
- If the user sounds distressed, anxious, mentions feeling overwhelmed, or describes a difficult situation, use the `escalate_to_support` function with the reason 'vulnerable-user'.
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
