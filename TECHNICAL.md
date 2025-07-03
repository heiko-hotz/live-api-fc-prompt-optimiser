# Technical Documentation - AI Voice Assistant Prompt Optimizer

This document provides comprehensive technical details about the implementation, architecture, and advanced usage of the automatic prompt engineering system.

## **High-Level Architecture & Purpose**

• **Automatic Prompt Engineering (APE) for voice assistants** - System iteratively improves prompts for AI models with function calling capabilities
• **Two-phase workflow**: Step 1 generates diverse audio test suites, Step 2 runs optimization using metaprompts
• **Target accuracy improvement**: Takes basic prompts from ~60-70% to 90%+ function calling accuracy
• **Voice-first design**: Built specifically for Google Cloud's Gemini Live API with real audio processing

## **Test Suite Generation (01_prepare_test_suite.py)**

### **Query Restatement Process**
• **Base queries from configs/input_queries.json** - 10 seed queries covering different function calling scenarios
• **AI-powered variation generation** - Uses Gemini 2.5 Flash to create 5 restatements per base query
• **Tone diversification** - Generates direct, casual, polite, and formal variations of each query
• **Prompt structure for restatements**: "Please restate the following user query for a financial voice assistant in 5 different ways"
• **Concurrent processing** - Batches of 5 queries processed simultaneously for efficiency
• **Intermediate output**: Saves to `data_generation/output_queries.json` before audio generation

### **Audio Generation Process** 
• **Multi-dialect voice support** - 9 different English voices across US, UK, Australian, and Indian dialects
• **Voice randomization** - Each query variation gets a randomly selected voice to increase test diversity
• **Critical detail: 1-second silence addition** - Every audio file gets exactly 1.0 seconds of silence appended using `add_silence()` function
• **16kHz sample rate requirement** - All audio generated at exactly 16kHz to match Gemini Live API requirements
• **Manual WAV header writing** - System manually constructs WAV headers with specific PCM format (16-bit, mono, 16kHz)
• **Natural speech enhancements** - Adds markup pauses after periods, question marks, and exclamation points
• **Speaking rate optimization** - Set to 0.95x speed for better clarity
• **Structured file organization** - Creates `query_01/`, `query_02/` directories with specific naming: `original_en-US_Charon.wav`, `restatement_01_en-GB_Puck.wav`

## **Optimization Engine (02_run_optimization.py)**

### **Prompt History & Metaprompt System**
• **Score history sorting** - Before adding to metaprompt, prompts are sorted by accuracy in ascending order (worst to best)
• **Enhanced history format** - Stores iteration number, prompt text, overall accuracy, query breakdown, and failing examples
• **Metaprompt template injection** - Uses `{prompt_scores}` placeholder replacement in `metaprompt_template.txt`
• **Detailed failure analysis** - Captures specific examples like `"What time is it?" → Expected: get_information(query='What time is it?'), Got: None`
• **Query-level performance tracking** - Marks queries as CRITICAL (<60% accuracy) or WEAK (<80% accuracy)

### **Evaluation Process**
• **Gemini Live API integration** - Tests prompts against real voice processing using `gemini-live-2.5-flash`
• **Function call comparison logic** - Compares expected vs actual function calls with detailed mismatch reporting
• **Concurrent test execution** - Runs up to 6 audio tests simultaneously for speed
• **20-second timeout per test** - Prevents hanging on problematic audio inputs
• **Real-time progress tracking** - Shows `[23/60]` style progress with color-coded PASS/FAIL status
• **Librosa audio processing** - Loads audio files, converts to required 16-bit PCM format for API consumption

### **Optimization Loop Details**
• **Exponential backoff for API calls** - Retries prompt generation up to 3 times with exponential delays
• **Response parsing with regex** - Extracts new prompts using `\[\[(.*?)\]\]` pattern matching
• **Early stopping threshold** - Automatically stops when accuracy exceeds configurable threshold (default 90%)
• **Timestamped run directories** - Creates `runs/optimization_20241201_143022/` folders for each session
• **Iteration-specific results** - Saves `iteration_0/`, `iteration_1/` subdirectories with full evaluation details

## **Technical Implementation Details**

### **Audio Processing Specifics**
• **Raw PCM data manipulation** - Converts audio to bytes, adds silence samples using NumPy
• **Silence calculation**: `silence_samples = int(sample_rate * silence_duration)` where duration = 1.0 seconds
• **WAV header construction** - Manually writes RIFF, WAVE, fmt, and data chunks with proper byte ordering
• **Google Cloud Text-to-Speech integration** - Uses `TextToSpeechClient` with `LINEAR16` encoding
• **Async audio generation** - All audio files generated concurrently using `asyncio.gather()`

### **Model Configuration & Usage**
• **Three distinct models**: Gemini 2.5 Flash for restatements, Gemini 2.5 Pro for prompt optimization, Gemini Live 2.5 Flash for evaluation
• **Temperature settings**: 0.7 for restatements, 0.8 for optimization, 0.0 for evaluation (deterministic)
• **Vertex AI client initialization** - Uses environment variables `GOOGLE_CLOUD_PROJECT` and `GOOGLE_CLOUD_LOCATION`
• **Function schema definitions** - Precisely defined `get_information` and `escalate_to_support` schemas with parameter validation

### **Result Tracking & Analytics**
• **Comprehensive logging** - Timestamped logs with emoji indicators (🎉 for improvements, ❌ for failures, ⏰ for timeouts)
• **Score history summary generation** - Creates query performance matrices showing accuracy across iterations
• **Query ID mapping** - Assigns `query_1`, `query_2` identifiers for tracking specific query performance
• **Best prompt preservation** - Automatically saves `best_prompt.txt` and `best_prompt_info.json` when improvements found
• **Detailed evaluation exports** - CSV files with per-test results including voice type, expected/actual function calls

### **Error Handling & Robustness**
• **Graceful API failure handling** - Continues optimization even if individual prompt generation fails
• **File system safety checks** - Warns before deleting existing test suites to prevent accidental regeneration
• **Environment validation** - Checks for required cloud credentials and environment variables before starting
• **Memory management** - Uses streaming file operations and async processing to handle large test suites

### **Function Calling Logic Preservation**
• **Critical preservation requirements** - Metaprompt specifically instructs to preserve non-function-calling elements (identity, branding, language handling)
• **Function-specific optimization** - Only modifies sections determining when to call `get_information` vs `escalate_to_support`
• **Reason parameter handling** - Precisely distinguishes between 'human-request' and 'vulnerable-user' escalation reasons
• **Response format validation** - Ensures generated prompts maintain voice interaction guidelines and response structure

## **Standalone Test Modes & User Usage**

### **Standalone Prompt Evaluation (evaluation/audio_fc_evaluator.py)**

**Purpose & Usage:**
• **Direct prompt testing** - Test individual prompts against the full audio test suite without running optimization
• **Command to run**: `python evaluation/audio_fc_evaluator.py` from project root
• **Use cases**: 
  - Quickly evaluate a specific prompt's performance
  - Test prompts from other sources or manual edits
  - Debug prompt performance issues without expensive optimization runs
  - Baseline testing before starting optimization

**Implementation Details:**
• **Hardcoded baseline prompt** - Uses a simple, human-written prompt for consistent testing
• **Full test suite execution** - Runs all ~60 audio test cases (original + restatements)
• **Concurrent execution** - Configurable batch size (default: 6 concurrent tests)
• **Comprehensive output** - Shows overall accuracy, detailed failures, timeouts, and errors
• **Timestamped results** - Creates `runs/standalone_eval_YYYYMMDD_HHMMSS/` directories
• **Dual output formats** - Saves both JSON (`evaluation_results.json`) and CSV (`evaluation_results.csv`)
• **Quick results file** - Also saves `standalone_eval_results.json` in project root for immediate inspection

**What Users See:**
• **Real-time progress** - Shows `[23/60]` style progress with ✅/❌ status indicators
• **Detailed failure analysis** - Lists specific failing audio files with expected vs actual function calls
• **Performance metrics** - Overall accuracy percentage, execution time, average time per test
• **Error categorization** - Separate reporting for failures, timeouts, and API errors

### **Standalone Debug Mode (optimization/prompt_optimizer.py)**

**Purpose & Usage:**
• **Development debugging** - Test prompt generation pipeline components in isolation
• **Command to run**: `python optimization/prompt_optimizer.py` from project root
• **Use cases**:
  - Validate metaprompt template functionality
  - Test API connectivity to optimization model (Gemini 2.5 Pro)
  - Debug prompt generation without running full evaluation
  - Troubleshoot optimization pipeline issues

**Implementation Details:**
• **Two-stage testing**:
  1. **Metaprompt generation test** - Validates template loading and placeholder replacement
  2. **API generation test** - Tests actual call to optimization model with metaprompt
• **Minimal setup** - Uses dummy evaluator, requires only test suite existence check
• **Error isolation** - Shows exactly where in the pipeline failures occur
• **Response validation** - Tests regex extraction of prompts from model responses using `[[...]]` format

**What Users See:**
• **Step-by-step validation**:
  - "✅ Metaprompt created successfully (X characters)"
  - "✅ Prompt generated successfully!"
• **Detailed API debugging** - Shows full metaprompt sent to model and complete model response
• **Generated prompt preview** - First 200 characters of successfully generated prompt
• **Specific error reporting** - API failures, parsing errors, authentication issues

### **Prerequisites for Standalone Tests**

**Required Setup:**
• **Test suite must exist** - Both require `audio_test_suite/audio_mapping.json` from `01_prepare_test_suite.py`
• **Environment variables** - `GOOGLE_CLOUD_PROJECT` and `GOOGLE_CLOUD_LOCATION` must be set
• **Authentication** - Must run `gcloud auth application-default login` first
• **API access** - Vertex AI and Gemini Live APIs must be enabled

### **When to Use Each Standalone Mode**

**Use Standalone Evaluator when:**
• Testing prompts created outside the optimization system
• Validating manual prompt improvements
• Getting baseline performance measurements
• Debugging why specific queries fail without optimization overhead
• Comparing different prompt variations quickly

**Use Standalone Debug when:**
• Optimization runs fail with prompt generation errors
• Testing new metaprompt templates
• Validating API connectivity and authentication
• Debugging response parsing issues
• Development and troubleshooting of the optimization pipeline

### **Integration with Main Workflow**

**Workflow Integration:**
• **Pre-optimization testing** - Run standalone evaluator to establish baseline before `02_run_optimization.py`
• **Troubleshooting** - Use debug mode when optimization fails to isolate issues
• **Manual iteration** - Test user-modified prompts outside of automatic optimization
• **Development validation** - Verify system components work independently before full integration

**File Output Compatibility:**
• **Same format as optimization runs** - Standalone evaluator creates results in same format as optimization iterations
• **Compatible analysis** - Results can be compared directly with optimization run outputs
• **Reusable test data** - Generated audio test suite works for both standalone and optimization modes

## **Advanced Configuration Examples**

### **Custom Voice Selection**

Modify `data_generation/audio_generator.py` to use specific voices:

```python
# Use only US English voices
VOICES = [
    ("en-US", "en-US-Polyglot-1"),
    ("en-US", "en-US-Studio-M"),
    ("en-US", "en-US-Studio-O")
]
```

### **Custom Function Schemas**

Extend function definitions in evaluation system:

```python
# Add new function to schema
FUNCTIONS = [
    {
        "name": "get_information",
        "description": "Get information about a topic",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "The user's question"}
            },
            "required": ["query"]
        }
    },
    {
        "name": "book_appointment",
        "description": "Book a calendar appointment",
        "parameters": {
            "type": "object", 
            "properties": {
                "datetime": {"type": "string"},
                "duration": {"type": "string"}
            },
            "required": ["datetime"]
        }
    }
]
```

### **Custom Metaprompt Templates**

Modify `optimization/metaprompt_template.txt` for domain-specific optimization:

```
You are optimizing prompts for a healthcare voice assistant.
Focus specifically on:
- Medical terminology accuracy
- HIPAA compliance awareness
- Emergency escalation protocols
- Patient privacy protection

Previous prompt performance:
{prompt_scores}

Generate an improved prompt that maintains medical accuracy while improving function calling performance.
```

### **Performance Tuning Parameters**

Adjust concurrency and timeouts in `02_run_optimization.py`:

```python
# For faster evaluation (more concurrent tests)
max_concurrent_tests = 12  # Default: 6

# For slower/more reliable evaluation
max_concurrent_tests = 3

# Adjust timeout for complex queries
timeout_seconds = 30  # Default: 20

# More aggressive early stopping
early_stopping_threshold = 0.85  # Stop at 85% accuracy
```

## **File Format Specifications**

### **Audio Mapping JSON Structure**

```json
{
  "query_1": {
    "original": {
      "text": "What's the weather like today?",
      "audio_file": "audio_test_suite/query_01/original_en-US_Charon.wav",
      "voice": "en-US_Charon",
      "expected_function": "get_information",
      "expected_args": {"query": "What's the weather like today?"}
    },
    "restatements": [
      {
        "text": "Can you tell me about today's weather conditions?",
        "audio_file": "audio_test_suite/query_01/restatement_01_en-GB_Puck.wav",
        "voice": "en-GB_Puck",
        "expected_function": "get_information",
        "expected_args": {"query": "Can you tell me about today's weather conditions?"}
      }
    ]
  }
}
```

### **Evaluation Results JSON Structure**

```json
{
  "overall_accuracy": 0.8667,
  "total_tests": 60,
  "passed": 52,
  "failed": 6,
  "timeouts": 1,
  "errors": 1,
  "query_breakdown": {
    "query_1": {"accuracy": 1.0, "passed": 6, "total": 6},
    "query_2": {"accuracy": 0.833, "passed": 5, "total": 6}
  },
  "failures": [
    {
      "query_id": "query_3",
      "audio_file": "audio_test_suite/query_03/restatement_02_en-AU_Ruby.wav",
      "expected": {"function": "escalate_to_support", "args": {"reason": "human-request"}},
      "actual": {"function": "get_information", "args": {"query": "Can I speak to someone?"}}
    }
  ]
}
```

## **Development & Contributing**

### **Adding New Query Types**

1. **Add to input_queries.json**:
```json
{
  "query": "Set a reminder for tomorrow",
  "trigger_function": true,
  "function_name": "create_reminder",
  "function_args": {"text": "reminder", "date": "tomorrow"}
}
```

2. **Update function schema** in evaluation system
3. **Add handling logic** to metaprompt template
4. **Regenerate test suite** with new queries

### **Custom Evaluation Metrics**

Extend `evaluation/audio_fc_evaluator.py`:

```python
def calculate_custom_metrics(self, results):
    """Calculate domain-specific metrics"""
    escalation_accuracy = self._calculate_escalation_accuracy(results)
    response_time_avg = self._calculate_avg_response_time(results)
    return {
        "escalation_accuracy": escalation_accuracy,
        "avg_response_time": response_time_avg
    }
```

### **Debugging Common Issues**

**Audio Generation Failures:**
- Check Google Cloud TTS API quotas
- Verify voice names in `VOICES` list
- Ensure proper sample rate conversion

**Evaluation Timeouts:**
- Increase timeout in evaluator configuration
- Check Gemini Live API connectivity
- Verify audio file formats (16kHz, mono, WAV)

**Optimization Convergence Issues:**
- Review metaprompt template effectiveness
- Check starting prompt quality
- Adjust temperature settings for prompt generation

This system demonstrates sophisticated automatic prompt engineering with real audio processing, providing a complete pipeline from test generation through iterative optimization with detailed performance analytics. 