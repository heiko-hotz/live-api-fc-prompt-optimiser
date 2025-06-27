# Live API Function Call Prompt Optimizer

A prompt optimization system that uses **Automatic Prompt Engineering (APE)** to iteratively improve system prompts for voice assistants with function calling capabilities.

## 🎯 What This Does

This project automatically optimizes AI system prompts by:
1. **Generating test cases** from input queries using AI-powered query restatement
2. **Creating audio test suites** for realistic voice assistant evaluation
3. **Running iterative optimization** using a metaprompt that learns from previous attempts
4. **Evaluating function calling accuracy** across diverse scenarios
5. **Providing detailed analytics** and early stopping when targets are met

## 🏗️ Project Structure

```
live-api-fc-prompt-optimiser/
├── 01_prepare_test_suite.py     # Step 1: Generate test cases and audio
├── 02_run_optimization.py       # Step 2: Run prompt optimization
├── audio_test_suite/           # Generated audio files and mappings
├── configs/
│   ├── input_queries.json      # Base queries for test generation
│   └── model_configs.py        # AI model configurations
├── data_generation/
│   ├── audio_generator.py      # Text-to-speech generation
│   └── query_restater.py       # Query variation generation
├── evaluation/
│   └── audio_fc_evaluator.py   # Function call evaluation system
├── optimization/
│   ├── metaprompt_template.txt # Template for prompt optimization
│   └── prompt_optimizer.py     # Core optimization engine
├── runs/                       # Optimization results (auto-generated)
└── requirements.txt            # Python dependencies
```

## 🚀 Quick Start

### Prerequisites

1. **Google Cloud Account** with Vertex AI enabled (or Developer API Key)
2. **Python 3.8+**
3. **Audio processing capabilities** (for TTS generation)

### 1. Environment Setup

```bash
# Clone the repository
git clone <your-repo-url>
cd live-api-fc-prompt-optimiser

# Install dependencies
pip install -r requirements.txt

# Set up Google Cloud authentication
gcloud auth application-default login

# Set environment variables
export GOOGLE_CLOUD_PROJECT="your-project-id"
export GOOGLE_CLOUD_LOCATION="us-central1"  # or your preferred region
```

### 2. Generate Test Suite

```bash
python 01_prepare_test_suite.py
```

This script will:
- Load base queries from `configs/input_queries.json`
- Generate multiple variations of each query using AI
- Create audio files using text-to-speech
- Build a comprehensive test mapping file

**Expected output**: `audio_test_suite/` directory with audio files and `audio_mapping.json`

### 3. Run Optimization

```bash
python 02_run_optimization.py
```

This will:
- Start with a baseline prompt
- Iteratively generate improved prompts using APE
- Evaluate each prompt against the audio test suite
- Save detailed results in timestamped `runs/` folders
- Stop early if accuracy threshold is exceeded

## ⚙️ Configuration

### Key Configuration Files

#### `configs/input_queries.json`
Define the base queries for test generation:
```json
{
  "general_queries": [
    "What services does Cymbal offer?",
    "Can you help me with my account?"
  ],
  "escalation_queries": [
    "I need to speak to a human",
    "This is urgent, connect me to an agent"
  ]
}
```

#### `configs/model_configs.py`
Configure AI models for different tasks:
```python
# Model for generating prompt variations
PROMPT_GENERATION_MODEL = "gemini-2.5-pro"

# Model for query restatement
QUERY_RESTATEMENT_MODEL = "gemini-2.5-flash"
```

#### `02_run_optimization.py` - Main Settings
Key parameters you can adjust:
```python
num_iterations = 3              # Number of optimization rounds
max_concurrent_tests = 1        # Parallel evaluation limit
early_stopping_threshold = 1.0  # Stop when accuracy exceeds this (0.0-1.0)
```

### Starting Prompt Customization

Edit the `starting_prompt` in `02_run_optimization.py`:
```python
starting_prompt = """
You are a helpful voice assistant for a service called Cymbal.
Your main job is to understand the user's intent and route their request to the correct tool.
- For general questions about Cymbal products or services, use the `get_chatbot_response` tool.
- If the user explicitly asks to speak to a human or a live agent, use the `escalate_to_human_agent` tool with the reason 'live-agent-request'.
- If the user sounds distressed, anxious, or mentions being in a vulnerable situation, use the `escalate_to_human_agent` tool with the reason 'vulnerable-user'.
""".strip()
```

## 📊 Understanding Results

### Output Structure

Each optimization run creates a timestamped folder in `runs/`:
```
runs/optimization_20241201_143022/
├── best_prompt.txt              # Final optimized prompt
├── best_prompt_info.json        # Metadata about best prompt
├── prompt_history.txt           # All prompts tried with scores
├── iteration_0/                 # Baseline evaluation
│   ├── evaluation_details.json
│   ├── prompt.txt
│   └── summary.json
├── iteration_1/                 # First optimization attempt
└── ...
```

### Key Metrics

- **Overall Accuracy**: Percentage of test cases where function calls matched expected results
- **Query Performance**: Breakdown by query type showing which areas need improvement
- **Critical Failures**: Specific examples of failing cases for debugging

### Early Stopping

The system will automatically stop when:
- The accuracy threshold is reached (configurable)
- Maximum iterations are completed
- Critical errors prevent continuation

Example early stopping log:
```
🚀 EARLY STOPPING TRIGGERED!
Accuracy threshold 100.0% reached: 100.0%
Stopping optimization at iteration 1
```

## 🔧 Advanced Usage

### Custom Evaluation Functions

To adapt for different function calling schemas, modify `evaluation/audio_fc_evaluator.py`:
```python
def _extract_function_call(self, response_text: str) -> Dict:
    # Implement your function call extraction logic
    pass
```

### Custom Metaprompt Templates

Edit `optimization/metaprompt_template.txt` to change how the optimizer learns:
```
You are an expert prompt engineer. Your task is to analyze the performance of previous prompts and generate an improved version.

Previous prompt performance:
{prompt_scores}

Generate a new prompt that addresses the failing cases while maintaining successful patterns.
Format your response as: [[your new prompt here]]
```

### Batch Processing

For large-scale optimization, adjust concurrency:
```python
max_concurrent_tests = 10  # Higher values = faster but more resource intensive
```

## 🐛 Troubleshooting

### Debug Mode

Enable detailed logging:
```python
# In the script you're debugging
import logging
logging.getLogger().setLevel(logging.DEBUG)
```

## 📈 Performance Tips

1. **Start Small**: Begin with fewer iterations and a small test suite
2. **Use Early Stopping**: Set realistic accuracy thresholds (0.85-0.95)
3. **Monitor Costs**: Each iteration makes multiple API calls
4. **Save Intermediate Results**: The system auto-saves, but check `runs/` regularly

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/new-capability`
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

See [LICENSE](LICENSE) file for details.

## 🆘 Support

For issues and questions:
1. Check the troubleshooting section above
2. Review logs in the `runs/` directory
3. Open an issue with detailed error messages and configuration

---

**Happy optimizing!** 🎯✨