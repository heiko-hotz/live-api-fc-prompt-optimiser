# AI Voice Assistant Prompt Optimizer

A demonstration of **Automatic Prompt Engineering (APE)** that iteratively improves system prompts for voice assistants with function calling capabilities.

## 🎯 What This Demonstrates

This project showcases how AI can automatically optimize AI system prompts by:
1. **Generating diverse test cases** from simple input queries using AI-powered query restatement
2. **Creating realistic audio test suites** for voice assistant evaluation
3. **Running iterative optimization** using a metaprompt that learns from previous attempts
4. **Evaluating function calling accuracy** across diverse scenarios and speaking styles
5. **Providing detailed analytics** with early stopping when performance targets are met

## 🎬 Demo Scenario

The demo uses a generic **virtual assistant** that can:
- **Answer general questions** using the `get_information` function
- **Escalate to human support** using the `escalate_to_support` function for:
  - Direct requests for human help (`human-request`)
  - Users in distress or vulnerable situations (`vulnerable-user`)

**Starting Point**: A basic prompt with ~60-70% function calling accuracy
**Goal**: Automatically improve to 90%+ accuracy through iterative optimization

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

1. **Google Cloud Account** with Vertex AI enabled
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
- Create audio files using text-to-speech with different voices and accents
- Build a comprehensive test mapping file

**Expected output**: `audio_test_suite/` directory with audio files and `audio_mapping.json`

### 3. Run Optimization

```bash
python 02_run_optimization.py
```

This will:
- Start with a baseline prompt (typically 60-70% accuracy)
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
  "queries": [
    {
      "query": "What's the weather like today?",
      "trigger_function": true,
      "function_name": "get_information",
      "function_args": {"query": "What's the weather like today?"}
    },
    {
      "query": "I need to speak with someone",
      "trigger_function": true,
      "function_name": "escalate_to_support",
      "function_args": {"reason": "human-request"}
    }
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
- The accuracy threshold is reached (default 90%)
- Maximum iterations are completed
- Critical errors prevent continuation

Example early stopping log:
```
🚀 EARLY STOPPING TRIGGERED!
Accuracy threshold 90.0% reached: 94.2%
Stopping optimization at iteration 3
```

## 🔧 Advanced Usage

### Custom Function Schema

To adapt for different function calling scenarios, modify the system prompt to include your functions:
```python
# Example: E-commerce assistant
functions = [
    {"name": "search_products", "description": "Search for products"},
    {"name": "get_order_status", "description": "Check order status"},
    {"name": "contact_support", "description": "Connect to customer service"}
]
```

### Custom Evaluation Logic

Edit `evaluation/audio_fc_evaluator.py` to match your function call extraction patterns:
```python
def _extract_function_call(self, response_text: str) -> Dict:
    # Implement your function call extraction logic
    pass
```

## 🎯 Use Cases

This framework can be adapted for various voice assistant scenarios:
- **Customer service bots** (routing, escalation)
- **E-commerce assistants** (search, orders, support)
- **Smart home controls** (device control, information)
- **Healthcare assistants** (appointment booking, information)
- **Educational tutors** (Q&A, assessments, help)

## 🚀 Demo Tips

For best demonstration results:
1. Start with a deliberately suboptimal prompt
2. Use diverse test queries that showcase edge cases
3. Run 3-5 optimization iterations to show improvement
4. Highlight specific failure patterns that get fixed

## 📈 Expected Results

Typical optimization runs show:
- **Baseline**: 60-75% accuracy
- **After 3 iterations**: 85-95% accuracy
- **Key improvements**: Better edge case handling, clearer function selection logic

## 🎯 Demo Walkthrough

Here's a step-by-step walkthrough for demonstrating the system:

### 1. Initial Setup (5 minutes)
```bash
# Clone and setup
git clone <your-repo-url>
cd live-api-fc-prompt-optimiser
pip install -r requirements.txt

# Set up Google Cloud
gcloud auth application-default login
export GOOGLE_CLOUD_PROJECT="your-project-id"
export GOOGLE_CLOUD_LOCATION="us-central1"
```

### 2. Generate Test Suite (3 minutes)
```bash
python 01_prepare_test_suite.py
```
This creates:
- 10 base queries (weather, AI info, homework help, human requests, distress scenarios)
- 5 restatements per query (50 total variations)
- Audio files with different voices and accents

### 3. Run Optimization (10 minutes)
```bash
python 02_run_optimization.py
```

### 4. Key Demo Points
- **Show the baseline prompt**: Simple, generic AI assistant instructions
- **Highlight initial accuracy**: Usually 60-75% (deliberately suboptimal)
- **Watch real-time optimization**: Each iteration shows improvement analysis
- **Point out failure patterns**: Which queries fail and why
- **Celebrate improvements**: How AI learns to handle edge cases better
- **Final results**: Usually 85-95% accuracy after 3-5 iterations

### 5. Example Failure → Success Patterns
- **Initial**: Struggles with indirect human requests like "Can I talk to someone?"
- **Optimized**: Learns to recognize these as escalation requests
- **Initial**: Confuses general distress with specific help requests  
- **Optimized**: Better distinguishes between 'human-request' vs 'vulnerable-user'

The demo perfectly showcases how AI can automatically improve AI through iterative prompt engineering!

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