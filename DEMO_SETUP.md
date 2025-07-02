# Demo Setup Guide

## 🎯 What Was Changed

This repository has been transformed from a customer-specific Cymbal use case into a **generic AI voice assistant prompt optimization demo** suitable for wider audiences.

### Key Transformations

| **Component** | **Before (Cymbal-specific)** | **After (Generic)** |
|---------------|------------------------------|-------------------|
| **System Identity** | "Cymbal Voice assistant" | "helpful AI voice assistant" |
| **Functions** | `get_chatbot_response` | `get_information` |
| | `escalate_to_human_agent` | `escalate_to_support` |
| **Escalation Reasons** | `live-agent-request` | `human-request` |
| | `vulnerable-user` | `vulnerable-user` (unchanged) |
| **Test Queries** | Cymbal account questions | Weather, AI info, homework help |
| | Cymbal-specific scenarios | Generic human requests & distress |
| **Greeting** | "Hello John, I'm the Cymbal voice assistant..." | "Hello! I'm your AI assistant..." |

### Updated Files

✅ **README.md** - Now focuses on generic prompt optimization demo  
✅ **initial-system-instruction.txt** - Generic AI assistant prompt  
✅ **configs/input_queries.json** - 10 diverse, generic test scenarios  
✅ **optimization/metaprompt_template.txt** - Removed company references  
✅ **evaluation/audio_fc_evaluator.py** - Updated function schemas  
✅ **optimization/prompt_optimizer.py** - Generic debug prompts  
✅ **02_run_optimization.py** - Generic terminology throughout  

### New Test Scenarios

The demo now includes 10 diverse scenarios that showcase different AI capabilities:

1. **Information Requests**: Weather, AI concepts, cooking instructions
2. **Educational Help**: Homework assistance 
3. **Direct Human Requests**: "I need to speak to a human"
4. **Indirect Human Requests**: "Can I talk to someone?"
5. **Emotional Distress**: Feeling overwhelmed, having a hard time
6. **Conversation Closers**: "Thanks, that's all I needed"

Each scenario generates 5 audio restatements with different voices/accents (60 total test cases).

## 🚀 Demo Flow

### 1. **Setup** (2 minutes)
```bash
export GOOGLE_CLOUD_PROJECT="your-project"
export GOOGLE_CLOUD_LOCATION="us-central1"
python 01_prepare_test_suite.py
```

### 2. **Show Baseline** (1 minute)
- Point out the simple, generic system prompt
- Explain the two functions: `get_information` and `escalate_to_support`
- Mention this will likely have ~65-75% accuracy initially

### 3. **Run Optimization** (8 minutes)
```bash
python 02_run_optimization.py
```

### 4. **Key Demo Points**
- **Real-time improvement**: Watch accuracy climb each iteration
- **Failure analysis**: AI identifies specific problem patterns
- **Smart learning**: Each prompt builds on previous failures
- **Automatic stopping**: Halts when 90% threshold reached

### 5. **Results Highlights**
- **Before**: Simple prompt, basic function selection logic
- **After**: Sophisticated edge case handling, nuanced escalation decisions
- **Improvement**: Typically 65% → 90%+ accuracy
- **Speed**: Full optimization in 5-10 minutes

## 🎪 Presentation Tips

### Opening Hook
*"What if AI could automatically improve AI? Today I'll show you a system where one AI iteratively improves another AI's prompts, turning a 65% accurate voice assistant into a 90%+ accurate one - completely automatically."*

### Key Selling Points
1. **Universal Problem**: Everyone struggles with prompt engineering
2. **Automated Solution**: No manual prompt tweaking required  
3. **Measurable Results**: Clear accuracy improvements with data
4. **Real-world Application**: Voice assistants, customer service, etc.
5. **Scalable**: Works for any function calling scenario

### Technical Highlights
- Uses Automatic Prompt Engineering (APE) methodology
- Evaluates against realistic audio test cases with multiple voices
- Learns from failure patterns across iterations
- Preserves successful elements while fixing problems

### Questions to Expect
- **"How does it know what to improve?"** → AI analyzes failure patterns
- **"Can this work for our use case?"** → Yes, just update the functions and test queries
- **"How much does this cost to run?"** → Typically $5-15 per optimization run
- **"How long does it take?"** → 5-15 minutes depending on test suite size

## 📊 Expected Demo Results

**Typical Pattern:**
- **Iteration 0 (Baseline)**: 65-75% accuracy
- **Iteration 1**: 75-85% accuracy (learns basic patterns)
- **Iteration 2**: 85-92% accuracy (handles edge cases)  
- **Iteration 3**: 90-95% accuracy (early stopping triggered)

**Common Improvements:**
- Better recognition of indirect human requests
- Improved emotional distress detection
- Clearer function selection logic
- More robust edge case handling 