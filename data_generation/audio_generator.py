import os
import json
import asyncio
import random
import shutil
from typing import List, Dict
from google.cloud import texttospeech
import numpy as np
import struct
import logging
import io
import sys
from pathlib import Path
from google.api_core import exceptions as google_exceptions

# Add the parent directory to Python path to find the configs module
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configuration
INPUT_FILE = "data_generation/output_queries.json"  # Reads the intermediate file
OUTPUT_DIR = "audio_test_suite"     # Main output directory
MAPPING_FILE = os.path.join(OUTPUT_DIR, "audio_mapping.json") # The final, important output

# Available English voices and dialects, EXACTLY as in the original project
VOICE_CONFIGS = [
    # US English voices
    {"name": "en-US-Chirp3-HD-Charon", "dialect": "en-US"},
    {"name": "en-US-Chirp3-HD-Kore", "dialect": "en-US"},
    {"name": "en-US-Chirp3-HD-Leda", "dialect": "en-US"},
    
    # UK English voices
    {"name": "en-GB-Chirp3-HD-Puck", "dialect": "en-GB"},
    {"name": "en-GB-Chirp3-HD-Aoede", "dialect": "en-GB"},
    
    # Australian English voices
    {"name": "en-AU-Chirp3-HD-Zephyr", "dialect": "en-AU"},
    {"name": "en-AU-Chirp3-HD-Fenrir", "dialect": "en-AU"},
    
    # Indian English voices
    {"name": "en-IN-Chirp3-HD-Orus", "dialect": "en-IN"},
    {"name": "en-IN-Chirp3-HD-Gacrux", "dialect": "en-IN"}
]

# --- Logging Configuration ---
logger = logging.getLogger(__name__)

# Suppress verbose logging from external libraries
logging.getLogger('google_genai.live').setLevel(logging.WARNING)
logging.getLogger('google_genai').setLevel(logging.WARNING)

def add_silence(audio_data: bytes, sample_rate: int = 16000, silence_duration: float = 1.0) -> bytes:
    """
    Adds silence to the end of raw 16-bit PCM audio data.
    This logic is preserved exactly from the original project.
    """
    silence_samples = int(sample_rate * silence_duration)
    silence_bytes = np.zeros(silence_samples, dtype=np.int16).tobytes()
    return audio_data + silence_bytes


async def _synthesize_audio_for_text(text: str, voice_config: Dict, output_path: str, client):
    """Synthesizes a single audio file, adds silence, and saves it with proper WAV headers."""
    try:
        # Add natural pauses and flow using markup
        markup_text = text.replace(". ", ". [pause] ")
        markup_text = markup_text.replace("? ", "? [pause] ")
        markup_text = markup_text.replace("! ", "! [pause] ")
        
        # Configure the voice
        voice = texttospeech.VoiceSelectionParams(
            name=voice_config["name"],
            language_code=voice_config["dialect"]
        )
        
        # Configure audio output - MUST be 16kHz for Gemini Live API
        audio_config = texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.LINEAR16,
            sample_rate_hertz=16000,  # Required by Gemini Live API
            speaking_rate=0.95  # Slightly slower for better clarity
        )
        
        # Generate the audio asynchronously
        response = await asyncio.to_thread(
            client.synthesize_speech,
            input=texttospeech.SynthesisInput(markup=markup_text),
            voice=voice,
            audio_config=audio_config
        )
        
        # ** CRITICAL: Add silence to the end of the generated audio **
        audio_with_silence = add_silence(response.audio_content)
        
        # Ensure parent directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save the final audio file with proper WAV headers
        with open(output_path, "wb") as out:
            # Write WAV header manually to ensure proper format
            # WAV header for 16-bit PCM at 16kHz mono
            sample_rate = 16000
            bits_per_sample = 16
            channels = 1
            byte_rate = sample_rate * channels * bits_per_sample // 8
            block_align = channels * bits_per_sample // 8
            data_size = len(audio_with_silence)
            file_size = 36 + data_size
            
            # Write WAV header
            out.write(b'RIFF')
            out.write(struct.pack('<I', file_size))
            out.write(b'WAVE')
            out.write(b'fmt ')
            out.write(struct.pack('<I', 16))  # fmt chunk size
            out.write(struct.pack('<H', 1))   # PCM format
            out.write(struct.pack('<H', channels))
            out.write(struct.pack('<I', sample_rate))
            out.write(struct.pack('<I', byte_rate))
            out.write(struct.pack('<H', block_align))
            out.write(struct.pack('<H', bits_per_sample))
            out.write(b'data')
            out.write(struct.pack('<I', data_size))
            out.write(audio_with_silence)
            
        logger.debug(f"Generated audio with silence: {output_path}")
        
    except Exception as e:
        logger.error(f"Error generating audio for text '{text[:50]}...': {e}")


async def generate_audio_files():
    """Processes all text queries to generate a suite of audio files and a mapping manifest."""
    try:
        if os.path.exists(OUTPUT_DIR):
            shutil.rmtree(OUTPUT_DIR)
        os.makedirs(OUTPUT_DIR)

        with open(INPUT_FILE, 'r') as f:
            data = json.load(f)

        client = texttospeech.TextToSpeechClient()
        tasks = []
        audio_mappings = []
        
        for i, query_obj in enumerate(data['queries']):
            original_query = query_obj['original_query']
            restatements = query_obj['restatements']
            # Using original query directory naming scheme
            query_dir = os.path.join(OUTPUT_DIR, f"query_{i+1:02d}")
            
            mapping_entry = {
                "original_query": original_query,
                "audio_files": {"original": None, "restatements": []}
            }
            
            expected_function = None
            if query_obj.get('trigger_function'):
                expected_function = {"name": query_obj["function_name"], "args": query_obj["function_args"]}

            # Process original query
            voice_config = random.choice(VOICE_CONFIGS)
            # Using original file naming scheme
            output_filename = f"original_{voice_config['dialect']}_{voice_config['name'].split('-')[-1]}.wav"
            output_path = os.path.join(query_dir, output_filename)
            tasks.append(_synthesize_audio_for_text(original_query, voice_config, output_path, client))
            mapping_entry["audio_files"]["original"] = {"path": output_path, "voice": voice_config["name"], "expected_function": expected_function}

            # Process restatements
            for j, restatement in enumerate(restatements):
                voice_config = random.choice(VOICE_CONFIGS)
                output_filename = f"restatement_{j+1:02d}_{voice_config['dialect']}_{voice_config['name'].split('-')[-1]}.wav"
                output_path = os.path.join(query_dir, output_filename)
                tasks.append(_synthesize_audio_for_text(restatement, voice_config, output_path, client))
                mapping_entry["audio_files"]["restatements"].append({"text": restatement, "path": output_path, "voice": voice_config["name"], "expected_function": expected_function})
            
            audio_mappings.append(mapping_entry)
        
        logger.info(f"Generating {len(tasks)} audio files...")
        await asyncio.gather(*tasks)

        with open(MAPPING_FILE, 'w') as f:
            json.dump({"audio_mappings": audio_mappings}, f, indent=2)

        # if os.path.exists(INPUT_FILE):
        #     os.remove(INPUT_FILE)

    except FileNotFoundError:
        logger.error(f"Intermediate file '{INPUT_FILE}' not found. Ensure restater runs first.")
        raise
    except Exception as e:
        logger.error(f"Error generating audio files: {e}")
        raise