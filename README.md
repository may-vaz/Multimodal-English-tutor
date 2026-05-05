# Multimodal-English-tutor

**Overview**

This project is a multimodal AI-powered English tutor that evaluates and improves a learner’s language skills through structured, stage-based interaction.

It combines LLMs, speech recognition, and adaptive task flows to simulate a real teacher-like feedback loop — focusing on grammar accuracy, correction, and guided improvement.

**Tech Stack**
LLMs: Ollama (llama3, llava)
Orchestration: LangGraph
Speech-to-Text: faster-whisper
Frontend: Streamlit
State Management: In-memory checkpointing

**Installation**
1. pip install -r requirements.txt
2. ollama pull llama3
3. ollama pull llava
4. streamlit run app.py

**Architecture**
Workflow Engine (LangGraph)
Manages stage progression, retries, and task flow using a state-based graph.
LLM Evaluation Layer
llama3 → text evaluation
llava → image-based evaluation
Returns strict JSON for consistent grading.
Speech Processing
Converts voice input to text using faster-whisper (CPU optimized).
Frontend (Streamlit)
Handles user interaction, transcription display, and feedback rendering.

**Key Features**
3-stage adaptive learning pipeline
 * Image Description (vision + language)
 * Sentence Correction (grammar precision)
 * Free Speaking (fluency + structure)
Retry-based learning system
 * Immediate feedback on mistakes
 * Second attempt with variation
 * Forced progression with correction
Multimodal input
 * Voice input (speech → text via Whisper)
 * Text input (direct evaluation)
Strict AI evaluation
 * Uses LLMs as deterministic graders
 * Enforces grammar correctness (not lenient scoring)
 * Structured JSON outputs for reliability

