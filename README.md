# On-Device Cybersecurity Conversational Assistant

## Overview
This project implements an intelligent, adaptive cybersecurity conversational assistant powered by:

- Reinforcement Learning (PPO)
- Retrieval-Augmented Generation (RAG)
- On-device deployable architecture

The assistant teaches cybersecurity best practices—phishing detection, safe browsing habits, OTP fraud prevention—through adaptive dialogue. Unlike rule-based chatbots, this system learns how to teach by modeling user behaviour and selecting optimal instructional strategies in real time.


## Key Features

### Adaptive Conversational Intelligence
Uses PPO reinforcement learning to personalize responses and evolve teaching strategies.

### Retrieval-Augmented Generation
Ensures factual accuracy by retrieving cybersecurity knowledge from a FAISS vector store using SentenceTransformers.

### User Behaviour Simulation
A Gymnasium environment models novices, intermediate users, and experts with awareness states between 0 and 1.

### Multi-Action Dialogue Framework
Seven high-level actions:
- Short Tip
- Detailed Explanation
- Quiz
- Warning/Escalation
- Conversation Closure

### On-Device Friendly
Supports quantization and lightweight models suitable for Jetson or mobile deployment.


## System Architecture

User → CyberEnv (State) → PPO Agent → Action → RAG Retrieval → Response Generator → User


## Methodology Summary

### 1. User Behaviour Simulation
- Awareness score ∈ [0, 1]
- 32-dimensional state vector
- Personas: novice, intermediate, expert
- Awareness changes based on agent actions
- Reward = awareness gain

### 2. Reinforcement Learning Model (PPO)
- Action space: 7 pedagogical strategies
- Training: 20,000 timesteps
- Objective: maximize long-term awareness

### 3. Retrieval-Augmented Generation
- Embedding model: all-MiniLM-L6-v2
- Vector index: FAISS

### 4. Conversational Response Layer
Templates ensure structured, natural responses.


## Installation

git clone https://github.com/Ravinder3113/On-Device-Chatbot-Response-Generation.git

cd On-Device-Chatbot-Response-Generation

pip install -r requirements.txt


## Training the PPO Agent

python train_ppo.py


## Running the Chatbot

Run project_demo.ipynb


## Future Work

- Use real user interaction data
- Expand knowledge base
- Add speech-based input/output
- Deploy on NVIDIA Jetson or smartphones

