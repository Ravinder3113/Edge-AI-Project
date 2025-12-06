
## Cyber Chatbot Demo

This small project demonstrates a toy RL environment (simulator.py) representing a simple 'user awareness' state,
a RAG retriever (rag_index.py) using sentence-transformers + faiss, a backend stub to load a PPO policy, and a training
script (train_ppo.py) that trains a PPO policy using Stable-Baselines3 (Gymnasium-compatible).

Notes for running locally (Windows/VSCode):
  1. Create a clean conda env (recommended name: rlproj) with Python 3.9 or 3.10.
  2. Install packages: pip install -r requirements.txt
  3. To train: python train_ppo.py  (may take time; reduce timesteps for demo)
  4. To run demo notebook: open project_demo.ipynb in Jupyter/VSCode and run cells.
