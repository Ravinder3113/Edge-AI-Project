# backend_stub.py
import numpy as np
try:
    from stable_baselines3 import PPO
except Exception as e:
    # We'll still allow the file to be imported even if SB3 isn't installed.
    PPO = None

ACTION_TEMPLATES = {
    0: "Short tip: {rag}",
    1: "Detailed explanation: {rag}",
    2: "Quick quiz: Which is safer — clicking an emailed link or visiting your account directly?",
    3: "Checklist: 1) Hover link 2) Check sender 3) Report if suspicious.",
    4: "Would you like to report this to your security team?",
    5: "Please escalate to SOC with the email headers attached.",
    6: "Thanks — stay safe!",
}

def load_policy(path='ppo_cyber.zip'):
    if PPO is None:
        raise ImportError('stable-baselines3 is not installed in this environment.')
    model = PPO.load(path)
    return model

def policy_infer(model, state):
    # accept both batched and single-state
    arr = np.asarray(state, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    action, _ = model.predict(arr, deterministic=True)
    return int(action[0]) if hasattr(action, '__len__') else int(action)
