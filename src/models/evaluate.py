import os
import pickle
from collections import defaultdict

import gymnasium as gym
import minigrid

import torch
import torch.nn as nn


# =====================
# 1. Load Dataset
# =====================

with open("data/demos/gotolocal_seq.pkl", "rb") as f:
    data = pickle.load(f)


# =====================
# 2. Group Dataset
# =====================

grouped = defaultdict(list)

for d in data:

    instr = d["instruction"]

    for obs, act in zip(d["obs_seq"], d["act_seq"]):

        grouped[instr].append((obs, act))


# =====================
# 3. Updated Model
# =====================

class PolicyNet(nn.Module):

    def __init__(self, n_actions=7):

        super().__init__()

        self.conv = nn.Sequential(

            # 7x7 -> 7x7
            nn.Conv2d(
                3,
                32,
                kernel_size=3,
                stride=1,
                padding=1
            ),

            nn.BatchNorm2d(32),

            nn.ReLU(),

            # 7x7 -> 7x7
            nn.Conv2d(
                32,
                64,
                kernel_size=3,
                stride=1,
                padding=1
            ),

            nn.BatchNorm2d(64),

            nn.ReLU()
        )

        self.fc = nn.Sequential(

            nn.Linear(
                64 * 7 * 7,
                256
            ),

            nn.ReLU(),

            nn.Dropout(0.2),

            nn.Linear(
                256,
                128
            ),

            nn.ReLU(),

            nn.Linear(
                128,
                n_actions
            )
        )

    def forward(self, x):

        # (B,7,7,3) -> (B,3,7,7)
        x = x.permute(0, 3, 1, 2)

        x = self.conv(x)

        x = x.reshape(x.size(0), -1)

        logits = self.fc(x)

        return logits


# =====================
# 4. Device
# =====================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", device)


# =====================
# 5. Environment
# =====================

env = gym.make(
    "BabyAI-GoToLocal-v0",
    render_mode=None
)

env = env.unwrapped


# =====================
# 6. Model Files
# =====================

model_dir = "data/models"

model_files = [
    f for f in os.listdir(model_dir)
    if f.endswith(".pt")
]

model_files.sort()


# =====================
# 7. Evaluation
# =====================

episodes_per_model = 100

results = {}


for model_file in model_files:

    # ---------------------------------
    # Recover instruction
    # ---------------------------------

    instruction = model_file.replace(".pt", "")
    instruction = instruction.replace("_", " ")

    print("\n========================")
    print("Evaluating:", instruction)
    print("========================")

    # ---------------------------------
    # Load model
    # ---------------------------------

    model = PolicyNet().to(device)

    model.load_state_dict(
        torch.load(
            os.path.join(model_dir, model_file),
            map_location=device
        )
    )

    model.eval()

    # =================================
    # OFFLINE ACCURACY
    # =================================

    correct = 0
    total = 0

    pairs = grouped[instruction]

    with torch.no_grad():

        for obs, expert_action in pairs:

            obs = torch.tensor(
                obs,
                dtype=torch.float32
            ).unsqueeze(0) / 255.0

            obs = obs.to(device)

            logits = model(obs)

            pred_action = torch.argmax(
                logits,
                dim=1
            ).item()

            if pred_action == expert_action:
                correct += 1

            total += 1

    accuracy = correct / total

    print(
        f"Offline Accuracy: "
        f"{accuracy * 100:.2f}%"
    )

    # =================================
    # ROLLOUT SUCCESS RATE
    # =================================

    success = 0

    for ep in range(episodes_per_model):

        # reset until matching instruction
        while True:

            obs, _ = env.reset()

            if env.mission == instruction:
                break

        done = False

        while not done:

            img = obs["image"]

            img = torch.tensor(
                img,
                dtype=torch.float32
            ).unsqueeze(0) / 255.0

            img = img.to(device)

            with torch.no_grad():

                logits = model(img)

                action = torch.argmax(
                    logits,
                    dim=1
                ).item()

            obs, reward, terminated, truncated, _ = env.step(action)

            done = terminated or truncated

        if reward > 0:
            success += 1

    success_rate = success / episodes_per_model

    print(
        f"Rollout Success Rate: "
        f"{success_rate * 100:.2f}%"
    )

    # =================================
    # Store Results
    # =================================

    results[instruction] = {
        "accuracy": accuracy,
        "success_rate": success_rate
    }


# =====================
# 8. Final Summary
# =====================

print("\n========================")
print("FINAL RESULTS")
print("========================")

for instr in results:

    acc = results[instr]["accuracy"] * 100
    sr = results[instr]["success_rate"] * 100

    print(
        f"{instr}"
        f"\n  Accuracy      : {acc:.2f}%"
        f"\n  Success Rate  : {sr:.2f}%"
        f"\n"
    )