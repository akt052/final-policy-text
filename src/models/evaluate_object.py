import os
import pickle
import re
from collections import defaultdict

import gymnasium as gym
import minigrid

import torch
import torch.nn as nn


# =====================
# 1. Normalize Instruction
# =====================

def normalize_instruction(instr):

    instr = instr.lower()

    instr = re.sub(
        r"\bgo to a (\w+ \w+)\b",
        r"go to the \1",
        instr
    )

    return instr.strip()


# =====================
# 2. Load Dataset
# =====================

with open("data/demos/gotoobj_seq.pkl", "rb") as f:
    data = pickle.load(f)


# =====================
# 3. Group Dataset
# =====================

grouped = defaultdict(list)

for d in data:

    instr = normalize_instruction(
        d["instruction"]
    )

    for obs, act in zip(
        d["obs_seq"],
        d["act_seq"]
    ):

        grouped[instr].append(
            (obs, act)
        )


# =====================
# 4. Policy Network
# =====================

class PolicyNet(nn.Module):

    def __init__(self, n_actions=7):

        super().__init__()

        self.conv = nn.Sequential(

            nn.Conv2d(
                3,
                32,
                kernel_size=3,
                stride=1,
                padding=1
            ),

            nn.BatchNorm2d(32),

            nn.ReLU(),

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
# 5. Device
# =====================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", device)


# =====================
# 6. Environment
# =====================

env = gym.make(
    "BabyAI-GoToObj-v0",
    render_mode=None
)

env = env.unwrapped


# =====================
# 7. Model Files
# =====================

model_dir = "data/demos/go_to_object"

model_files = [
    f for f in os.listdir(model_dir)
    if f.endswith(".pt")
]

model_files.sort()


# =====================
# 8. Evaluation
# =====================

episodes_per_model = 100

results = {}


for model_file in model_files:

    # ---------------------------------
    # Recover instruction
    # ---------------------------------

    instruction = model_file.replace(".pt", "")
    instruction = instruction.replace("_", " ")

    instruction = normalize_instruction(
        instruction
    )

    print("\n========================")
    print("Evaluating:", instruction)
    print("========================")

    # ---------------------------------
    # Load model
    # ---------------------------------

    model = PolicyNet().to(device)

    model.load_state_dict(
        torch.load(
            os.path.join(
                model_dir,
                model_file
            ),
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

            mission = normalize_instruction(
                env.mission
            )

            if mission == instruction:
                break

        done = False

        steps = 0
        max_steps = 150

        while not done and steps < max_steps:

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

            steps += 1

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
# 9. Final Summary
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