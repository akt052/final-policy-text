import os
import re
import imageio

import gymnasium as gym
import minigrid

import torch
import torch.nn as nn


# =====================================
# 1. NORMALIZE INSTRUCTION
# =====================================

def normalize_instruction(instr):

    instr = instr.lower()

    instr = re.sub(
        r"\bgo to a (\w+ \w+)\b",
        r"go to the \1",
        instr
    )

    return instr.strip()


# =====================================
# 2. POLICY NETWORK
# =====================================

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


# =====================================
# 3. DEVICE
# =====================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", device)


# =====================================
# 4. ENVIRONMENT
# =====================================

env = gym.make(
    "BabyAI-GoToObj-v0",
    render_mode="rgb_array"
)

env = env.unwrapped


# =====================================
# 5. MODEL DIRECTORY
# =====================================

model_dir = "data/demos/go_to_object"

model_files = sorted([
    f for f in os.listdir(model_dir)
    if f.endswith(".pt")
])

print("\nFound models:", len(model_files))


# =====================================
# 6. GENERATE GIF
# =====================================

all_frames = []

for model_file in model_files:

    # ---------------------------------
    # RECOVER INSTRUCTION
    # ---------------------------------

    instruction = model_file.replace(".pt", "")

    instruction = instruction.replace("_", " ")

    instruction = normalize_instruction(
        instruction
    )

    print("\n====================")
    print("Instruction:", instruction)
    print("====================")

    # ---------------------------------
    # RESET UNTIL MATCHING MISSION
    # ---------------------------------

    while True:

        obs, _ = env.reset()

        mission = normalize_instruction(
            env.mission
        )

        if mission == instruction:
            break

    print("Environment Mission:", mission)

    # ---------------------------------
    # LOAD MODEL
    # ---------------------------------

    model_path = os.path.join(
        model_dir,
        model_file
    )

    model = PolicyNet().to(device)

    model.load_state_dict(
        torch.load(
            model_path,
            map_location=device
        )
    )

    model.eval()

    # =================================
    # ROLLOUT
    # =================================

    done = False

    reward = 0

    steps = 0

    max_steps = 150

    while not done and steps < max_steps:

        # -----------------------------
        # SAVE FRAME
        # -----------------------------

        frame = env.get_frame(
            highlight=True,
            tile_size=32
        )

        all_frames.append(frame)

        # -----------------------------
        # MODEL ACTION
        # -----------------------------

        img = torch.tensor(
            obs["image"],
            dtype=torch.float32
        ).unsqueeze(0) / 255.0

        img = img.to(device)

        with torch.no_grad():

            logits = model(img)

            action = torch.argmax(
                logits,
                dim=1
            ).item()

        # -----------------------------
        # STEP ENVIRONMENT
        # -----------------------------

        obs, reward, terminated, truncated, _ = env.step(action)

        done = terminated or truncated

        steps += 1

    # ---------------------------------
    # FINAL FRAME
    # ---------------------------------

    frame = env.get_frame(
        highlight=True,
        tile_size=32
    )

    all_frames.append(frame)

    print("Reward:", reward)

    print("Success:", reward > 0)

    print("Steps:", steps)

    # ---------------------------------
    # SMALL PAUSE BETWEEN EPISODES
    # ---------------------------------

    for _ in range(100):

        all_frames.append(frame)


# =====================================
# 7. SAVE FINAL GIF
# =====================================

output_path = "all_policies.gif"

imageio.mimsave(
    output_path,
    all_frames,
    fps=5,
    loop=0
)

print("\n================================")
print("Saved GIF:", output_path)
print("================================")