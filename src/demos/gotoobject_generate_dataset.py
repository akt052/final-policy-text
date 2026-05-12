import pickle
import os
import gymnasium as gym
import numpy as np
import hashlib
import re
from minigrid.utils.baby_ai_bot import BabyAIBot


def normalize_instruction(instr):
    """
    Normalize BabyAI instructions so variants like:
    'go to a red ball'
    become:
    'go to the red ball'
    """
    instr = instr.lower()
    instr = re.sub(r"\bgo to a (\w+ \w+)\b", r"go to the \1", instr)
    return instr.strip()


def get_traj_hash(obs_seq, act_seq):
    obs_bytes = np.array(obs_seq, dtype=np.uint8).tobytes()
    act_bytes = np.array(act_seq, dtype=np.uint8).tobytes()
    return hashlib.sha1(obs_bytes + act_bytes).hexdigest()


def generate_dataset(
    env_name="BabyAI-GoToObj-v0",
    target_per_mission=550,
    save_path="data/demos/gotoobj_seq.pkl",
    max_steps=150
):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    env = gym.make(env_name).unwrapped

    demos = []
    mission_counts = {}
    seen = set()

    while True:

        obs, _ = env.reset()

        instruction = normalize_instruction(env.mission)

        if instruction not in mission_counts:
            mission_counts[instruction] = 0

        # Skip if enough demos collected
        if mission_counts[instruction] >= target_per_mission:
            continue

        bot = BabyAIBot(env)

        obs_seq = []
        act_seq = []

        done = False
        reward = 0

        while not done:

            try:
                action = bot.replan(obs)

            except AssertionError:
                # Bot failed planning
                break

            next_obs, reward, terminated, truncated, _ = env.step(action)

            done = terminated or truncated

            # Store current observation and action
            obs_seq.append(
                np.array(obs["image"], dtype=np.uint8)
            )

            act_seq.append(int(action))

            obs = next_obs

            # Safety cutoff
            if len(obs_seq) > max_steps:
                break

        # Keep only successful trajectories
        if done and reward > 0:

            traj_hash = get_traj_hash(obs_seq, act_seq)

            # Remove duplicate trajectories
            if traj_hash in seen:
                continue

            seen.add(traj_hash)

            demos.append({
                "instruction": instruction,
                "obs_seq": obs_seq,
                "act_seq": act_seq
            })

            mission_counts[instruction] += 1

            total = sum(mission_counts.values())

            print(
                f"{total} | "
                f"{instruction}: "
                f"{mission_counts[instruction]}"
            )

            # Stop condition
            if (
                len(mission_counts) >= 18 and
                all(c >= target_per_mission
                    for c in mission_counts.values())
            ):
                break

    with open(save_path, "wb") as f:
        pickle.dump(demos, f)

    print("\nSaved dataset:", save_path)
    print("Total demos:", len(demos))


if __name__ == "__main__":

    generate_dataset(
        env_name="BabyAI-GoToObj-v0",
        target_per_mission=550,
        save_path="data/demos/gotoobj_seq.pkl"
    )