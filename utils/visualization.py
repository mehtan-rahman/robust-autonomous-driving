# utils/visualization.py
"""
Visualization utilities for the autonomous driving project.
"""

import os
import inspect
from base64 import b64encode
from IPython.display import HTML, display


def show_latest_video(path="videos"):
    """Display the latest video in the specified folder"""
    files = sorted(os.listdir(path))
    for file in files:
        if file.endswith(".mp4"):
            mp4_path = os.path.join(path, file)
            break
    mp4 = open(mp4_path, 'rb').read()
    b64 = b64encode(mp4).decode()
    return HTML(f'<video width=640 controls><source src="data:video/mp4;base64,{b64}" type="video/mp4"></video>')


def record_and_show_successful_episode(make_env_fn, planner, video_folder="videos",
                                       max_steps=200, max_tries=10):
    """
    Runs an agent in the environment until it completes an episode without
    crashing and records the video.

    Args:
        make_env_fn: Function that creates the environment
        planner: The planner/agent to use for decision making
        video_folder: Folder to save the video in
        max_steps: Maximum number of steps per episode
        max_tries: Maximum number of attempts to complete an episode
    """
    # Check if the planner needs a `step` argument
    needs_step = 'step' in inspect.signature(planner.act).parameters

    env_vid = make_env_fn()
    env_vid = record_videos(env_vid, video_folder=video_folder)

    for attempt in range(1, max_tries + 1):
        obs = env_vid.reset()[0]
        done = False
        steps = 0

        while not done and steps < max_steps:
            if needs_step:
                action = planner.act(obs, steps)
            else:
                action = planner.act(obs)

            obs, r, term, trunc, _ = env_vid.step(action)
            done = term or trunc
            steps += 1

        if steps >= max_steps:
            print(f"[Attempt {attempt}] Success: survived {steps} steps.")
            break
        else:
            print(f"[Attempt {attempt}] Crashed at step {steps}, retrying...")

    env_vid.close()

    # Find and display the latest video
    video_files = sorted([f for f in os.listdir(video_folder) if f.endswith(".mp4")],
                         key=lambda f: os.path.getmtime(os.path.join(video_folder, f)))
    if not video_files:
        print(f"No video found in '{video_folder}'.")
        return

    latest = os.path.join(video_folder, video_files[-1])
    with open(latest, "rb") as f:
        encoded = b64encode(f.read()).decode()
    display(HTML(
        f'<video width=640 controls>'
        f'<source src="data:video/mp4;base64,{encoded}" type="video/mp4">'
        f'</video>'
    ))