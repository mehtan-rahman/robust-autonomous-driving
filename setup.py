# setup.py
"""
Setup script for Robust Autonomous Driving Decision Making project.
Handles dependencies and environment setup.
"""

import sys
import os
from pathlib import Path


def setup_environment():
    """Set up the required dependencies and paths for the project"""
    # Install required dependencies
    os.system("pip install -q numpy==2.0.0 protobuf>=5.31.0 docopt")
    os.system("pip install --user git+https://github.com/eleurent/highway-env")

    # Fix sys.path to include --user installs
    sys.path.append("/root/.local/lib/python3.11/site-packages")

    # Clone rl-agents repo
    os.system("cd /content && rm -rf rl-agents")
    os.system("cd /content && git clone --depth=1 https://github.com/eleurent/rl-agents.git")

    # Patch logger.py
    logger_path = Path("/content/rl-agents/rl_agents/trainer/logger.py")
    logger_text = logger_path.read_text()
    logger_text = logger_text.replace("gym_level=gym.logger.INFO", "gym_level=__import__('logging').INFO")
    logger_lines = [line for line in logger_text.splitlines() if "gym.logger.set" not in line]
    logger_path.write_text("\n".join(logger_lines))

    # Patch evaluation.py
    eval_path = Path("/content/rl-agents/rl_agents/trainer/evaluation.py")
    eval_text = eval_path.read_text()
    eval_text = eval_text.replace("np.infty", "np.inf")
    eval_text = eval_text.replace(
        "from gymnasium.wrappers import RecordVideo, RecordEpisodeStatistics, capped_cubic_video_schedule",
        "from gymnasium.wrappers import RecordVideo, RecordEpisodeStatistics"
    )

    inject_fn = """
def capped_cubic_video_schedule(episode_id):
    return True
"""
    eval_lines = eval_text.splitlines()
    for i, line in enumerate(eval_lines):
        if not line.strip().startswith("import") and not line.strip().startswith("from"):
            inject_index = i
            break
    eval_lines = eval_lines[:inject_index] + [inject_fn.strip()] + eval_lines[inject_index:]
    eval_path.write_text("\n".join(eval_lines))

    # Patch rl-agents to avoid using tensorboardX
    eval_code = eval_path.read_text()
    eval_code = eval_code.replace("from tensorboardX import SummaryWriter", "SummaryWriter = lambda *a, **kw: None")
    eval_code = eval_code.replace("self.writer = SummaryWriter(log_dir=str(self.result_path))", "self.writer = None")
    eval_path.write_text(eval_code)
    print("tensorboardX disabled – agent will run without logging.")

    # Set up video display tools
    os.system("pip install -q pyvirtualdisplay")
    os.system("apt-get -y install xvfb ffmpeg")
    os.system("cd /content && git clone https://github.com/Farama-Foundation/HighwayEnv.git 2> /dev/null")

    # Add paths to rl-agents and scripts
    sys.path.append("/content/rl-agents")
    sys.path.append("/content/HighwayEnv/scripts")

    # Set random seeds for reproducibility
    import random
    import numpy as np
    import torch

    SEED = 42
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    print("Environment setup complete!")


if __name__ == "__main__":
    setup_environment()
    