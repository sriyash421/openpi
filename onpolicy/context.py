"""Past-attempt context shared by rollouts and training: frame subsampling, tiling, prompt.

The previous attempt enters π0.5 through the unused third image slot (right_wrist_0_rgb) as a 2x4 grid of
8 agentview frames, and its feedback enters the prompt. Kept numpy/PIL-only so it runs in the py3.8 LIBERO client.
"""
import numpy as np
from PIL import Image

N_FRAMES = 8
GRID = (2, 4)  # rows, cols
SIZE = 224  # openpi image resolution
FEEDBACK_MAX_WORDS = 40


def subsample(frames, n=N_FRAMES):
    """n frames uniformly spaced over the attempt, always including the first and the last."""
    idx = np.linspace(0, len(frames) - 1, n).round().astype(int)
    return [frames[i] for i in idx]


def tile(frames):
    """8 HxWx3 uint8 frames -> one 224x224x3 grid (each cell 112x56)."""
    rows, cols = GRID
    ch, cw = SIZE // rows, SIZE // cols
    grid = np.zeros((SIZE, SIZE, 3), np.uint8)
    for k, f in enumerate(frames[: rows * cols]):
        r, c = divmod(k, cols)
        grid[r * ch:(r + 1) * ch, c * cw:(c + 1) * cw] = np.asarray(Image.fromarray(f).resize((cw, ch), Image.BILINEAR))
    return grid


def empty_context():
    return np.zeros((SIZE, SIZE, 3), np.uint8)


def prompt(instruction, feedback=None):
    """Instruction alone for attempt 0 / plain demos; otherwise instruction + capped feedback."""
    if not feedback:
        return instruction
    fb = f"Feedback on previous attempt: {' '.join(feedback.split()[:FEEDBACK_MAX_WORDS])}"
    return f"{instruction}. {fb}" if instruction else fb  # instruction=None: training-time instruction dropout
