from __future__ import annotations

from pathlib import Path

import imageio.v2 as imageio
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from exo_rl.envs.dynamics import forward_kinematics


def figure_to_rgb(fig: plt.Figure) -> np.ndarray:
    fig.canvas.draw()
    width, height = fig.canvas.get_width_height()
    buffer = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    rgba = buffer.reshape(height, width, 4)
    return rgba[:, :, :3].copy()


def arm_points(q: np.ndarray, link_lengths: np.ndarray) -> np.ndarray:
    q1, q2 = q
    l1, l2 = link_lengths
    elbow = np.array([l1 * np.cos(q1), l1 * np.sin(q1)])
    wrist = forward_kinematics(q, link_lengths)
    return np.vstack([np.zeros(2), elbow, wrist])


def make_even_frame(frame: np.ndarray) -> np.ndarray:
    height, width = frame.shape[:2]
    pad_h = height % 2
    pad_w = width % 2
    if pad_h == 0 and pad_w == 0:
        return frame
    return np.pad(frame, ((0, pad_h), (0, pad_w), (0, 0)), mode="edge")


def save_frames_mp4_gif(frames: list[np.ndarray], mp4_path: Path, gif_path: Path, fps: int = 20) -> None:
    mp4_path.parent.mkdir(parents=True, exist_ok=True)
    gif_path.parent.mkdir(parents=True, exist_ok=True)
    even_frames = [make_even_frame(frame) for frame in frames]
    imageio.mimsave(mp4_path, even_frames, fps=fps, macro_block_size=1)
    imageio.mimsave(gif_path, even_frames, fps=fps)
