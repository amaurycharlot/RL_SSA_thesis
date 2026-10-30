import os
from pathlib import Path
import argparse

import matplotlib.pyplot as plt
import numpy as np
import imageio.v2 as imageio
from PIL import Image
from scripts.plot_episode import plot_episode as plot_scene

def blend_images(img1, img2, num_blends):
    """
    Create intermediate blended PIL images between img1 and img2.
    num_blends=0 means no transition frames.
    """
    blended = []
    for k in range(1, num_blends + 1):
        alpha = k / (num_blends + 1)
        frame = Image.blend(img1, img2, alpha)
        blended.append(frame)
    return blended


def make_fade_video(
    args,
    output_video="episode.mp4",
    frames_dir="frames",
    transition_frames=5,
    fps=10,
):
    """
    Render one plot per timestep, then create a video with fade transitions.
    """
    frames_path = Path(frames_dir)
    frames_path.mkdir(parents=True, exist_ok=True)

    run_dir = args.run_folder
    rew_path = os.path.join(run_dir, "rewards.npy")
    rewards = np.load(rew_path)  # (T,)
    T = rewards.shape[0]

    # Step 1: render base frames
    image_paths = []
    for t in range(T):  # or use actual episode length
        img_path = Path.joinpath(frames_path, f"frame_{t:04d}.png")
        plot_scene(args, time_step=t, out_path=img_path, legend=False)
        image_paths.append(img_path)

    # Step 2: build full frame sequence with fades
    video_frames = []

    for i in range(len(image_paths) - 1):
        img1 = Image.open(image_paths[i]).convert("RGBA")
        img2 = Image.open(image_paths[i + 1]).convert("RGBA")

        # Add the current frame
        video_frames.append(img1.copy())

        # Add fade frames
        for blended in blend_images(img1, img2, transition_frames):
            video_frames.append(blended)

    # Add final frame
    if image_paths:
        last_img = Image.open(image_paths[-1]).convert("RGBA")
        video_frames.append(last_img.copy())

    # Step 3: write video
    # Convert PIL images to numpy arrays for imageio
    with imageio.get_writer(output_video, fps=fps, codec="libx264") as writer:
        for frame in video_frames:
            writer.append_data(np.array(frame))

    print(f"Video saved to: {output_video}")

# Params: none; returns: None.
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "run_folder",
        type=str,
        help="Path to a saved rollout folder, e.g. rollouts/heuristic_seed1",
    )
    parser.add_argument("--show", action="store_true", help="Do not use this option.") 
    args = parser.parse_args()

    make_fade_video(
        args,
        output_video=args.run_folder + "/episode_video.mp4",
        frames_dir=args.run_folder + "/frames",
        transition_frames=4,  # more = smoother fade
        fps=12,
    )



if __name__ == "__main__":
    main()

"""
if __name__ == "__main__":
    import numpy as np

    # Example fake data: 100 sensor pointings in a 100x100 world
    np.random.seed(0)
    actions = np.random.uniform(0, 100, size=(100, 2))
    world_size = 100

    make_fade_video(
        actions=actions,
        world_size=world_size,
        timesteps=range(50),
        output_video="sensor_fade.mp4",
        frames_dir="frames",
        transition_frames=4,  # more = smoother fade
        fps=12,
    )"""