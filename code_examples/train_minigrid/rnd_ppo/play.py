#!/usr/bin/env python3

import gymnasium as gym
from minigrid.utils.window import Window
from minigrid.wrappers import ImgObsWrapper, RGBImgPartialObsWrapper
from pytorchrl.envs.minigrid.minigrid_env_factory import minigrid_train_env_factory
from code_examples.train_minigrid.rnd_ppo.train_rebel import get_args


def redraw(window, img):
    window.show_img(img)


def reset(env, window, seed=None):
    env.reset(seed=seed)

    if hasattr(env, "mission"):
        print("Mission: %s" % env.mission)
        window.set_caption(env.mission)

    img = env.get_frame()

    redraw(window, img)


def step(env, window, action):
    obs, reward, terminated, info = env.step(action)
    print(f"step={env.step_count}, reward={reward:.2f}")

    if terminated:
        print("terminated!")
        reset(env, window)
    else:
        img = env.get_frame()
        redraw(window, img)


def key_handler(env, window, event):
    print("pressed", event.key)

    if event.key == "escape":
        window.close()
        return

    if event.key == "backspace":
        reset(env, window)
        return

    if event.key == "left":
        step(env, window, env.actions.left)
        return
    if event.key == "right":
        step(env, window, env.actions.right)
        return
    if event.key == "up":
        step(env, window, env.actions.forward)
        return

    # Spacebar
    if event.key == "t":
        step(env, window, env.actions.toggle)
        return
    if event.key == "k":
        step(env, window, env.actions.pickup)
        return
    if event.key == "d":
        step(env, window, env.actions.drop)
        return

    if event.key == "enter":
        step(env, window, env.actions.done)
        return


if __name__ == "__main__":

    args = get_args()
    env = minigrid_train_env_factory(env_id=args.env_id)

    window = Window("minigrid - MiniGrid-DeceivingRewards-v0")
    window.reg_key_handler(lambda event: key_handler(env, window, event))

    seed = None
    reset(env, window, seed)

    # Blocking event loop
    window.show(block=True)
