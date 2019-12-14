"""
This script shows you how to select gripper for an environment.
This is controlled by gripper_type keyword argument
"""
import numpy as np
import math
import robosuite as suite
import cv2
from robosuite.wrappers import GymWrapper, IKWrapper
import robosuite.utils.transform_utils as T

from mujoco_py import load_model_from_path, MjSim, MjViewer, MjRenderContextOffscreen

if __name__ == "__main__":

    '''
    python3 examples/policy.py GQCNN-4.0-PJ --depth_image <depth.npy> --segmask <seg.png> --camera_intr data/calib/phoxi/phoxi.intr
    '''
    world = suite.make(
            "SawyerPickPlace",
            gripper_type="TwoFingerGripper",
            use_camera_obs=False,  # do not use pixel observations
            has_offscreen_renderer=False,  # not needed since not using pixel obs
            has_renderer=True,  # make sure we can render to the screen
            reward_shaping=False,  # use dense rewards
            control_freq=30,  # control should happen fast enough so that simulation looks smooth
        )

    world.mode = 'human'
    world.reset()
    world.viewer.set_camera(camera_id=2)
    world.render()

    for i in range(5000):

        dpos, rotation, grasp = [0, 0, -0.01], T.rotation_matrix(angle=0, direction=[1., 0., 0.])[:3, :3], 0

        current = world._right_hand_orn
        drotation = current.T.dot(rotation)  # relative rotation of desired from current
        dquat = T.mat2quat(drotation)
        grasp = grasp - 1
        print(current)
        print(drotation)
        print(dquat)

        # print(grasp)

        action = np.concatenate([dpos, dquat, [grasp]])
        world.step(action)
        world.render()
        pass

