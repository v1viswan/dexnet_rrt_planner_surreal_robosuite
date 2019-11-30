"""
This script shows you how to select gripper for an environment.
This is controlled by gripper_type keyword argument

demo script: 
python3 examples/policy.py GQCNN-4.0-PJ --depth_image <depth.npy> --segmask <seg.png> --camera_intr data/calib/phoxi/phoxi.intr
"""
import numpy as np
import math
import cv2
import robosuite as suite
from robosuite.wrappers import GymWrapper
from mujoco_py import load_model_from_path, MjSim, MjViewer, MjRenderContextOffscreen
from dexnet import DexNet

def set_camera_birdview(viewer):
    '''
    world.mujoco_arena.bin_abs => the center of the table is (0.6, -0.15, 0.8)
    the camera is positioned directly on top of the table looking downward, with camera calibration:
    - with default fov: 45
    [[544.40515832   0.         258.        ]
     [  0.         544.40515832 193.        ]
     [  0.           0.           1.        ]]
    the camera position in the world coordinate is:
    T: [0.6, -0.15, 0.8 + 0.9 (distance)]
    took the front view camera and rotate around y-axis:
    

    looking from robot orientation with azimuth = -180, rotation around z-axis
    '''
    viewer.cam.fixedcamid = 1
    viewer.cam.distance = 0.8
    viewer.cam.elevation = -90
    viewer.cam.lookat[0] = 0.6
    viewer.cam.lookat[1] = -0.15
    viewer.cam.lookat[2] = 0.8
    viewer.cam.azimuth = -180

if __name__ == "__main__":

    # configuration for depth and segmask rig
    top_pad, left_pad = 145, 100
    width, height = 516, 386

    # create simulation environment
    world = suite.make(
            "SawyerPickPlace",
            gripper_type="TwoFingerGripper",
            use_camera_obs=False,  # do not use pixel observations
            has_offscreen_renderer=False,  # not needed since not using pixel obs
            has_renderer=False,  # make sure we can render to the screen
            reward_shaping=True,  # use dense rewards
            control_freq=30,  # control should happen fast enough so that simulation looks smooth
        )



    world.mode = 'human'
    world.reset()
    sim = world.sim

    viewer = MjRenderContextOffscreen(sim, 0)
    set_camera_birdview(viewer)
    viewer.render(width, height)
    image = np.asarray(viewer.read_pixels(width, height, depth=False)[:, :, :], dtype=np.uint8)
    depth = np.asarray((viewer.read_pixels(width, height, depth=True)[1]))    

    cdepth = max(depth[height//2, width//2], depth[left_pad, top_pad], depth[-left_pad, -top_pad])
    print(cdepth)
    depth[depth > cdepth] = cdepth
    seg = depth != cdepth
    depth[:, :top_pad], depth[:, -top_pad:] = cdepth, cdepth
    depth[:left_pad, :], depth[-left_pad:, :] = cdepth, cdepth
    depth = (depth - np.min(depth)) / (np.max(depth) - np.min(depth))

    seg[:, :top_pad], seg[:, -top_pad:] = False, False
    seg[:left_pad, :], seg[-left_pad:, :] = False, False
    
    dexnet = DexNet()
    dexnet.prepare_dexnet()
    print(dexnet)
    state, rgbd_im = dexnet.get_state(depth, seg)
    action = dexnet.get_action(state)
    dexnet.visualization(action, rgbd_im)


    # visualization
    cv2.imwrite('test_dataset/seg.png', seg * 255)
    # normalize the depth
    np.save('test_dataset/depth_0.npy', depth)
    cv2.imwrite('test_dataset/depth.png', depth * 255)
    cv2.imwrite('test_dataset/visual.png', image)

