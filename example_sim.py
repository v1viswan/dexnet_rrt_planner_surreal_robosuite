"""
This script shows you how to select gripper for an environment.
This is controlled by gripper_type keyword argument
"""
import numpy as np
import math
import robosuite as suite
import cv2
from robosuite.wrappers import GymWrapper
from mujoco_py import load_model_from_path, MjSim, MjViewer, MjRenderContextOffscreen

def set_camera_birdview(viewer):
    viewer.cam.fixedcamid = 1
    viewer.cam.distance = 1
    viewer.cam.elevation = -90
    viewer.cam.lookat[0] = 0.6
    viewer.cam.lookat[1] = -0.15
    viewer.cam.lookat[2] = 0.8


if __name__ == "__main__":

    # create environment with selected grippers
    # env = GymWrapper(
    #     suite.make(
    #         "SawyerPickPlace",
    #         gripper_type="TwoFingerGripper",
    #         use_camera_obs=False,  # do not use pixel observations
    #         has_offscreen_renderer=False,  # not needed since not using pixel obs
    #         has_renderer=True,  # make sure we can render to the screen
    #         reward_shaping=True,  # use dense rewards
    #         control_freq=100,  # control should happen fast enough so that simulation looks smooth
    #     )
    # )
    '''
    python3 examples/policy.py GQCNN-4.0-PJ --depth_image <depth.npy> --segmask <seg.png> --camera_intr data/calib/phoxi/phoxi.intr
    '''

    padding = 170
    t_padding = 90

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
    print(dir(world.mujoco_arena))
    print(world.mujoco_arena.table_full_size)
    print(world.mujoco_arena.bin_abs)

    print(type(world.sim))
    sim = world.sim
    print(dir(sim))
    viewer = MjRenderContextOffscreen(sim, 0)
    print(dir(viewer.scn))
    set_camera_birdview(viewer)
    width, height = 516, 386
    viewer.render(width, height)
    image = np.asarray(viewer.read_pixels(width, height, depth=False)[:, :, :], dtype=np.uint8)
    depth = np.asarray((viewer.read_pixels(width, height, depth=True)[1]))
    # find center depth, this is a hack
    cdepth = depth[height//2, width//2]
    print(cdepth)
    depth[depth > cdepth] = cdepth
    seg = depth != cdepth
    seg[:, :padding], seg[:, -padding:] = False, False
    seg[:t_padding, :], seg[-t_padding:, :] = False, False
    cv2.imwrite('test_dataset/seg.png', seg * 255)
    # normalize the depth
    depth = (depth - np.min(depth)) / (np.max(depth) - np.min(depth))
    depth[:, :padding], depth[:, -padding:] = 0, 0
    depth[:t_padding, :], depth[-t_padding:, :] = 0, 0
    np.save('test_dataset/depth_0.npy', depth)
    from matplotlib import pyplot as plt 
    cv2.imwrite('test_dataset/depth.png', depth * 255)
    cv2.imwrite('test_dataset/visual.png', image)

    fovy = sim.model.cam_fovy[1]
    print(fovy)
    f = 0.5 * (height + width)/2 / math.tan(fovy * math.pi / 360)
    print("compute camera matrix: ")
    print(np.array(((f, 0, width / 2), (0, f, height / 2), (0, 0, 1))))


    # world.viewer.distance = world.model.extent * 0.5
    # env.mode = 'human'
    # env.viewer.set_camera(0)
    # print(env.camera_names)
    # # run a random policy
    # observation = env.reset()

    # while True:
    #     viewer.render()







        # action = env.action_space.sample()
        # observation, reward, done, info = env.step(action)
        # if done:
        #     print("Episode finished after {} timesteps".format(t + 1))
        #     break

    # # close window
    # env.close()


