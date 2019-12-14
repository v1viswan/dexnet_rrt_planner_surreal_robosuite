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
from mujoco_py import load_model_from_path, MjSim, MjViewer, MjRenderContextOffscreen, MjRenderContext
from dexnet import DexNet
from ik_controller import robosuite_IKmover
from robosuite.environments import SawyerLift_vj

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

if __name__ == "__main__":

    # configuration for depth and segmask rig
    top_pad, left_pad = 110, 50
    width, height = 516, 386

    # create simulation environment
#     world = suite.make(
#             "SawyerPickPlace",
#             gripper_type="TwoFingerGripper",
#             use_camera_obs=False,  # do not use pixel observations
#             has_offscreen_renderer=False,  # not needed since not using pixel obs
#             has_renderer=True,  # make sure we can render to the screen
#             reward_shaping=False,  # use dense rewards
#             control_freq=100,  # control should happen fast enough so that simulation looks smooth
#             ignore_done=True
#         )
    world = SawyerLift_vj(
        ignore_done=True,
        gripper_type="TwoFingerGripper",
        use_camera_obs=False,
        has_offscreen_renderer=False,
        has_renderer=True,
        camera_name=None,
        control_freq=100)

    world.reset()
    world.mode = 'human'
    ik_wrapper = robosuite_IKmover(world)
    sim = world.sim

    viewer = MjRenderContextOffscreen(sim, 0)
    set_camera_birdview(viewer)
    viewer.render(width, height)
    image = np.asarray(viewer.read_pixels(width, height, depth=False)[:, :, :], dtype=np.uint8)
    depth = np.asarray((viewer.read_pixels(width, height, depth=True)[1]))    
# , depth[left_pad, top_pad], depth[-left_pad, -top_pad]
    cdepth = depth[height//2, width//2]
    print(cdepth)
    depth[depth > cdepth] = cdepth
    seg = depth != cdepth
    depth[:, :top_pad], depth[:, -top_pad:] = cdepth, cdepth
    depth[:left_pad, :], depth[-left_pad:, :] = cdepth, cdepth

    offset, scale = np.min(depth), np.max(depth) - np.min(depth)
    depth = (depth - np.min(depth)) / (np.max(depth) - np.min(depth))

    seg[:, :top_pad], seg[:, -top_pad:] = False, False
    seg[:left_pad, :], seg[-left_pad:, :] = False, False
    
    dexnet = DexNet()
    dexnet.prepare_dexnet()
    print("dexnet prepared")
    state, rgbd_im = dexnet.get_state(depth, seg)
    action = dexnet.get_action(state)




    '''
    get depth of the action and the x, y
    apply inverse from camera coordinate to the world coordinate
    '''
    dexnet.visualization(action, rgbd_im, offset, scale)

    action.grasp.depth = action.grasp.depth * scale + offset
    rigid_transform = action.grasp.pose()
    print('center: {}, {}'.format(action.grasp.center.x, action.grasp.center.y))
    print('depth: {}'.format(action.grasp.depth))
    print('rot: {}'.format(rigid_transform.rotation))
    print('tra: {}'.format(rigid_transform.translation))
    print('camera intr: {}'.format(dir(action.grasp.camera_intr)))
    print('proj matrix: {}'.format(action.grasp.camera_intr.proj_matrix))
    print('other attr: {}'.format(dir(action.grasp)))
    # gripper_pos = (rigid_transform.rotation.T @ rigid_transform.translation).flatten()
    # gripper_pos[2] = (1-gripper_pos[2]) * scale + offset
    # gripper_pos[0] += 0.6
    # gripper_pos[1] += -0.15


    gripper_pos = rigid_transform.translation.flatten() + np.array([0.6, -0.15, -0.11])
    print('gripper position: {}'.format(gripper_pos))
    # world_coord = invCamR @ (gripper_pos + invCamT).reshape(3,1)
    # world_coord[2] += 1
    # print(world_coord)


    # print('x: {}, y: {}'.format(action.grasp.center.x, action.grasp.center.y))
    # print(action.grasp.depth)



    # # visualization
    cv2.imwrite('test_dataset/seg.png', seg * 255)
    # # normalize the depth
    np.save('test_dataset/depth_0.npy', depth)
    cv2.imwrite('test_dataset/depth.png', depth * 255)
    cv2.imwrite('test_dataset/visual.png', image)

    initial_pos = np.copy(world.observation_spec()['eef_pos'])
    ik_wrapper.move(gripper_pos, rigid_transform.rotation)
    ik_wrapper.lift(initial_pos)

    # viewer = MjViewer(sim)

    # while True:
    #     # viewer.add_marker(pos=np.array([0.6, -0.15, 0.8 + 0.8]),
    #     #                   label=str('camera position'))
    #     viewer.add_marker(size=np.ones(3) * 0.01,
    #                       pos=gripper_pos.flatten(),
    #                       label=str('target position'))
    #     viewer.render()

