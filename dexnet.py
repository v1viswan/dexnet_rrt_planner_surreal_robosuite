# -*- coding: utf-8 -*-
"""
Copyright ©2017. The Regents of the University of California (Regents).
All Rights Reserved. Permission to use, copy, modify, and distribute this
software and its documentation for educational, research, and not-for-profit
purposes, without fee and without a signed licensing agreement, is hereby
granted, provided that the above copyright notice, this paragraph and the
following two paragraphs appear in all copies, modifications, and
distributions. Contact The Office of Technology Licensing, UC Berkeley, 2150
Shattuck Avenue, Suite 510, Berkeley, CA 94720-1620, (510) 643-7201,
otl@berkeley.edu,
http://ipira.berkeley.edu/industry-info for commercial licensing opportunities.

IN NO EVENT SHALL REGENTS BE LIABLE TO ANY PARTY FOR DIRECT, INDIRECT, SPECIAL,
INCIDENTAL, OR CONSEQUENTIAL DAMAGES, INCLUDING LOST PROFITS, ARISING OUT OF
THE USE OF THIS SOFTWARE AND ITS DOCUMENTATION, EVEN IF REGENTS HAS BEEN
ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

REGENTS SPECIFICALLY DISCLAIMS ANY WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
PURPOSE. THE SOFTWARE AND ACCOMPANYING DOCUMENTATION, IF ANY, PROVIDED
HEREUNDER IS PROVIDED "AS IS". REGENTS HAS NO OBLIGATION TO PROVIDE
MAINTENANCE, SUPPORT, UPDATES, ENHANCEMENTS, OR MODIFICATIONS.

Displays robust grasps planned using a GQ-CNN grapsing policy on a set of saved
RGB-D images. The default configuration for the standard GQ-CNN policy is
`cfg/examples/cfg/examples/gqcnn_pj.yaml`. The default configuration for the
Fully-Convolutional GQ-CNN policy is `cfg/examples/fc_gqcnn_pj.yaml`.

Author
------
Jeff Mahler & Vishal Satish
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import json
import os
import time

import numpy as np

from autolab_core import YamlConfig, Logger
from perception import (BinaryImage, CameraIntrinsics, ColorImage, DepthImage,
                        RgbdImage)
from visualization import Visualizer2D as vis

from gqcnn.grasping import (RobustGraspingPolicy,
                            CrossEntropyRobustGraspingPolicy, RgbdImageState,
                            FullyConvolutionalGraspingPolicyParallelJaw,
                            FullyConvolutionalGraspingPolicySuction)
from gqcnn.utils import GripperMode


class DexNet(object):

    def __init__(self, model_name='GQCNN-4.0-PJ'):
        self.model_dir = 'models'
        self.model_name = model_name
        self.model_path = os.path.join(self.model_dir, self.model_name)
        self.camera_intr_filename = 'data/calib/primesense/primesense.intr'
        # "cfg/examples/fc_gqcnn_suction.yaml"
        self.config_filename = "cfg/examples/gqcnn_pj.yaml"
        self.fully_conv = False
        self.width, self.height = 640, 480

    def __str__(self):
        return "dexnet imported with model name: {}".format(self.model_name)

    def prepare_dexnet(self):
        # Get configs.
        model_config = json.load(open(os.path.join(self.model_path, "config.json"),
                                      "r"))
        # self.model_config = model_config
        # try:
        #     gqcnn_config = model_config["gqcnn"]
        #     gripper_mode = gqcnn_config["gripper_mode"]
        # except KeyError:
        #     gqcnn_config = model_config["gqcnn_config"]
        #     input_data_mode = gqcnn_config["input_data_mode"]
        #     if input_data_mode == "tf_image":
        #         gripper_mode = GripperMode.LEGACY_PARALLEL_JAW
        #     elif input_data_mode == "tf_image_suction":
        #         gripper_mode = GripperMode.LEGACY_SUCTION
        #     elif input_data_mode == "suction":
        #         gripper_mode = GripperMode.SUCTION
        #     elif input_data_mode == "multi_suction":
        #         gripper_mode = GripperMode.MULTI_SUCTION
        #     elif input_data_mode == "parallel_jaw":
        #         gripper_mode = GripperMode.PARALLEL_JAW
        #     else:
        #         raise ValueError(
        #             "Input data mode {} not supported!".format(input_data_mode))

        # Read config.
        config = YamlConfig(self.config_filename)
        self.inpaint_rescale_factor = config["inpaint_rescale_factor"]
        policy_config = config["policy"]
        self.policy_config = policy_config

        # Make relative paths absolute.
        if "gqcnn_model" in policy_config["metric"]:
            policy_config["metric"]["gqcnn_model"] = self.model_path
            if not os.path.isabs(policy_config["metric"]["gqcnn_model"]):
                policy_config["metric"]["gqcnn_model"] = os.path.join(
                    os.path.dirname(os.path.realpath(__file__)),
                    policy_config["metric"]["gqcnn_model"])

        # Setup sensor.
        self.camera_intr = CameraIntrinsics.load(self.camera_intr_filename)
            # Init policy.
            # Set input sizes for fully-convolutional policy.
        if self.fully_conv:
            policy_config["metric"]["fully_conv_gqcnn_config"][
                "im_height"] = self.height
            policy_config["metric"]["fully_conv_gqcnn_config"][
                "im_width"] = self.width

        if self.fully_conv:
            # TODO(vsatish): We should really be doing this in some factory policy.
            if policy_config["type"] == "fully_conv_suction":
                self.policy = FullyConvolutionalGraspingPolicySuction(policy_config)
            elif policy_config["type"] == "fully_conv_pj":
                self.policy = FullyConvolutionalGraspingPolicyParallelJaw(policy_config)
            else:
                raise ValueError(
                    "Invalid fully-convolutional policy type: {}".format(
                        policy_config["type"]))
        else:
            policy_type = "cem"
            if "type" in policy_config:
                policy_type = policy_config["type"]
            if policy_type == "ranking":
                self.policy = RobustGraspingPolicy(policy_config)
            elif policy_type == "cem":
                self.policy = CrossEntropyRobustGraspingPolicy(policy_config)
            else:
                raise ValueError("Invalid policy type: {}".format(policy_type))

    def get_state(self, depth, segmask):
        # Read images.
        depth_im = DepthImage(depth, frame=self.camera_intr.frame)
        color_im = ColorImage(np.zeros([depth_im.height, depth_im.width,
                                        3]).astype(np.uint8),
                              frame=self.camera_intr.frame)
        segmask = BinaryImage(segmask.astype(np.uint8) * 255,
                              frame=self.camera_intr.frame)

        # Inpaint.
        depth_im = depth_im.inpaint(rescale_factor=self.inpaint_rescale_factor)

        # Create state.
        rgbd_im = RgbdImage.from_color_and_depth(color_im, depth_im)
        state = RgbdImageState(rgbd_im, self.camera_intr, segmask=segmask)
        return state, rgbd_im

    def get_action(self, state):
        action = self.policy(state)
        print(action.grasp.pose())
        print('Action quality: {}'.format(action.q_value))
        print('Action depth: {}'.format(action.grasp.depth))  
        return action

    def visualization(self, action, rgbd_im):
        # Vis final grasp.
        if self.policy_config["vis"]["final_grasp"]:
            vis.figure(size=(10, 10))
            vis.imshow(rgbd_im.depth,
                       vmin=self.policy_config["vis"]["vmin"],
                       vmax=self.policy_config["vis"]["vmax"])
            vis.grasp(action.grasp, scale=2.5, show_center=True, show_axis=True)
            vis.title("Planned grasp at depth {0:.3f}m with Q={1:.3f}".format(
                action.grasp.depth, action.q_value))
            vis.savefig('test_dataset/grasp.png')