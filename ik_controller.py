import robosuite
from robosuite.wrappers import IKWrapper
import numpy as np
import os
import pickle

import robosuite.utils.transform_utils as T

class robosuite_IKmover():
    def __init__(self, env):
        self.env_init = env
        self.env = IKWrapper(env)
        self.grasp = -1

    def move_orient(self, env, q1):

        q0 = env.observation_spec()['eef_quat']
        x_target = env.observation_spec()['eef_pos']

        fraction_size = 50
        for i in range(fraction_size):
            q_target = T.quat_slerp(q0,q1,fraction=(i+1)/fraction_size)
            steps = 0
            lamda = 0.01

            current = env._right_hand_orn
            target_rotation = T.quat2mat(q_target)
            drotation = current.T.dot(target_rotation)

            while (np.linalg.norm(T.mat2euler(drotation)) > 0.01):

                current = env._right_hand_orn
                target_rotation = T.quat2mat(q_target)
                drotation = current.T.dot(target_rotation)
                dquat = T.mat2quat(drotation)
                x_current = env.observation_spec()['eef_pos']
                d_pos = np.clip( (x_target - x_current)*lamda, -0.05, 0.05)
                d_pos = [0,0,0]

                action = np.concatenate((d_pos,dquat,[self.grasp]))
                env.step(action)
                env.render()
                steps += 1

                if (steps > 20):
                    break
        return

    def move_xy(self, env, x_target, target_rotation):

    #     target_rotation = np.array([[-1., 0., 0.], [0., 1., 0.], [0., 0., -1.]])

        x_current = env.observation_spec()['eef_pos'][:2]
        steps = 0
        lamda = 0.1
        while (np.linalg.norm(x_target - x_current) > 0.00001):

            if (np.linalg.norm(x_target - x_current) < 0.01):
                lamda = lamda*1
            else :
                lamda = 0.1

            x_current = env.observation_spec()['eef_pos'][:2]
            current = env._right_hand_orn
            drotation = current.T.dot(target_rotation)
            dquat = T.mat2quat(drotation)
            d_pos = (x_target - x_current)*lamda

            action = np.concatenate((d_pos,[0],dquat,[self.grasp]))

            env.step(action)
            env.render()

            for i in range(4):
                # Now do action for zero xyz change so that bot stabilizes
                current = env._right_hand_orn
                drotation = current.T.dot(target_rotation)
                dquat = T.mat2quat(drotation)
                action = np.concatenate(([0,0,0],dquat,[self.grasp]))
                env.step(action)
                env.render()
            steps +=1


            if (steps>500):
                break

        return

    def move_z(self, env, z_target, target_rotation):

    #     target_rotation = np.array([[-1., 0., 0.], [0., 1., 0.], [0., 0., -1.]])

        x_current = env.observation_spec()['eef_pos']
        x_target = np.copy(x_current)
        x_target[2] = z_target
        steps = 0
        lamda = 0.1
        while (np.linalg.norm(x_target - x_current) > 0.00001):

            if (np.linalg.norm(x_target - x_current) < 0.01):
                lamda = lamda*1.01
            else :
                lamda = 0.1

            x_current = env.observation_spec()['eef_pos']
            current = env._right_hand_orn
            drotation = current.T.dot(target_rotation)
            dquat = T.mat2quat(drotation)
            d_pos = (x_target - x_current)*lamda

            action = np.concatenate((d_pos,dquat,[self.grasp]))

            env.step(action)
            env.render()

            for i in range(4):
                # Now do action for zero xyz change so that bot stabilizes
                current = env._right_hand_orn
                drotation = current.T.dot(target_rotation)
                dquat = T.mat2quat(drotation)
                action = np.concatenate(([0,0,0],dquat,[self.grasp]))
                env.step(action)
                env.render()
            steps +=1


            if (steps>500):
                break

        return

    def move_xyz(self, env, x_target, target_rotation):

    #     target_rotation = np.array([[-1., 0., 0.], [0., 1., 0.], [0., 0., -1.]])

        x_current = env.observation_spec()['eef_pos']
        steps = 0
        lamda = 0.1
        while (np.linalg.norm(x_target - x_current) > 0.00001):

            if (np.linalg.norm(x_target - x_current) < 0.01):
                lamda = lamda*1.01
            else :
                lamda = 0.1

            x_current = env.observation_spec()['eef_pos']
            current = env._right_hand_orn
            drotation = current.T.dot(target_rotation)
            dquat = T.mat2quat(drotation)
            d_pos = (x_target - x_current)*lamda

            action = np.concatenate((d_pos,dquat,[self.grasp]))

            env.step(action)
            env.render()

            for i in range(4):
                # Now do action for zero xyz change so that bot stabilizes
                current = env._right_hand_orn
                drotation = current.T.dot(target_rotation)
                dquat = T.mat2quat(drotation)
                action = np.concatenate(([0,0,0],dquat,[self.grasp]))
                env.step(action)
                env.render()
            steps +=1


            if (steps>500):
                break

        return
    
    def move(self, pos, rotation):
        
        # First align the gripper angle
        q1 = T.mat2quat(rotation)
        self.move_orient(self.env, q1)
        
        # Now move to x-y coordinate given by the pos
        self.move_xy(self.env, pos[:2], rotation)
        
        # Now move to x-y-z target position
        self.move_xyz(self.env, pos, rotation)
    
    def lift(self, pos):
        print("Lifting")
        rotation = T.quat2mat(self.env.observation_spec()['eef_quat'])
        self.grasp = 1
        self.move_z(self.env, z_target = pos[2], target_rotation=rotation)
        self.move_xyz(self.env, x_target = pos, target_rotation=rotation)