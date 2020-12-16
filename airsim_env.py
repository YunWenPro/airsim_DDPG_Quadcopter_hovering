import time
import numpy as np
import airsim
import cv2
from collections import namedtuple

# define destination
object_pos = [0, 0, 5]
# define boundary
outZ = [-10, 10]
outY = [-10, 10]
outX = [-10, 10]

vel_base = 0.6


class windENV():
    def __init__(self):
        # connect to the airsim environment
        self.cl = airsim.MultirotorClient()
        self.cl.confirmConnection()
        self.action_space = namedtuple('action_space', ['shape', 'high', 'low'])
        self.action_space.shape = [4]  # four motor PWMs
        self.action_space.high = [0.4] * 4
        self.action_space.low = [-0.6] * 4

        obs_dim = 9
        self.observation_space = namedtuple('observation_space', ['shape'])
        self.observation_space.shape = [obs_dim]
        self.duration = 0.2

    def reset(self):
        self.cl.reset()
        self.cl.enableApiControl(True)
        self.cl.armDisarm(True)

        # take off
        self.cl.simPause(False)

        self.cl.moveToPositionAsync(0, 0, 5, 20).join()
        self.cl.hoverAsync().join()
        self.add_wind()
        self.cl.simPause(True)
        state = self.getState()

        return state

    def step(self, action):
        self.cl.simPause(False)

        self.cl.moveByMotorPWMsAsync(action[0] + vel_base, action[1] + vel_base, action[2] + vel_base, action[3] + vel_base, duration=self.duration).join()
        start = time.time()
        collision_count = 0
        has_collided = False
        while time.time() - start < self.duration:
            time.sleep(0.05)
            collided = self.cl.simGetCollisionInfo().has_collided

            if collided:
                collision_count += 1
            if collision_count > 10:
                has_collided = True
                break

        self.cl.simPause(True)

        state = self.cl.getMultirotorState().kinematics_estimated
        pos = state.position
        linear_v = state.linear_velocity
        angle_v = state.angular_velocity
        linear_v = np.array([linear_v.x_val, linear_v.y_val, linear_v.z_val], dtype=np.float32)
        angle_v = np.array([angle_v.x_val, angle_v.y_val, angle_v.z_val], dtype=np.float32)

        stop = pos.y_val < outY[0] or pos.y_val > outY[1] or pos.z_val < outZ[0] or \
               pos.z_val > outZ[1] or pos.x_val < outX[0] or pos.x_val > outX[1] or \
               has_collided
        pos = np.array([pos.x_val, pos.y_val, pos.z_val], dtype=np.float32)
        reward = self.comput_reward(linear_v, angle_v, pos, stop, action)
        state = self.getState()
        info = []
        info.append(state)
        info.append(reward)

        return state, reward, stop, info

    def comput_reward(self, linear_v, angle_v, position, collision, action):
        reward = 0.0 if collision else 1.0
        velocity_norm = np.linalg.norm(linear_v)
        angular_velocity_norm = np.linalg.norm(angle_v)
        action_norm = np.linalg.norm(action)
        alpha, beta = 1.0, 1.0
        reward -= alpha * velocity_norm + beta * angular_velocity_norm + action_norm
        hovering_range, in_range_r, out_range_r = 10, 10, -20
        z_move = abs(position[2]) + abs(position[1] - object_pos[1]) + abs(position[0] - object_pos[0])
        if z_move < hovering_range:
            reward += in_range_r
        else:
            reward += max(out_range_r, hovering_range - z_move)

        return reward

    def getState(self):
        k_e = self.cl.getMultirotorState().kinematics_estimated
        # k_e_a_a = k_e.angular_acceleration
        # k_e_a_v = k_e.angular_velocity
        # k_e_l_a = k_e.linear_acceleration
        k_e_l_v = k_e.linear_velocity
        k_e_q = k_e.orientation
        pitch, roll, yaw = airsim.utils.to_eularian_angles(k_e_q)
        k_e_p = k_e.position
        # state = np.array([k_e_a_a.x_val, k_e_a_a.y_val, k_e_a_a.z_val,
        #                   k_e_a_v.x_val, k_e_a_v.y_val, k_e_a_v.z_val,
        #                   k_e_l_a.x_val, k_e_l_a.y_val, k_e_l_a.z_val,
        #                   k_e_l_v.x_val, k_e_l_v.y_val, k_e_l_v.z_val,
        #                   pitch, roll, yaw,
        #                   k_e_p.x_val, k_e_p.y_val, k_e_p.z_val, ])
        state = np.array([np.float(k_e_l_v.x_val), np.float(k_e_l_v.y_val), np.float(k_e_l_v.z_val),
                          np.float(pitch), np.float(roll), np.float(yaw),
                          np.float(k_e_p.x_val), np.float(k_e_p.y_val), np.float(k_e_p.z_val)])

        return state

    def disconnect(self):
        self.cl.enableApiControl(False)
        self.cl.armDisarm(False)
        print('Disconnected')
