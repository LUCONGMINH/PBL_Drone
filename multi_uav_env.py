# FILE: multi_uav_env.py (Đã nâng cấp lên Gymnasium)
import airsim
import numpy as np
import time
import math
import gymnasium as gym
from gymnasium import spaces

class MultiUavEnv(gym.Env):
    def __init__(self, latency_mode=False):
        super(MultiUavEnv, self).__init__()
        self.client = airsim.MultirotorClient()
        self.client.confirmConnection()
        self.drone_names = ["Drone1", "Drone2", "Drone3"]
        self.latency_mode = latency_mode 
        
        # Action space: Mỗi drone có 2 action (V_x, V_y), tổng 3 drone = 6
        self.action_space = spaces.Box(low=-1, high=1, shape=(6,), dtype=np.float32)
        
        # Observation space: Mỗi drone (x, y, z, target_x, target_y, target_z) * 3
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(18,), dtype=np.float32)
        
        self.targets = [np.array([10, 10, -5]), np.array([12, 12, -5]), np.array([8, 8, -5])] 

    def _get_obs(self):
        obs_list = []
        for i, name in enumerate(self.drone_names):
            state = self.client.getMultirotorState(vehicle_name=name)
            pos = state.kinematics_estimated.position
            
            # SIMULATE NETWORK LATENCY
            if self.latency_mode:
                time.sleep(0.005) 

            obs_list.extend([pos.x_val, pos.y_val, pos.z_val])
            obs_list.extend(self.targets[i])
            
        return np.array(obs_list, dtype=np.float32)

    def reset(self, seed=None, options=None):
        # Gymnasium yêu cầu khai báo seed ở reset
        super().reset(seed=seed)
        
        self.client.reset()
        time.sleep(1)
        p1 = airsim.Pose(airsim.Vector3r(0, 0, 0), airsim.to_quaternion(0, 0, 0))
        p2 = airsim.Pose(airsim.Vector3r(4, 0, 0), airsim.to_quaternion(0, 0, 0))
        p3 = airsim.Pose(airsim.Vector3r(-4, 0, 0), airsim.to_quaternion(0, 0, 0))
        
        
        self.client.simSetVehiclePose(p1, True, "Drone1")
        self.client.simSetVehiclePose(p2, True, "Drone2")
        self.client.simSetVehiclePose(p3, True, "Drone3")
        time.sleep(1)
        for name in self.drone_names:
            self.client.enableApiControl(True, name)
            self.client.armDisarm(True, name)
            self.client.takeoffAsync(vehicle_name=name).join()
        
        time.sleep(3) # Ổn định drone
        observation = self._get_obs()
        info = {} # Gymnasium yêu cầu trả về thêm info
        return observation, info

    def step(self, actions):
        scaled_actions = actions * 5.0 
        
        for i, name in enumerate(self.drone_names):
            vx = float(scaled_actions[i*2])
            vy = float(scaled_actions[i*2+1])
            self.client.moveByVelocityZAsync(vx, vy, -5, 0.5, vehicle_name=name)
        
        time.sleep(0.1) 
        
        next_obs = self._get_obs()
        terminated = False
        truncated = False # Gymnasium tách done thành terminated và truncated
        reward = 0
        
        positions = []
        for i in range(3):
            pos = next_obs[i*6 : i*6+3]
            target = next_obs[i*6+3 : i*6+6]
            dist = np.linalg.norm(pos - target)
            reward -= dist * 0.1 
            positions.append(pos)
            
            if dist < 2: 
                reward += 100
                terminated = True

        d1_d2 = np.linalg.norm(positions[0] - positions[1])
        d2_d3 = np.linalg.norm(positions[1] - positions[2])
        d1_d3 = np.linalg.norm(positions[0] - positions[2])
        
        min_dist = min(d1_d2, d2_d3, d1_d3)
        current_altitude = positions[0][2]
        if current_altitude < -1.0:
            if min_dist < 1.0: 
                reward -= 50 
                terminated = True 
                print(f"!!! CRASH TRÊN KHÔNG: Dist={min_dist:.2f}m !!!")
        
        # Nếu đang ở dưới đất mà quá gần nhau thì kệ nó (do lỗi khởi động)
        else:
             pass 

        # Kiểm tra bay quá xa (Safety Fail-safe)
        if np.linalg.norm(positions[0]) > 50:
             terminated = True
             reward -= 100
             print("!!! OUT OF BOUNDS !!!")
        info = {}
        return next_obs, reward, terminated, truncated, info