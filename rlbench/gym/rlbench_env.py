from typing import Union, Dict, Tuple

import gym
import numpy as np
from gym import spaces
from pyrep.const import RenderMode
from pyrep.objects.dummy import Dummy
from pyrep.objects.vision_sensor import VisionSensor

from rlbench.action_modes.action_mode import MoveArmThenGripper
from rlbench.action_modes.arm_action_modes import JointVelocity
from rlbench.action_modes.gripper_action_modes import Discrete
from rlbench.environment import Environment
from rlbench.observation_config import ObservationConfig


class RLBenchEnv(gym.Env):
    """An gym wrapper for RLBench."""

    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(self, task_class, observation_mode='state',
                 render_mode: Union[None, str] = None, n_points=512, 
                 shaped_rewards=False):
        self._observation_mode = observation_mode
        self._render_mode = render_mode
        self.n_points = n_points
        obs_config = ObservationConfig()
        if observation_mode == 'state':
            obs_config.set_all_high_dim(False)
            obs_config.set_all_low_dim(True)
        elif observation_mode in ["rgb", "rgbd", "pcd"]:
            obs_config.set_all(True)
        else:
            raise ValueError(
                'Unrecognised observation_mode: %s.' % observation_mode)

        action_mode = MoveArmThenGripper(JointVelocity(), Discrete())
        self.env = Environment(
            action_mode, obs_config=obs_config, headless=True, shaped_rewards=shaped_rewards)
        self.env.launch()
        self.task = self.env.get_task(task_class)

        _, obs = self.task.reset()

        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=self.env.action_shape)

        if observation_mode == 'state':
            self.observation_space = spaces.Box(
                low=-np.inf, high=np.inf, shape=obs.get_low_dim_data().shape)
        elif observation_mode in ['rgb', "rgbd"]:
            ret = {
                "state": spaces.Box(
                    low=-np.inf, high=np.inf,
                    shape=obs.get_low_dim_data().shape),
                "rgb": spaces.Box(
                    low=0, high=1, shape=obs.left_shoulder_rgb.shape[:-1] + (12,))
            }
            if observation_mode == "rgbd":
                ret["depth"] = spaces.Box(
                    low=0, high=1, shape=obs.left_shoulder_depth.shape[:-1] + (4, ))
                
            self.observation_space = spaces.Dict(ret)
        elif observation_mode == 'pcd':
            self.observation_space = spaces.Dict({
                "state": spaces.Box(
                    low=-np.inf, high=np.inf,
                    shape=obs.get_low_dim_data().shape),
                "xyz": spaces.Box(
                    low=0, high=1, shape=(self.n_points, 3)),
                "rgb": spaces.Box(
                    low=0, high=1, shape=(self.n_points, 3)),
                })

        if render_mode is not None:
            # Add the camera to the scene
            cam_placeholder = Dummy('cam_cinematic_placeholder')
            self._gym_cam = VisionSensor.create([640, 360])
            self._gym_cam.set_pose(cam_placeholder.get_pose())
            if render_mode == 'human':
                self._gym_cam.set_render_mode(RenderMode.OPENGL3_WINDOWED)
            else:
                self._gym_cam.set_render_mode(RenderMode.OPENGL3)

    def _extract_obs(self, obs) -> Dict[str, np.ndarray]:
        # from IPython import embed; embed()
        if self._observation_mode == 'state':
            return obs.get_low_dim_data()
        elif self._observation_mode in ['rgb', "rgbd"]:
            rgb = np.concatenate([
                    obs.left_shoulder_rgb, 
                    obs.right_shoulder_rgb,
                    obs.wrist_rgb,
                    obs.front_rgb
                ], axis=-1) / 255.0
        
            ret = {
                "state": obs.get_low_dim_data(),
                "rgb": rgb,
            }
            if self._observation_mode == "rgbd":
                depth = np.concatenate([
                        obs.left_shoulder_depth, 
                        obs.right_shoulder_depth, 
                        obs.wrist_depth, 
                        obs.front_depth, 
                    ], axis=-1)
                ret["depth"] = depth
            return ret
        elif self._observation_mode == 'pcd':
            # from pyrl.utils.visualization import visualize_pcd, plot_show_image
            
            xyz = np.concatenate([obs.left_shoulder_point_cloud, obs.right_shoulder_point_cloud, obs.wrist_point_cloud, obs.front_point_cloud], axis=0).reshape(-1, 3)
            
            mask = np.int32(np.concatenate([obs.left_shoulder_mask, obs.right_shoulder_mask, obs.wrist_mask, obs.front_mask], axis=0).reshape(-1))
            
            not_include_mask = np.array([10, 48, 52, 55])
            sign = np.all(mask[:, None] != not_include_mask, axis=-1)
            
            # print(mask.min(), mask.max())
            
            depth = np.concatenate([
                    obs.left_shoulder_depth, 
                    obs.right_shoulder_depth, 
                    obs.wrist_depth, 
                    obs.front_depth, 
                ], axis=0).reshape(-1)
            
            
            rgb = np.concatenate([obs.left_shoulder_rgb, obs.right_shoulder_rgb, obs.wrist_rgb, obs.front_rgb], axis=0).reshape(-1, 3) / 255.0
            
            xyz = xyz[sign]
            rgb = rgb[sign]
            mask = mask[sign]
            
            idx = np.arange(xyz.shape[0])
            np.random.shuffle(idx)
            idx = idx[:self.n_points]
            xyz = xyz[idx]
            rgb = rgb[idx]
            mask = mask[idx]
            
            return {
                "state": obs.get_low_dim_data(),
                "xyz": xyz,
                "rgb": rgb
            }

    def render(self, mode='human') -> Union[None, np.ndarray]:
        if mode != self._render_mode:
            raise ValueError(
                'The render mode must match the render mode selected in the '
                'constructor. \nI.e. if you want "human" render mode, then '
                'create the env by calling: '
                'gym.make("reach_target-state-v0", render_mode="human").\n'
                'You passed in mode %s, but expected %s.' % (
                    mode, self._render_mode))
        if mode == 'rgb_array':
            frame = self._gym_cam.capture_rgb()
            frame = np.clip((frame * 255.).astype(np.uint8), 0, 255)
            return frame

    def reset(self) -> Dict[str, np.ndarray]:
        descriptions, obs = self.task.reset()
        del descriptions  # Not used.
        return self._extract_obs(obs)

    def step(self, action) -> Tuple[Dict[str, np.ndarray], float, bool, dict]:
        obs, reward, terminate = self.task.step(action)
        return self._extract_obs(obs), reward, terminate, {}

    def close(self) -> None:
        self.env.shutdown()
