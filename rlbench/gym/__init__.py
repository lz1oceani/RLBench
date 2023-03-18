from gym.envs.registration import register
import rlbench.backend.task as task
import os
from rlbench.utils import name_to_task_class
from rlbench.gym.rlbench_env import RLBenchEnv

TASKS = [t for t in os.listdir(task.TASKS_PATH)
         if t != '__init__.py' and t.endswith('.py')]

for task_file in TASKS:
    task_name = task_file.split('.py')[0]
    task_class = name_to_task_class(task_name)
    register(
        id='%s-v0' % task_name,
        entry_point='rlbench.gym:RLBenchEnv',
        max_episode_steps=20,
        kwargs={
            'task_class': task_class,
        }
    )
    register(
        id='%s-state-v0' % task_name,
        entry_point='rlbench.gym:RLBenchEnv',
        max_episode_steps=20,
        kwargs={
            'task_class': task_class,
            'obs_mode': 'state'
        }
    )
    register(
        id='%s-rgb-v0' % task_name,
        entry_point='rlbench.gym:RLBenchEnv',
        max_episode_steps=20,
        kwargs={
            'task_class': task_class,
            'obs_mode': 'rgb'
        }
    )
    register(
        id='%s-rgbd-v0' % task_name,
        entry_point='rlbench.gym:RLBenchEnv',
        max_episode_steps=20,
        kwargs={
            'task_class': task_class,
            'obs_mode': 'rgbd'
        }
    )
    register(
        id='%s-pcd-v0' % task_name,
        entry_point='rlbench.gym:RLBenchEnv',
        max_episode_steps=20,
        kwargs={
            'task_class': task_class,
            'obs_mode': 'pcd'
        }
    )
