import gym
import os
import rlkit.torch.pytorch_util as ptu
from rlkit.data_management.env_replay_buffer import EnvReplayBuffer
from rlkit.envs.wrappers import NormalizedBoxEnv
from rlkit.launchers.launcher_util import setup_logger
from rlkit.samplers.data_collector import MdpPathCollector
from rlkit.torch.sac.policies import TanhGaussianPolicy, MakeDeterministic
from rlkit.torch.sac.sac import SACTrainer
from rlkit.torch.networks import ConcatMlp
from rlkit.torch.torch_rl_algorithm import TorchBatchRLAlgorithm
from continuous_mountain_car_10_11 import Continuous_MountainCarEnv
from datetime import datetime
import pandas as pd
import numpy as np


def experiment(variant):
    #expl_env = NormalizedBoxEnv(gym.make('MountainCarContinuous-v0'))
    #eval_env = NormalizedBoxEnv(gym.make('MountainCarContinuous-v0'))
    gym.envs.register(id='MyGymEnv-v0', entry_point='my_gym_env:MyGymEnv', max_episode_steps=1000)   
    expl_env = NormalizedBoxEnv(gym.make('MyGymEnv-v0'))
    eval_env = NormalizedBoxEnv(gym.make('MyGymEnv-v0'))

    obs_dim = expl_env.observation_space.low.size
    action_dim = eval_env.action_space.low.size
    post_epoch_funcs =[]
    M = variant['layer_size']
    qf1 = ConcatMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        hidden_sizes=[M, M],
    )
    qf2 = ConcatMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        hidden_sizes=[M, M],
    )
    target_qf1 = ConcatMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        hidden_sizes=[M, M],
    )
    target_qf2 = ConcatMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        hidden_sizes=[M, M],
    )
    policy = TanhGaussianPolicy(
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_sizes=[M, M],
    )
    eval_policy = MakeDeterministic(policy)
    eval_path_collector = MdpPathCollector(
        eval_env,
        eval_policy,
    )
    expl_path_collector = MdpPathCollector(
        expl_env,
        policy,
    )
    replay_buffer = EnvReplayBuffer(
        variant['replay_buffer_size'],
        expl_env,
    )
    sac_trainer = SACTrainer(
        env=eval_env,
        policy=policy,
        qf1=qf1,
        qf2=qf2,
        target_qf1=target_qf1,
        target_qf2=target_qf2,
        **variant['trainer_kwargs']
    )

    algorithm = TorchBatchRLAlgorithm(
        trainer=sac_trainer,
        exploration_env=expl_env,
        evaluation_env=eval_env,
        exploration_data_collector=expl_path_collector,
        evaluation_data_collector=eval_path_collector,
        replay_buffer=replay_buffer,
        **variant['algorithm_kwargs']
    )
    columns =['Epoch','mean','std']
    result_df = pd.DataFrame(columns =columns)
    result_csv = os.path.join(variant['log_dir'],'result.csv')
    def post_epoch_func(self, epoch):
        print(f'-------------post_epoch_func start-------------')
        print(result_csv)
        nonlocal result_df
        eval_count = 100
        policy = self.trainer.networks[0]
        env = NormalizedBoxEnv(gym.make('MyGymEnv-v0'))

        steps_counts=[]
        for _ in range(eval_count):
            done = False
            state = env.reset()
            steps=0
            while not done:
                steps+=1
                actions = policy.get_actions(state)
                state, reward, done, _ = env.step(actions)
            steps_counts.append(steps)
            df_new = pd.DataFrame ([[epoch,np.mean(steps_counts),np.std(steps_counts)]],
                                    columns =columns)
        
        result_df = pd.concat([result_df,df_new])
        result_df.set_index('Epoch')
        print(result_df)
        result_df.to_csv(result_csv)
        print(f'-------------post_epoch_func done-------------')        

    algorithm.post_epoch_funcs=[post_epoch_func,]
    algorithm.to(ptu.device)
    algorithm.train()

if __name__ == "__main__":
    # noinspection PyTypeChecker


    timestamp =datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = f"./rlkit_out_{timestamp}/"
    variant = dict(
        algorithm="SAC",
        version="normal",
        log_dir=log_dir,
        layer_size=256,
        replay_buffer_size=int(1E6),
        algorithm_kwargs=dict(
            num_epochs=1500,
            num_eval_steps_per_epoch=5000,
            num_trains_per_train_loop=100, #1000,
            num_expl_steps_per_train_loop=1000,
            min_num_steps_before_training=1000,
            max_path_length=1000,
            batch_size=256,
        ),
        trainer_kwargs=dict(
            discount=0.99,
            soft_target_tau=5e-3,
            target_update_period=1,
            policy_lr=3E-4,
            qf_lr=3E-4,
            reward_scale=1,
            use_automatic_entropy_tuning=True,
        ),
    )


    setup_logger('name-of-experiment', variant=variant,snapshot_mode='gap_and_last',log_dir=log_dir)
    ptu.set_gpu_mode(True)  # optionally set the GPU (default=False)
    experiment(variant)