import gym
from rlkit.envs.wrappers import NormalizedBoxEnv
from rlkit.torch.sac.policies import MakeDeterministic
from rlkit.samplers.data_collector import MdpPathCollector
from rlkit.launchers.launcher_util import setup_logger
from rlkit.data_management.env_replay_buffer import EnvReplayBuffer
import pandas as pd
from pandas import Timestamp
import os
import common
from common.trainer import get_sac_model
import rlkit.torch.pytorch_util as ptu
import numpy as np
from datetime import datetime
from rlkit.torch.torch_rl_algorithm import TorchBatchRLAlgorithm

def load_dataset():
    current_folder = os.path.dirname(__file__)
    ret_csv_train = os.path.join(current_folder, './data/investments_returns_train.csv')
    ret_csv_val = os.path.join(current_folder, './data/investments_returns_validation.csv')
    features_csv = os.path.join(current_folder, './data/features.csv')
    df_ret_train = pd.read_csv(ret_csv_train, parse_dates=['Date'], index_col=['Date'])
    df_ret_val = pd.read_csv(ret_csv_val, parse_dates=['Date'], index_col=['Date'])
    df_feature = pd.read_csv(features_csv, parse_dates=['Date'], index_col=['Date'])
    return df_ret_train, df_ret_val, df_feature


gym.envs.register(id='MarketEnv-v0', entry_point='common.market_env:MarketEnv', max_episode_steps=1000)



def my_eval_policy(env, algorithm, epoch, eval_result, output_csv):
    done = False
    policy = model.policy
    while not done:
        actions = policy.get_actions(state)
        state, reward, done, info = env.step(actions)
        print(reward)
        print(info)

def train_model(variant):
    df_ret_train, df_ret_val, df_feature = load_dataset()
    expl_env = NormalizedBoxEnv(gym.make('MarketEnv-v0', returns=df_ret_train, features=df_feature,
                                      trade_freq='weeks', show_info=False, trade_pecentage=0.2))

    eval_env = NormalizedBoxEnv(gym.make('MarketEnv-v0', returns=df_ret_val, features=df_feature,
                                    trade_freq='weeks', show_info=False, trade_pecentage=1.0))

    post_epoch_funcs = []
    M = variant['layer_size']
    trainer = get_sac_model(env=eval_env, hidden_sizes=[M, M])
    policy = trainer.policy
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
    algorithm = TorchBatchRLAlgorithm(
        trainer=trainer,
        exploration_env=expl_env,
        evaluation_env=eval_env,
        exploration_data_collector=expl_path_collector,
        evaluation_data_collector=eval_path_collector,
        replay_buffer=replay_buffer,
        **variant['algorithm_kwargs']
    )

    algorithm.to(ptu.device)
    algorithm.train()


ptu.set_gpu_mode(True)  

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_dir = f"./output/train_out_{timestamp}/"

variant = dict(
    algorithm="SAC",
    version="normal",
    log_dir=log_dir,
    layer_size=256,
    replay_buffer_size=int(1E6),
    algorithm_kwargs=dict(
        num_epochs=2500,
        num_eval_steps_per_epoch=5000,
        num_trains_per_train_loop=10000,
        num_expl_steps_per_train_loop=1000,
        min_num_steps_before_training=1000,
        max_path_length=1000,
        batch_size=256,
    ),
)

setup_logger('name-of-experiment', variant=variant,
                snapshot_mode='gap_and_last', snapshot_gap=20, log_dir=log_dir)

train_model(variant)