import sys
sys.path.append(".")
sys.path.append("..") 
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

def get_sac_model(env, hidden_sizes=[256, 256]):
    obs_dim = env.observation_space.low.size
    action_dim = env.action_space.low.size
    qf1 = ConcatMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        hidden_sizes=hidden_sizes
    )
    qf2 = ConcatMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        hidden_sizes=hidden_sizes,
    )
    target_qf1 = ConcatMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        hidden_sizes=hidden_sizes,
    )
    target_qf2 = ConcatMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        hidden_sizes=hidden_sizes,
    )
    policy = TanhGaussianPolicy(
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_sizes=hidden_sizes,
    )

    trainer = SACTrainer(
        env=env,
        policy=policy,
        qf1=qf1,
        qf2=qf2,
        target_qf1=target_qf1,
        target_qf2=target_qf2,
        discount=0.99,
        soft_target_tau=5e-3,
        target_update_period=1,
        policy_lr=3E-4,
        qf_lr=3E-4,
        reward_scale=1,
        use_automatic_entropy_tuning=True
    )
    return trainer


def get_env():
    return NormalizedBoxEnv(gym.make('TestGymEnv-v0'))


def my_eval_policy(env, algorithm, epoch, eval_result, output_csv):
    columns = eval_result.columns
    policy = algorithm.trainer.policy
    eval_count = 100
    steps_counts = []
    for _ in range(eval_count):
        done = False
        state = env.reset()
        steps = 0
        while not done:
            steps += 1
            actions = policy.get_actions(state)
            state, reward, done, _ = env.step(actions)
        steps_counts.append(steps)
    df_new = pd.DataFrame([[epoch, np.mean(steps_counts), np.std(steps_counts)]],
                          columns=columns)
    eval_result = pd.concat([eval_result, df_new])
    eval_result.set_index('Epoch')
    print(eval_result)
    eval_result.to_csv(output_csv)
    return eval_result

def experiment(variant):

    expl_env = get_env()
    eval_env = get_env()

    obs_dim = expl_env.observation_space.low.size
    action_dim = eval_env.action_space.low.size
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
    columns = ['Epoch', 'mean', 'std']
    eval_result = pd.DataFrame(columns=columns)
    eval_output_csv = os.path.join(variant['log_dir'], 'eval_result.csv')

    def post_epoch_func(self, epoch):
        nonlocal eval_result
        nonlocal policy
        print(f'-------------post_epoch_func start-------------')
        eval_result = my_eval_policy(
            env=get_env(), algorithm=self, epoch=epoch, eval_result=eval_result, output_csv=eval_output_csv,
        )
        print(f'-------------post_epoch_func done-------------')

    algorithm.post_epoch_funcs = [post_epoch_func, ]
    algorithm.to(ptu.device)
    algorithm.train()


if __name__ == "__main__":
    # noinspection PyTypeChecker
    gym.envs.register(
        id='TestGymEnv-v0', entry_point='common.test_gym_env:TestGymEnv', max_episode_steps=1000)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = f"./rlkit_out_{timestamp}/"
    variant = dict(
        algorithm="SAC",
        version="normal",
        log_dir=log_dir,
        layer_size=256,
        replay_buffer_size=int(1E6),
        algorithm_kwargs=dict(
            num_epochs=2500,
            num_eval_steps_per_epoch=5000,
            num_trains_per_train_loop=100,  # 1000,
            num_expl_steps_per_train_loop=1000,
            min_num_steps_before_training=1000,
            max_path_length=1000,
            batch_size=256,
        ),
    )

    setup_logger('name-of-experiment', variant=variant,
                 snapshot_mode='gap_and_last', snapshot_gap=20, log_dir=log_dir)
    ptu.set_gpu_mode(True)  # optionally set the GPU (default=False)
    experiment(variant)
