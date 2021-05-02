from rlkit.samplers.data_collector import MdpPathCollector
from rlkit.torch.sac.policies import TanhGaussianPolicy, MakeDeterministic
from rlkit.torch.sac.sac import SACTrainer
from rlkit.torch.ddpg.ddpg import DDPGTrainer
from rlkit.torch.td3.td3 import TD3Trainer
from rlkit.torch.networks import ConcatMlp, TanhMlpPolicy
import copy


def get_sac_model(env, hidden_sizes=[256, 256], reward_scale=1):
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
        reward_scale=reward_scale,
        use_automatic_entropy_tuning=True
    )
    return trainer


def get_ddpg_model(env, hidden_sizes=[256, 256]):
    obs_dim = env.observation_space.low.size
    action_dim = env.action_space.low.size
    qf = ConcatMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        hidden_sizes=hidden_sizes
    )
    policy = TanhMlpPolicy(
        input_size=obs_dim,
        output_size=action_dim,
        hidden_sizes=hidden_sizes
    )
    target_qf = copy.deepcopy(qf)
    target_policy = copy.deepcopy(policy)

    trainer = DDPGTrainer(
        qf=qf,
        target_qf=target_qf,
        policy=policy,
        target_policy=target_policy,
        use_soft_update=True,
        tau=1e-2,
        discount=0.99,
        qf_learning_rate=1e-3,
        policy_learning_rate=1e-4
    )
    return trainer


def get_td3_model(env, hidden_sizes=[256, 256],**kwargs):
    obs_dim = env.observation_space.low.size
    action_dim = env.action_space.low.size
    qf1 = ConcatMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        hidden_sizes = hidden_sizes
    )
    qf2 = ConcatMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        hidden_sizes = hidden_sizes
    )
    target_qf1 = ConcatMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        hidden_sizes = hidden_sizes
    )
    target_qf2 = ConcatMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        hidden_sizes = hidden_sizes
    )
    policy = TanhMlpPolicy(
        input_size=obs_dim,
        output_size=action_dim,
        hidden_sizes = hidden_sizes
    )
    target_policy = TanhMlpPolicy(
        input_size=obs_dim,
        output_size=action_dim,
        hidden_sizes = hidden_sizes
    )

    trainer = TD3Trainer(
        policy=policy,
        qf1=qf1,
        qf2=qf2,
        target_qf1=target_qf1,
        target_qf2=target_qf2,
        target_policy=target_policy,
        hidden_sizes=hidden_sizes
    )
    return trainer


def get_model(env,algorithm,reward_scale,**kwargs):
    if(algorithm =='SAC'):
        trainer = get_sac_model(env=env,reward_scale=reward_scale,**kwargs)
    elif (algorithm=='DDPG'):
        trainer = get_ddpg_model(env=env, **kwargs)
    elif (algorithm=='TD3'):
        trainer = get_ddpg_model(env=env, **kwargs)
    return trainer
    
