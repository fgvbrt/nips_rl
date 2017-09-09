import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['THEANO_FLAGS'] = 'device=cpu'

from rllab.algos.trpo import TRPO
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.envs.normalized_env import normalize
from rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy
from rllab.misc.instrument import run_experiment_lite
from env import RunEnvRllab


def run_train(*_):
    env = normalize(RunEnvRllab(), normalize_obs=True, normalize_reward=True)
    policy = GaussianMLPPolicy(
        env_spec=env.spec,
        # The neural network policy should have two hidden layers, each with 32 hidden units.
        hidden_sizes=(100, 50),
        adaptive_std=True,
    )
    baseline = LinearFeatureBaseline(env_spec=env.spec)

    algo = TRPO(
        env=env,
        policy=policy,
        baseline=baseline,
        batch_size=10000,
        max_path_length=1000,
        n_itr=1000,
        discount=0.95,
        step_size=0.05,
        # Uncomment both lines (this and the plot parameter below) to enable plotting
        # plot=True,
    )

    algo.train()


run_experiment_lite(
    run_train,
    # Number of parallel workers for sampling
    n_parallel=12,
    # Only keep the snapshot parameters for the last iteration
    snapshot_mode="last",
    # Specifies the seed for the experiment. If this is not provided, a random seed
    # will be used
    seed=1,
    # plot=True,
)
