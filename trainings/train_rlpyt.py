from pathlib import Path

from environments.rlpyt_env import Rlpyt_env, AaaiTrajInfo, PytConfig
from models.frap import Frap
from rlpyt.agents.dqn.dqn_agent import DqnAgent
from rlpyt.algos.dqn.dqn import DQN
from rlpyt.runners.async_rl import AsyncRl, AsyncRlEval
from rlpyt.runners.minibatch_rl import MinibatchRl, MinibatchRlEval
from rlpyt.samplers.async_.gpu_sampler import AsyncGpuSampler
from rlpyt.samplers.parallel.cpu.collectors import CpuWaitResetCollector
from rlpyt.samplers.parallel.gpu.collectors import GpuWaitResetCollector, GpuEvalCollector
from rlpyt.samplers.parallel.gpu.sampler import GpuSampler
from rlpyt.samplers.serial.sampler import SerialSampler
from rlpyt.utils.launching.affinity import make_affinity
from rlpyt.utils.logging.context import logger_context
from settings import JSONS_FOLDER, PROJECT_ROOT


def build_and_train(game="aaai_multi", run_ID=0):
    # Change these inputs to match local machine and desired parallelism.
    affinity = make_affinity(
        run_slot=0,
        n_cpu_core=8,  # Use 16 cores across all experiments.
        n_gpu=1,  # Use 8 gpus across all experiments.
        sample_gpu_per_run=1,
        async_sample=True,
        optim_sample_share_gpu=True
        # hyperthread_offset=24,  # If machine has 24 cores.
        # n_socket=2,  # Presume CPU socket affinity to lower/upper half GPUs.
        # gpu_per_run=2,  # How many GPUs to parallelize one run across.
        # cpu_per_run=1,
    )

    train_conf = PytConfig([
        Path(JSONS_FOLDER, 'configs', '2v2', 'all_equal.json'),
        Path(JSONS_FOLDER, 'configs', '2v2', 'more_horizontally.json'),
        Path(JSONS_FOLDER, 'configs', '2v2', 'more_vertically.json'),
        Path(JSONS_FOLDER, 'configs', '2v2', 'more_from_west.json'),
        Path(JSONS_FOLDER, 'configs', '2v2', 'more_from_east.json'),
        Path(JSONS_FOLDER, 'configs', '2v2', 'more_from_north.json'),
        Path(JSONS_FOLDER, 'configs', '2v2', 'more_from_south.json'),
    ])

    eval_conf = PytConfig({
        'all_equal': Path(JSONS_FOLDER, 'configs', '2v2', 'all_equal.json'),
        'more_horizontally': Path(JSONS_FOLDER, 'configs', '2v2', 'more_horizontally.json'),
        'more_vertically': Path(JSONS_FOLDER, 'configs', '2v2', 'more_vertically.json'),
        'more_south': Path(JSONS_FOLDER, 'configs', '2v2', 'more_from_south.json'),
        'more_east': Path(JSONS_FOLDER, 'configs', '2v2', 'more_from_east.json')
    })

    sampler = AsyncGpuSampler(
        EnvCls=Rlpyt_env,
        TrajInfoCls=AaaiTrajInfo,
        env_kwargs={
            'pyt_conf': train_conf,
            'max_steps': 1500
        },
        batch_T=1,
        batch_B=8,
        max_decorrelation_steps=100,
        eval_env_kwargs={
            'pyt_conf': eval_conf,
            'max_steps': 1500
        },
        eval_max_steps=10510,
        eval_n_envs=2,
    )
    algo = DQN(
        replay_ratio=1024,
        double_dqn=True,
        prioritized_replay=True,
        min_steps_learn=5000,
        learning_rate=0.0001,
        target_update_tau=1.0,
        target_update_interval=300,
        eps_steps=3e4,
        batch_size=1024,
        pri_alpha=0.6,
        pri_beta_init=0.4,
        pri_beta_final=1.,
        pri_beta_steps=int(5e4),
        replay_size=int(1e6)
    )
    agent = DqnAgent(ModelCls=Frap)
    runner = AsyncRlEval(
        algo=algo,
        agent=agent,
        sampler=sampler,
        log_interval_steps=1000,
        affinity=affinity,
        n_steps=4e4
    )

    config = dict(game=game)
    name = "frap_" + game
    log_dir = Path(PROJECT_ROOT, "saved", "rlpyt", "multi", "frap")

    save_path = Path(log_dir, 'run_{}'.format(run_ID))
    for f in save_path.glob('**/*'):
        print(f)
        f.unlink()

    with logger_context(str(log_dir), run_ID, name, config,
                        snapshot_mode='last', use_summary_writer=True, override_prefix=True):
        runner.train()
