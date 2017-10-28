import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['THEANO_FLAGS'] = 'device=cpu'
import argparse
import worker
from multiprocessing import Process, Array, Value, cpu_count
from model import build_model
from ctypes import c_float
import numpy as np
from time import sleep
from state import StateVelCentr


def get_args():
    parser = argparse.ArgumentParser(description="Run commands")
    parser.add_argument('--num_workers', default=cpu_count(), type=int, help="Number of workers.")
    parser.add_argument('--gamma', type=float, default=0.9, help="Discount factor for reward.")
    parser.add_argument('--layer_norm', action='store_true', help="Use layer normaliation.")
    parser.add_argument('--learning_rate', type=float, default=3e-4, help="Learning rate")
    parser.add_argument('--max_steps', type=int, default=100000000, help="Number of steps.")
    parser.add_argument('--n_steps', type=int, default=5, help="Number of  rollout steps.")
    parser.add_argument('--weights_save_interval', default=10, type=int, help="Period testing and saving weighs.")
    parser.add_argument('--num_test_episodes', type=int, default=2, help="Number of test episodes.")
    parser.add_argument('--sleep', type=int, default=0, help="Sleep time in seconds before start each worker.")
    return parser.parse_args()


def init_shared_weights(model_params):
    _, _, _, _, params, params_target = build_model(**model_params)

    def get_shared_weights(params):
        weights_shared = []
        for p in params:
            p_shape = p.get_value().shape
            w = Array(c_float, p.get_value().ravel())
            weights_shared.append(np.ctypeslib.as_array(w.get_obj()).reshape(p_shape))
        return weights_shared

    shared_cur = get_shared_weights(params)
    shared_target = get_shared_weights(params_target)

    return shared_cur, shared_target


def main():
    args = get_args()

    state_transform = StateVelCentr(obstacles_mode='standard',
                                    exclude_centr=True,
                                    vel_states=[])
    # build model
    model_params = {
        'state_size': state_transform.state_size,
        'num_act': 18,
        'learning_rate': args.learning_rate,
        'layer_norm': args.layer_norm
    }

    weights_shared_cur, weights_shared_target = init_shared_weights(model_params)
    global_step = Value('i', 0)
    best_reward = Value('f', -1e8)

    # start workers
    target_fn = worker.run
    workers = []
    for i in range(args.num_workers):
        w = Process(target=target_fn,
                    args=(model_params,
                          weights_shared_cur,
                          weights_shared_target,
                          state_transform,
                          global_step,
                          best_reward,
                          args.n_steps,
                          args.max_steps,
                          args.gamma)
                    )
        w.daemon = True
        w.start()
        sleep(args.sleep)
        workers.append(w)

    # end all processes
    for w in workers:
        w.join()


if __name__ == '__main__':
    main()
