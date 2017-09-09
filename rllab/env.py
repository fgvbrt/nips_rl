import os
from rllab.misc import logger
from rllab.envs.base import Env, Step
from rllab.core.serializable import Serializable
from osim.env import RunEnv
from rllab.envs.gym_env import convert_gym_space
from multiprocessing import Process, Queue
from state import State, StateVel, StateVelCentr


class RunEnv2(RunEnv):
    def __init__(self, state_transform, visualize=False, max_obstacles=0):
        super(RunEnv2, self).__init__(visualize, max_obstacles)
        self.state_transform = state_transform

    def reset(self, difficulty=2, seed=None):
        s = super(RunEnv2, self).reset(difficulty=difficulty, seed=seed)
        self.state_transform.reset()
        return self.state_transform.process(s)

    def _step(self, action):
        s, r, t, info = super(RunEnv2, self)._step(action)
        s = self.state_transform.process(s)
        return s, r*100, t, info


def standalone_headless_isolated(pq, cq):
    print('starting headless...',pq,cq)
    try:
        import traceback
        state_transform = StateVel(last_n_bodies=0)
        e = RunEnv2(state_transform)
    except Exception as e:
        print('error on start of standalone')
        traceback.print_exc()
        return

    def floatify(np):
        return [float(np[i]) for i in range(len(np))]

    try:
        while True:
            msg = pq.get()
            # messages should be tuples,
            # msg[0] should be string

            if msg[0] == 'reset':
                o = e.reset(difficulty=2)
                cq.put(floatify(o))
            elif msg[0] == 'step':
                ordi = e.step(msg[1])
                ordi[0] = floatify(ordi[0])
                cq.put(ordi)
            else:
                cq.close()
                pq.close()
                del e
                break
    except Exception as e:
        traceback.print_exc()

    return # end process


class ei: # Environment Instance
    def __init__(self):
        self.pretty('instance creating')
        self.newproc()

    # create a new RunEnv in a new process.
    def newproc(self):
        self.pq, self.cq = Queue(1), Queue(1) # two queue needed

        self.p = Process(
            target = standalone_headless_isolated,
            args=(self.pq, self.cq)
        )
        self.p.daemon = True
        self.p.start()
        return

    # send x to the process
    def send(self,x):
        return self.pq.put(x)

    # receive from the process.
    def recv(self):
        r = self.cq.get()
        return r

    def reset(self):
        self.send(('reset',))
        r = self.recv()
        return r

    def step(self,actions):
        self.send(('step',actions,))
        r = self.recv()
        return r

    def kill(self):
            self.send(('exit',))
            self.pretty('waiting for join()...')

            while 1:
                self.p.join(timeout=5)
                if not self.p.is_alive():
                    break
                else:
                    self.pretty('process is not joining after 5s, still waiting...')
            self.pretty('process joined.')

    def __del__(self):
        self.pretty('__del__')
        self.kill()
        self.pretty('__del__ accomplished.')

    # pretty printing
    def pretty(self,s):
        print(('(ei) {} ').format(self.id)+str(s))


class RunEnvRllab(Env, Serializable):
    def __init__(self, log_dir=None):
        if log_dir is None:
            if logger.get_snapshot_dir() is None:
                logger.log("Warning: skipping Gym environment monitoring since snapshot_dir not configured.")
            else:
                log_dir = os.path.join(logger.get_snapshot_dir(), "gym_log")
        Serializable.quick_init(self, locals())

        #env = RunEnv3(visualize=False)
        #env = RunEnv2(visualize=False)
        #env = RunEnv(visualize=False)
        env = ei()
        self.env = env

        self._observation_space = convert_gym_space(env.observation_space)
        logger.log("observation space: {}".format(self._observation_space))
        self._action_space = convert_gym_space(env.action_space)
        logger.log("action space: {}".format(self._action_space))
        self._horizon = env.timestep_limit
        self._log_dir = log_dir

    @property
    def observation_space(self):
        return self._observation_space

    @property
    def action_space(self):
        return self._action_space

    @property
    def horizon(self):
        return self._horizon

    def reset(self):
        return self.env.reset()

    def step(self, action):
        next_obs, reward, done, info = self.env.step(action)
        return Step(next_obs, reward, done, **info)
