from osim.env import RunEnv
from itertools import chain


def flatten(listOfLists):
    "Flatten one level of nesting"
    return list(chain.from_iterable(listOfLists))


class RunEnv3(RunEnv):
    bodies = ['head', 'torso', 'toes_l', 'toes_r', 'talus_l', 'talus_r']
    jnts = ['hip_r', 'knee_r', 'ankle_r', 'hip_l', 'knee_l', 'ankle_l']

    mass_x = 0
    mass_y = 1
    mass_x_vel = 2

    plevis_x = 5
    pelvis_y = 6

    ninput = 36
    noutput = 18

    def __init__(self, visualize=False, max_obstacles=0):
         super(RunEnv3, self).__init__(visualize, max_obstacles)

    def get_observation(self):
        mass_pos = [self.osim_model.model.calcMassCenterPosition(self.osim_model.state)[i] for i in range(2)]
        mass_vel = [self.osim_model.model.calcMassCenterVelocity(self.osim_model.state)[i] for i in range(2)]

        pelvis_pos = [self.pelvis.getCoordinate(i).getValue(self.osim_model.state) for i in range(3)]
        pelvis_vel = [self.pelvis.getCoordinate(i).getSpeedValue(self.osim_model.state) for i in range(3)]
        #pelvis_pos[1] = pelvis_pos[1] - mass_pos[0]
        #pelvis_vel[1] = pelvis_vel[1] - mass_vel[0]
        pelvis_x = pelvis_pos[1]
        mass_pos[0] = mass_pos[0] - pelvis_x

        body_transforms = [[self.osim_model.get_body(body).getTransformInGround(self.osim_model.state).p()[i] - pelvis_x*(i==0)
                            for i in range(2)
                            ] for body in self.bodies]
        # join angles
        joint_angles = [self.osim_model.get_joint(j).getCoordinate().getValue(self.osim_model.state)
                        for j in self.jnts]
        joint_vel = [self.osim_model.get_joint(j).getCoordinate().getSpeedValue(self.osim_model.state)
                     for j in self.jnts]
        muscles = [self.env_desc['muscles'][self.MUSCLES_PSOAS_L], self.env_desc['muscles'][self.MUSCLES_PSOAS_R]]
        #obstacle = self.next_obstacle()
        self.current_state = mass_pos + mass_vel + pelvis_pos + pelvis_vel + flatten(body_transforms) \
                             + joint_angles + joint_vel + muscles
        return self.current_state

    def is_pelvis_too_low(self):
        return self.current_state[self.pelvis_y] < 0.65

    def compute_reward(self):
        r_vel = self.current_state[self.mass_x_vel]
        y_penalty = max(0, 0.7 - self.current_state[self.pelvis_y])
        return r_vel - y_penalty*200

    @property
    def total_reward(self):
        return self.current_state[self.plevis_x]
