import collections

import numpy as np
from dm_control import mjcf
from dm_control import composer
from dm_control.composer.observation import observable
from dm_control.locomotion.arenas import floors
from dm_control import viewer


REWARDING_Z = 0.7
FILEPATH = "tripod_mini.xml"
TIME_LIMIT = float('inf')  # Use float('inf') for no episode termination (not great for learning)
NUM_SUBSTEPS = 10  # The number of physics substeps per control timestep. The default is 25.


class Creature(composer.Entity):
    def _build(self):
        self._model = mjcf.from_path(FILEPATH)

    def _build_observables(self):
        return CreatureObservables(self)

    @property
    def mjcf_model(self):
        return self._model

    @property
    def actuators(self):
        return tuple(self._model.find_all('actuator'))


# Add simple observable features for joint angles and velocities.
class CreatureObservables(composer.Observables):

    @composer.observable
    def joint_positions(self):
        all_joints = self._entity.mjcf_model.find_all('joint')
        return observable.MJCFFeature('qpos', all_joints)

    @composer.observable
    def joint_velocities(self):
        all_joints = self._entity.mjcf_model.find_all('joint')
        return observable.MJCFFeature('qvel', all_joints)


class Task(composer.Task):

    def __init__(self, creature):
        self._creature = creature
        self._arena = floors.Floor()

        self._arena.add_free_entity(self._creature)
        self._arena.mjcf_model.worldbody.add('light', pos=(0, 0, 4))

        # Configure and enable observables
        self._creature.observables.joint_positions.enabled = True
        self._creature.observables.joint_velocities.enabled = True
        self._task_observables = {}

        for obs in self._task_observables.values():
            obs.enabled = True

        self.control_timestep = NUM_SUBSTEPS * 0.002

    @property
    def root_entity(self):
        return self._arena

    @property
    def task_observables(self):
        return self._task_observables

    def initialize_episode_mjcf(self, random_state):
        pass

    def initialize_episode(self, physics, random_state):
        self._creature.set_pose(physics, position=(0, 0, 0.2))

    def get_reward(self, physics):
        z = physics.named.data.xpos['tripod/base'][2]
        reward = -abs(REWARDING_Z - z)  # Max reward is 0, but it is unreachable
        # print(round(z, 3), round(reward, 3))
        return reward

    def get_observation(self, physics):
        """Returns an observation of the (bounded) physics state."""
        obs = collections.OrderedDict()
        obs['position'] = physics.bounded_position()
        obs['velocity'] = physics.velocity()
        return obs


creature = Creature()
task = Task(creature)
env = composer.Environment(task, random_state=np.random.RandomState(42), time_limit=TIME_LIMIT)
env.reset()


if __name__ == '__main__':
    viewer.launch(env)
