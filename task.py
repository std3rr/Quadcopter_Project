import numpy as np
import math
#from physics_sim import PhysicsSim

class Hover():
    """Task (environment) that defines the goal and provides feedback to the agent."""
    def __init__(self, init_pose=None, init_velocities=None,
        init_angle_velocities=None, runtime=5., target_pos=None):
        """Initialize a Task object.
        Params
        ======
            init_pose: initial position of the quadcopter in (x,y,z) dimensions and the Euler angles
            init_velocities: initial velocity of the quadcopter in (x,y,z) dimensions
            init_angle_velocities: initial radians/second for each of the three Euler angles
            runtime: time limit for each episode
            target_pos: target/goal (x,y,z) position for the agent
        """
        # Simulation
        self.sim = PhysicsSim(init_pose, init_velocities, init_angle_velocities, runtime)
        self.action_repeat = 3

        self.state_size = self.action_repeat * (len(self.sim.pose))
        self.action_low = 600
        self.action_high = 900
        self.action_size = 4
        self.prev_distance = None

        # Goal
        self.target_pos = target_pos if target_pos is not None else np.array([0., 0., 10.])

    def get_reward(self):
        """Reward function. Been exploring alot here."""

        # D = √[(x₂ - x₁)² + (y₂ - y₁)² +(z₂ - z₁)²]
        distance = np.linalg.norm(self.sim.pose[:3] - self.target_pos)
        velocity = np.linalg.norm(self.sim.v+0.0)

        #distance = abs(sum(self.sim.pose[:3] - self.target_pos))
        #velocity = abs(sum(self.sim.v))
        #if(distance == nan):

        reward  = 100. - (distance * velocity)
        #reward  = 100. - distance
        if reward <= 0.0:
            reward = 0.0
        return reward
        if math.isnan(distance) or math.isnan(velocity):
            return 0.0

        print(distance, velocity)
        #distance = distance * 0.003
        dist_reward = 1. - (distance**0.4)
        #print("drew:",dist_reward)
        #print(np.max([velocity,0.1]))
        avel = (1. - max([velocity,0.1]))
        adist = (1./max([distance,0.1]))
        #print("avel:",avel)
        #print("adist:",adist)
        vel_discount = avel ** adist
        #print("vdis:",vel_discount)
        reward = np.max([vel_discount * dist_reward,0.0])
        #print("rew:",reward)


        #reward  = 100. - (distance * 0.1)
        #if self.prev_distance == None or self.prev_distance == distance:
    #        reward = 10.0 - (distance * 0.1)
        #elif self.prev_distance > distance:
        #    reward = 20.0 - (distance * 0.1)
        #elif self.prev_distance < distance:
        #    reward = -20.0 - (distance * 0.1)

        self.prev_distance = distance
        return reward

    def step(self, rotor_speeds):
        """Uses action to obtain next state, reward, done."""
        reward = 0
        pose_all = []
        done = False
        #print("Taking repeated steps:")
        for _ in range(self.action_repeat):
            done = self.sim.next_timestep(rotor_speeds) # update the sim pose and velocities
            reward += self.get_reward()
            #if reward <= -3.0:
            #    done = True
            #if done and self.sim.time < self.sim.runtime:
            #    reward = -100

            #print(self.sim.pose)
            #distances = np.array(self.sim.pose[:3] - self.target_pos)
            #print(distances)
            #print(np.concatenate([self.sim.pose,distances]))
            #pose_all.append(np.concatenate([self.sim.pose,distances]))
            pose_all.append(np.concatenate([self.sim.pose]))
        next_state = np.concatenate(pose_all)
        return next_state, reward, done

    def reset(self):
        """Reset the sim to start a new episode."""
        self.sim.reset()
        distances = np.array(self.sim.pose[:3] - self.target_pos)
        state = np.concatenate([self.sim.pose])
        state = np.concatenate([state] * self.action_repeat)
        return state



import numpy as np
from physics_sim import PhysicsSim

class Task():
    """Task (environment) that defines the goal and provides feedback to the agent.
       The task to be learned is how to takeoff.
       The quadcopter has an initial position of (0, 0, 0) and a target position of (0, 0, 20)
    """

    def __init__(self, init_pose=None, init_velocities=None,
        init_angle_velocities=None, runtime=5., target_pos=None):
        """Initialize a Task object.
        Params
        ======
            init_pose: initial position of the quadcopter in (x,y,z) dimensions and the Euler angles
            init_velocities: initial velocity of the quadcopter in (x,y,z) dimensions
            init_angle_velocities: initial radians/second for each of the three Euler angles
            runtime: time limit for each episode
            target_pos: target/goal (x,y,z) position for the agent
        """
        # Simulation
        self.sim = PhysicsSim(init_pose, init_velocities, init_angle_velocities, runtime)
        self.action_repeat = 3

        self.state_size = self.action_repeat * 6
        self.action_low = 0
        self.action_high = 900
        self.action_size = 4

        # Goal
        self.target_pos = target_pos if target_pos is not None else np.array([0., 0., 10.])

    def get_reward(self):
        """Uses current pose of sim to return reward."""
        # this is the original, commented out
        # reward = 1.-.3*(abs(self.sim.pose[:3] - self.target_pos)).sum()

        # new reward function, start reward at 0
        reward = 0.
        # calculate the coordinate distance from the target position
        dist_x = abs(self.sim.pose[0] - self.target_pos[0])
        dist_y = abs(self.sim.pose[1] - self.target_pos[1])
        dist_z = abs(self.sim.pose[2] - self.target_pos[2])
        # create penalty, starting with 0.03
        penalty = 0.3*(np.sqrt((dist_x**2) + (dist_y**2) + (dist_z**2)))
        # add bonus
        bonus = 10.
        # calculate reward
        reward = reward + bonus - penalty
        return reward



    def step(self, rotor_speeds):
        """Uses action to obtain next state, reward, done."""
        reward = 0
        pose_all = []
        for _ in range(self.action_repeat):
            done = self.sim.next_timestep(rotor_speeds) # update the sim pose and velocities
            reward += self.get_reward()
            pose_all.append(self.sim.pose)
        next_state = np.concatenate(pose_all)
        return next_state, reward, done

    def reset(self):
        """Reset the sim to start a new episode."""
        self.sim.reset()
        state = np.concatenate([self.sim.pose] * self.action_repeat)
        return state
