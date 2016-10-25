import random
from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator
import numpy as np
import collections 
from sets import Set

class State(object):
    def __init__(self, actions_enabled, heading, delta):
        self.actions_enabled = actions_enabled
        self.heading = heading
        self.delta = delta
        
        
    def __eq__(self, a):
        if (type(a) != type(self)):
            return False
        if (a.heading[0] != self.heading[0] or a.heading[1] != self.heading[1]):
            return False
        if (a.delta[0] != self.delta[0] or a.delta[1] != self.delta[1]):
            return False
        for x in self.actions_enabled:
            if x not in a.actions_enabled:
                return False
        return True
    
    def __str__(self):
        return ("position: " + self.delta + " heading: " + self.heading + " actions_enabled: " + self.actions_enabled)
    
    def __hash__(self):
        hashN = self.heading[0] * 3
        hashN += self.heading[1] * 5
        hashN += self.delta[0] * 7
        hashN += self.delta[1] * 11
        for x in self.actions_enabled:
            if x == 'right':
                hashN += 23 * 13
            if x == 'left':
                hashN += 23 * 17
            if x == 'forward':
                hashN += 23 * 19

        return hashN
    
class LearningAgent(Agent):
    """An agent that learns to drive in the smartcab world."""
    

    def __init__(self, env, qTable=dict(), epsilon=0.9):
        print (qTable)
        super(LearningAgent, self).__init__(env)  # sets self.env = env, state = None, next_waypoint = None, and a default color
        self.color = 'red'  # override color
        self.planner = RoutePlanner(self.env, self)  # simple route planner to get next_waypoint
        # TODO: Initialize any additional variables here
        self.actions = ['left', 'right', 'forward', None]
        self.qTable = qTable
        self.alpha = 0.1
        self.epsilon = epsilon
        self.State = collections.namedtuple("State", 'actions_enabled heading delta')
        self.sumReward = 0.0
        self.discount = 0.4
        
        
        # self.discount
        # self.gamma
        
    def updateQValue (self, state, action, nextState, reward):
        if((state, action) not in self.qTable): 
            self.qTable[(state, action)] = self.alpha * (reward + (self.discount * self.getMaxQValue(nextState)[0]))
        else:
            # print((state, action))
            self.qTable[(state, action)] = self.qTable[(state, action)] + (self.alpha * (reward + (self.discount * self.getMaxQValue(nextState)[0]) - self.qTable[(state, action)]))

    def getQValue (self, state, action):
        return self.qTable.get((state, action), 0)
    
    def getMaxQValue (self, state):
        bestQ = -999999
        bestAction = None
        for a in self.actions:
            if (self.getQValue(state, a) > bestQ):
                bestQ = self.getQValue(state, a)
                bestAction = a
            elif(self.getQValue(state, a) == bestQ):
                bestAction = np.random.choice([bestAction, a], 1)[0]
        return [bestQ, bestAction]
    
    def calcReward(self, lastpos, curpos, target, envReward, deadline):
        rw = 0  # -5.0 / (deadline + 1)   
        if (envReward == -1.0):
            return rw + -2.0  # in case of incident or illegal action return a bad reward no matter how closer to the target  
        if (envReward == 12.0):
            return envReward * 2  # hitting the target is 24 points reward!
        if (self.env.compute_dist(lastpos, target) > self.env.compute_dist(curpos, target)):
            rw = 2.0  # reward if it gets closer to the target
        if (self.env.compute_dist(lastpos, target) < self.env.compute_dist(curpos, target)):
            rw = -0.5  # rw * 2.0  # going farther away from the target is twice the reward of the deadline
        return rw
    
    def makeState(self, inputs):
        location = self.env.agent_states[self]['location']
        heading = self.env.agent_states[self]['heading']
        destination = self.planner.destination
        # Delta is just a vector that represents the direction of the target from the position of the cab
        delta = [location[0] - destination[0], location[1] - destination[1]]
        if (delta[0] > 0):
            delta[0] = 1
        if(delta[1] > 0):
            delta[1] = 1
        if (delta[0] < 0):
            delta[0] = -1
        if(delta[1] < 0):
            delta[1] = -1
        
        # in the status we don't need all the outputs, but just the enabled ways. all the rest will not affect the reward
        actions_enabled = Set()
        if inputs['light'] == 'red':
            if inputs['left'] != 'forward':
                actions_enabled.add('right')
        if inputs['light'] == 'green':
            actions_enabled.add('forward')
            actions_enabled.add('right')
            if inputs['oncoming'] != 'right' and inputs['oncoming'] != 'forward':
                actions_enabled.add('left')
                
        state = State(actions_enabled=actions_enabled, delta=delta, heading=heading)
        return hash(state)
    
    def actionToTake(self, state):
        return  np.random.choice([self.getMaxQValue(state)[1], np.random.choice(self.actions, 1)[0]], 1, [1 - self.epsilon, self.epsilon])[0]
        
                     
    def reset(self, destination=None):
        self.planner.route_to(destination)
        # TODO: Prepare for a new trip; reset any variables here, if required
        print("Total Reward: " + str(self.sumReward))
        self.sumReward = 0.0
        print("NStates: " + str(self.qTable.__len__()))
    
    def update(self, t):
        # Gather inputs
          # from route planner, also displayed by simulator
        curpos = self.env.agent_states[self]['location']
        inputs = self.env.sense(self)
        deadline = self.env.get_deadline(self)
        curstate = self.makeState(inputs)
        self.next_waypoint = self.actionToTake(curstate)
        destination = self.planner.destination

        # action = None
        # if action_okay:
        action = self.next_waypoint
            # self.next_waypoint = self.planner.next_waypoint()
        reward = self.env.act(self, action)
        newpos = self.env.agent_states[self]['location']
        reward = self.calcReward(curpos, newpos, destination, reward, deadline)
        self.updateQValue(curstate, action, self.makeState(self.env.sense(self)), reward)
        self.sumReward += reward
        
        self.epsilon = self.epsilon * 0.99
        # TODO: Update state
        # print(inputs)

        
        # TODO: Select action according to your policy
        # action = None

        # Execute action and get reward
        # reward = self.env.act(self, action)

        # TODO: Learn policy based on state, action, reward

        # print "LearningAgent.update(): deadline = {}, inputs = {}, action = {}, reward = {}".format(deadline, inputs, action, reward)  # [debug]


def run():
    """Run the agent for a finite number of trials."""

    # Set up environment and agent
    e = Environment(0)  # create environment (also adds some dummy traffic)
    a = e.create_agent(LearningAgent)  # create agent
    e.set_primary_agent(a, enforce_deadline=False)  # specify agent to track
    
    # NOTE: You can set enforce_deadline=False while debugging to allow longer trials

    # Now simulate it
    sim = Simulator(e, update_delay=0.0, display=False)  # create simulator (uses pygame when display=True, if available)
    # NOTE: To speed up simulation, reduce update_delay and/or set display=False

    sim.run(n_trials=1000)  # run for a specified number of trials
    
    e = Environment(100)
    table = a.qTable.copy()
    a = e.create_agent(LearningAgent, qTable=table)
    e.set_primary_agent(a, enforce_deadline=False)
    sim = Simulator(e, update_delay=0.0, display=False)
    sim.run(n_trials=1000) 
    
    e = Environment()
    table = a.qTable.copy()
    a = e.create_agent(LearningAgent, qTable=table, epsilon=0)
    e.set_primary_agent(a, enforce_deadline=True)
    sim = Simulator(e, update_delay=0.3, display=True)
    sim.run(n_trials=100)
    # NOTE: To quit midway, press Esc or close pygame window, or hit Ctrl+C on the command-line


if __name__ == '__main__':
    run()
