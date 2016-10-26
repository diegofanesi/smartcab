import random
from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator
import numpy as np
import collections 
from sets import Set

class State(object):
    def __init__(self, actions_enabled, next_step):
        self.actions_enabled = actions_enabled.copy()
        self.next_step = next_step
        
        
    def __eq__(self, a):
        if (type(a) != type(self)):
            return False
        if (a.next_step != self.next_step):
            return False
        for x in self.actions_enabled:
            if x not in a.actions_enabled:
                return False
        for x in a.actions_enabled:
            if x not in self.actions_enabled:
                return False
        return True
    
    def __str__(self):
        return ("next_step: " + self.next_step + " actions_enabled: " + self.actions_enabled)
    
    def __hash__(self):
        hashN=1
        if self.next_step == 'right':
                hashN *= 7 
        if self.next_step == 'left':
                hashN *= 3
        if self.next_step == 'forward':
                hashN *= 5
        for x in self.actions_enabled:
            if x == 'right':
                hashN *= 11
            if x == 'left':
                hashN *= 13
            if x == 'forward':
                hashN *= 17
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
        self.alpha = 1
        self.epsilon = epsilon
        self.State = collections.namedtuple("State", 'actions_enabled heading delta')
        self.sumReward = 0.0
        self.discount = 0.5
        
        
        # self.discount
        # self.gamma
        
    def updateQValue (self, state, action, nextState, reward):
        #if((state, action) not in self.qTable): 
            self.qTable[(state, action)] = self.alpha * (reward + (self.discount * self.getMaxQValue(nextState)[0]))
        #else:
            # print((state, action))
        #    self.qTable[(state, action)] = self.qTable[(state, action)] + (self.alpha * (reward + (self.discount * (self.getMaxQValue(nextState)[0])) - self.qTable[(state, action)]))

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
    
    def makeState(self, inputs, next_step=None):
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
                
        if next_step==None:
            next_Step=self.next_waypoint
        state = State(actions_enabled=actions_enabled, next_step=next_step)
        return state
    
    def actionToTake(self, state):
        return  np.random.choice([self.getMaxQValue(state)[1], np.random.choice(self.actions, 1)[0]], 1, [1 - self.epsilon, self.epsilon])[0]
        
                     
    def reset(self, destination=None):
        self.planner.route_to(destination)
        # TODO: Prepare for a new trip; reset any variables here, if required
        print("Total Reward: " + str(self.sumReward))
        self.sumReward = 0.0
        print("NStates: " + str(len(self.qTable.keys())))
    
    def update(self, t):
        # Gather inputs
          # from route planner, also displayed by simulator
        self.next_waypoint = self.planner.next_waypoint()
        curpos = self.env.agent_states[self]['location']
        inputs = self.env.sense(self)
        deadline = self.env.get_deadline(self)
        curstate = self.makeState(inputs)

        action = self.actionToTake(curstate)
        reward = self.env.act(self, action)
        newpos = self.env.agent_states[self]['location']
        self.updateQValue(curstate, action, self.makeState(self.env.sense(self),self.planner.next_waypoint()), reward)
        self.sumReward += reward
        
        self.epsilon = self.epsilon * 0.99

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
    
    e = Environment(40)
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
