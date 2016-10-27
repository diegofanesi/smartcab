import random
from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator
import numpy as np
import os
import collections 

from sets import Set

class State(object):
    def __init__(self, actions_enabled, next_step):
        self.actions_enabled = actions_enabled
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
        if self.next_step == None:
                hashN *= 19
        for x in self.actions_enabled:
            if x == 'right':
                hashN *= 11
            if x == 'left':
                hashN *= 13
            if x == 'forward':
                hashN *= 17
            if x == None:
                hashN *= 23
        return hashN
       
class LearningAgent(Agent):
    """An agent that learns to drive in the smartcab world."""
    

    def __init__(self, env, qTable=dict(), epsilon=0.1):
        super(LearningAgent, self).__init__(env)  # sets self.env = env, state = None, next_waypoint = None, and a default color
        self.color = 'red'  # override color
        self.planner = RoutePlanner(self.env, self)  # simple route planner to get next_waypoint
        # TODO: Initialize any additional variables here
        self.actions = ['left', 'right', 'forward', None]
        self.qTable = qTable
        self.alpha = 0.26
        self.epsilon = epsilon
        self.State = collections.namedtuple("State", 'actions_enabled heading delta')
        self.sumReward = 0.0
        self.discount = 0.33
        
        ############ stat variables ##############
        self.arrayRewards=[]
        self.arrayTrialResults=[]
        self.lastReward =-5.0
        
    def updateQValue (self, state, action, nextState, reward):
            self.qTable[(state, action)] = (1-self.alpha) * self.getQValue(state, action) + (self.alpha * (reward + self.discount * self.getMaxQValue(nextState)[0]- self.getQValue(state, action)))

    def getQValue (self, state, action):
        return self.qTable.get((state, action), 1)
    
    def getMaxQValue (self, state):
        bestQ = -999999.99
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
            next_step=self.next_waypoint
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
    
        if self.lastReward != -5.0:
            self.arrayTrialResults.append( 1 if self.lastReward>8 else 0 )
            print ("success rate: " + str(float(self.arrayTrialResults.count(1))/float(self.arrayTrialResults.__len__())*100) + "%")
            print ("penalties: " + str(float(self.arrayRewards.count(-1.0)+self.arrayRewards.count(-0.5))/float(self.arrayRewards.__len__())*100) + "%")
        
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
        
        self.lastReward = reward
        self.arrayRewards.append(reward)
        

        #print "LearningAgent.update(): deadline = {}, inputs = {}, action = {}, reward = {}".format(deadline, inputs, action, reward)  # [debug]


def run():
    """Run the agent for a finite number of trials."""
    
    e = Environment(15)
    a = e.create_agent(LearningAgent)
    e.set_primary_agent(a, enforce_deadline=True)
    sim = Simulator(e, update_delay=0.0, display=False)
    sim.run(n_trials=50) 
    os.system('read -s -n 1 -p "Press any key to continue..."')
    e = Environment()
    table = a.qTable.copy()
    a = e.create_agent(LearningAgent, qTable=table)
    e.set_primary_agent(a, enforce_deadline=True)
    sim = Simulator(e, update_delay=0.3, display=True)
    sim.run(n_trials=50)
    # NOTE: To quit midway, press Esc or close pygame window, or hit Ctrl+C on the command-line


if __name__ == '__main__':
    run()
