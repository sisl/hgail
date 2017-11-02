from rllab.envs.base import Env
from rllab import spaces

import numpy as np

mov_dic={0:[0,1],1:[1,0],2:[-1,0],3:[0,-1]}
class DualGoalEnv(Env):
    def __init__(self,checkpoint=False):

        self.numrow=6
        self.numcol=6
        self.checkpoint=checkpoint
        self.chkpassed=False
        self.reset()

    def step(self, action):

        reward=0
        done=False
        goal1=[self.numrow/2,self.numcol-1]
        goal2=[self.numrow-1,self.numcol/2]
        move_probs=[0.025,0.025,0.025,0.025]
        move_probs[action]+=0.9
        mov=np.random.choice([0,1,2,3],p=move_probs)
        new_state=np.copy(self._state)
        if(new_state[0]+mov_dic[mov][0]>=0 and new_state[0]+mov_dic[mov][0]<self.numrow and new_state[1]+mov_dic[mov][1]>=0 and new_state[1]+mov_dic[mov][1]<self.numcol):
            new_state[0]=new_state[0]+mov_dic[mov][0]
            new_state[1]=new_state[1]+mov_dic[mov][1]
        self._state=new_state

        if self.checkpoint:
            if self.chkpassed:
                #Go to goal 2
                if(list(self._state[:2])==goal2):
                    reward=10
                    done=True
            else:
                #go to goal 1
                if(list(self._state[:2])==goal1):
                    self.chkpassed=True

        else:
            #check if goal
            if(list(self._state[:2])==goal1):
                reward=10
                done=True

        return np.copy(self._state), reward, done, {}

    def reset(self):
        self._state=[0,0,1]
        self.chkpassed=False
        observation = np.copy(self._state)
        return observation
    def render(self):
        print('current state:', self._state)
    @property
    def action_space(self):
        return spaces.Discrete(4)

    @property
    def observation_space(self):
        return spaces.Product([spaces.Discrete(self.numrow),spaces.Discrete(self.numcol),spaces.Discrete(2)])
