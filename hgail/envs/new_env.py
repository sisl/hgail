
import numpy as np

from rllab.envs.base import Env
from rllab import spaces

from hgail.misc.utils import to_onehot

mov_dic={0:[0,1],1:[1,0],2:[-1,0],3:[0,-1]}
class DualGoalEnv(Env):
    def __init__(self,checkpoint=False,typ=0):

        self.numrow=6
        self.numcol=6
        self.type=typ
        self.checkpoint=checkpoint
        self.chkpassed=False
        self.reset()

    def step(self, action):

        reward=-1
        done=False
        goal1=[self.numrow/2,self.numcol-1]
        goal2=[self.numrow-1,self.numcol/2]
        if(self._state[2]==1):
            goal2=[self.numrow/2,self.numcol-1]
            goal1=[self.numrow-1,self.numcol/2]

        if self.checkpoint:
            if self.chkpassed:
                #Go to goal 
                if(list(self._state[:2])==goal1):
                    done=True
                    reward=10
            else:
                #go to checkpoint
                if(list(self._state[:2])==goal2):
                    self.chkpassed=True
        else:
            #check if goal
            if(list(self._state[:2])==goal1):
                done=True
                reward=10

        move_probs=[0.025,0.025,0.025,0.025]
        move_probs[action]+=0.9
        mov=np.random.choice([0,1,2,3],p=move_probs)
        new_state=np.copy(self._state)
        if(new_state[0]+mov_dic[mov][0]>=0 and new_state[0]+mov_dic[mov][0]<self.numrow and new_state[1]+mov_dic[mov][1]>=0 and new_state[1]+mov_dic[mov][1]<self.numcol):
            new_state[0]=new_state[0]+mov_dic[mov][0]
            new_state[1]=new_state[1]+mov_dic[mov][1]
        self._state=new_state

        env_info = dict(domain=to_onehot(np.reshape(self._state[-1], (1,1)), 2)[0])
        return np.copy(self._state), reward, done, env_info

    def reset(self):
        self._state=[0,0,self.type]
        if(self.type==2):
            self._state[2]=np.random.randint(2)
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
