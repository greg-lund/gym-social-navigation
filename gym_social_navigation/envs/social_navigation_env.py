import gym,colorsys,os
import matplotlib.pyplot as plt
import numpy as np
from gym import spaces

class SocialNavigationEnv(gym.Env):

    def __init__(self):

        # Parameters worth changing
        self.agent_hist = 4 # How many previous states for each agent do we keep track of?
        max_vel = 0.75 # Maximum velocity of the robot in m/s
        dv = 0.125 # Discretization for velocity in actions
        self.steering_angle = 30 * np.pi/180 # We assume a Dubins Car with fixed steering angle
        self.prox_rad = 10 # In what radius around our robot do we keep track of human agents?
        self.num_agents = 10 # How many agents can we keep track of / include in our state?
        self.num_path_points = 2 # How many points to include in our discretized path
        self.min_path_len = 0.25 # Normalized min dist for the path [0,1]

        # Starting pose of the robot is pointing towards the goal
        self.set_random_path()
        self.pose = np.array([self.start[0],self.start[1],np.arctan2((self.goal-self.start)[1],(self.goal-self.start)[0])%(2*np.pi)])
        self.pose_history = np.array([self.pose])

        # Human Pedestrian Data
        datafile = os.path.dirname(os.path.abspath(__file__)) + '/data/eth/train/crowds_zara01_train.txt'
        data = np.loadtxt(datafile)
        self.timestep = 0.4 # Don't change! Discretization of the dataset
        self.mint,self.maxt = data[:,0].min(),data[:,0].max()
        self.minx,self.maxx = data[:,2].min(),data[:,2].max()
        self.miny,self.maxy = data[:,3].min(),data[:,3].max()

        # Agent data and colors
        self.total_agents = int(data[:,1].max() - data[:,1].min())
        self.colors = self.get_spaced_colors(self.total_agents)
        plt.rcParams['axes.facecolor'] = (0.9,0.9,0.9)
        self.agents = {id:{} for id in range(self.total_agents+1)}
        for x in data:
            id = int(x[1]-1)
            t = int((x[0]-self.mint)/10)
            self.agents[id][t] = np.array([(x[2]-self.minx)/(self.maxx-self.minx),(x[3]-self.miny)/(self.maxy-self.miny)])
        self.maxt = (self.maxt - self.mint)/10
        self.mint = 0

        # Setup observation and action spaces
        n = int( 3 + 2*self.num_agents*(self.agent_hist+1)  + 2*self.num_path_points )
        self.low = np.zeros(n)
        self.high = np.ones(n)
        self.observation_space = spaces.Box(np.float32(self.low),np.float32(self.high))
        self.actions = [('L',v) for v in np.arange(dv,max_vel+dv,dv)] + [('S',v) for v in np.arange(0,max_vel+dv,dv)] + [('R',v) for v in np.arange(dv,max_vel+dv,dv)]
        self.action_space = spaces.Discrete(len(self.actions))
        self.time = 0


    def set_random_path(self):
        '''
        Generate random starting and goal positions, and a path that connects them
        '''
        p1 = np.random.random(2)
        p2 = np.random.random(2)
        while np.linalg.norm(p1-p2) < self.min_path_len:
            p1 = np.random.random(2)
            p2 = np.random.random(2)

        self.start = p1
        self.goal = p2
        self.path = np.linspace(p1,p2,self.num_path_points).flatten()

    def get_agent_states(self,time):
        '''
        Return a sorted list of agent poses and histories, where the closest
        agent appears first. Only agents within self.prox_rad are included
        '''

        states = None
        n = 0
        for id in self.agents:
            if time in self.agents[id]:
                # If this agent is outside of our proximity radius, don't include it
                p = self.agents[id][time] - self.pose[0:2]
                if np.dot(p,p) >= self.prox_rad**2: continue
                if states is None:
                    states = np.zeros((1,2*(self.agent_hist+1)))
                else:
                    states = np.vstack([states,np.zeros(2*(self.agent_hist+1))])
                states[n,0:2] = self.agents[id][time]
                for i in range(1,self.agent_hist):
                    if time-i in self.agents[id]:
                        states[n,2*i:2*i+2]=self.agents[id][time-i]
                    else:
                        states[n,2*i:2*i+(self.agent_hist-i+1)*2] = np.array([self.agents[id][time-i+1] for _ in range(self.agent_hist-i+1)]).flatten()
                        break
                n+=1
        if states is None:
            return np.zeros(2*self.num_agents*(self.agent_hist+1))

        # Sort states by how close the agents are to robot
        robot_pos = np.array([(self.pose[0]-self.minx)/(self.maxx-self.minx),(self.pose[1]-self.miny)/(self.maxy-self.miny)])
        states = np.array(sorted(states, key=lambda row: np.linalg.norm(row[0:2]-robot_pos)))
        if len(states) < self.num_agents:
            states = np.vstack([states,np.zeros((self.num_agents-len(states),2*(self.agent_hist+1)))])
        return states[0:self.num_agents].flatten()
        
    def reset(self):
        '''
        Reset the entire simulation, including new start and goal
        '''
        self.set_random_path()
        self.pose = [self.start[0]*(self.maxx-self.minx)+self.minx,self.start[1]*(self.maxy-self.miny)+self.miny,np.arctan2((self.goal-self.start)[1],(self.goal-self.start)[0])%(2*np.pi)]
        self.pose_history = np.array([self.pose])
        self.time = 0
        robot_pos = np.array([(self.pose[0]-self.minx)/(self.maxx-self.minx),(self.pose[1]-self.miny)/(self.maxy-self.miny),self.pose[2]/(2*np.pi)])
        return np.concatenate((robot_pos,self.get_agent_states(0),self.path))

    def step(self,action):
        ''' 
        Main function called at each timestep. Given some action, in self.actions
        return our new observed state, reward given and whether we've terminated
        '''
        a = self.actions[action]

        # Update robot pose based on action
        if a[0] == 'L': # Positive theta dot
            self.pose[0] += a[1]/self.steering_angle*(np.sin(self.steering_angle*self.timestep+self.pose[2])-np.sin(self.pose[2]))
            self.pose[1] += a[1]/self.steering_angle*(-np.cos(self.steering_angle*self.timestep+self.pose[2])+np.cos(self.pose[2]))
            self.pose[2] += self.steering_angle*self.timestep
        elif a[0] == 'R': # Negative theta dot
            self.pose[0] += a[1]/self.steering_angle*(-np.sin(-self.steering_angle*self.timestep+self.pose[2])+np.sin(self.pose[2]))
            self.pose[1] += a[1]/self.steering_angle*(np.cos(-self.steering_angle*self.timestep+self.pose[2])-np.cos(self.pose[2]))
            self.pose[2] -= self.steering_angle*self.timestep
        else: # Theta dot is zero
            self.pose[0] += self.timestep*a[1]*np.cos(self.pose[2])
            self.pose[1] += self.timestep*a[1]*np.sin(self.pose[2])

        # Update pose history for rendering
        self.pose_history = np.insert(self.pose_history,0,np.copy(self.pose),0)
        self.pose_history = self.pose_history[0:self.agent_hist]

        # Update observation
        self.time += 1
        robot_pos = np.array([(self.pose[0]-self.minx)/(self.maxx-self.minx),(self.pose[1]-self.miny)/(self.maxy-self.miny),self.pose[2]/(2*np.pi)])
        state = np.concatenate((robot_pos,self.get_agent_states(self.time),self.path))

        # Calculate reward
        reward = 0

        # Is this a terminal state?
        done = False
        #if self.time >= self.maxt: done = True
        #if np.sqrt((self.pose[0]-self.goal[0])**2+(self.pose[1]-self.goal[1])**2) < 1e-1: done = True

        return state,reward,done,{}

    def render(self):
        '''
        Render the current timestep using pyplot
        '''
        plt.clf()
        for id in self.agents:
            # Is this agent in the scene during this timestep?
            if self.time in self.agents[id]:
                plt.axis([0,1,0,1])
                plt.scatter(self.agents[id][self.time][0],self.agents[id][self.time][1],color=tuple(self.colors[id]),zorder=1)
                for i in range(1,self.agent_hist+1):
                    if self.time-i in self.agents[id]:
                        plt.scatter(self.agents[id][self.time-i][0],self.agents[id][self.time-i][1],facecolors='none',color=tuple(self.colors[id]),zorder=1)
        plt.plot([self.start[0],self.goal[0]],[self.start[1],self.goal[1]],color=(0,1,0),zorder=0)
        plt.scatter((self.pose[0]-self.minx)/(self.maxx-self.minx),(self.pose[1]-self.miny)/(self.maxy-self.miny),s=125,marker=(5,1),color=(0,0,0),zorder=2)
        for i in range(1,len(self.pose_history)):
            plt.scatter((self.pose_history[i][0]-self.minx)/(self.maxx-self.minx),(self.pose_history[i][1]-self.miny)/(self.maxy-self.miny),s=125,marker=(5,1),facecolors='none',color=(0,0,0),zorder=2)
        ax = plt.gca()
        ax.axes.xaxis.set_visible(False)
        ax.axes.yaxis.set_visible(False)
        plt.pause(0.05)

    # Create a list of n "equally spaced" rgb colors
    @staticmethod
    def get_spaced_colors(n):
        '''
        Helper function that assigns somewhat unique colors to agents
        '''
        colors = np.zeros((n,3))
        idxs = np.arange(0,n,1).astype(int)
        np.random.shuffle(idxs)
        j = 0
        for i in idxs:
            h = j*1.0/n
            rgb = colorsys.hsv_to_rgb(h,1,1)
            colors[i] = rgb
            j+=1
        return colors
