import gym,colorsys,os
import matplotlib.pyplot as plt
import numpy as np
from gym import spaces

class SocialNavEnv(gym.Env):

    def __init__(self,test=False):

        # Parameters worth changing
        self.agent_hist = 3 # How many previous states for each agent do we keep track of?
        max_vel = 0.75 # Maximum velocity of the robot in m/s
        dv = 0.25 # Discretization for velocity in actions
        self.steering_angle = 30 * np.pi/180 # We assume a Dubins Car with fixed steering angle
        self.prox_rad = 10 # In what radius around our robot do we keep track of human agents?
        self.num_agents = 4 # How many agents can we keep track of / include in our state?
        self.num_path_points = 4 # How many points to include in our discretized path
        self.min_path_len = 0.45 # Normalized min dist for the path [0,1]
        self.episode_time = 60 # How long should the episodes last before being terminated?
         
        # Reward function parameters
        self.alpha = 3.5 # Delta path parameter
        self.beta = 5  # Delta goal parameter
        self.goal_reward = 10
        self.collision_penalty = 20
        self.out_of_bounds_penalty = 15
        self.goal_thresh = 0.05
        self.collision_dist = 0.30
        
        # Human Pedestrian Data
        if test:
            files = ['data/val/biwi_eth_val.txt','data/val/crowds_zara01_val.txt',
                    'data/val/crowds_zara02_val.txt','data/val/crowds_zara03_val.txt','data/val/students001_val.txt',
                    'data/val/students003_val.txt','data/val/uni_examples_val.txt']
        else:
            files = ['data/train/biwi_eth_train.txt','data/train/crowds_zara01_train.txt',
                    'data/train/crowds_zara02_train.txt','data/train/crowds_zara03_train.txt','data/train/students001_train.txt',
                    'data/train/students003_train.txt','data/train/uni_examples_train.txt']

        for i in range(len(files)):
            files[i] = os.path.dirname(os.path.abspath(__file__)) + '/' + files[i]

        # Get all agent data in self.agent_data
        self.parse_data(files)
        total_agents = max([x['num_agents'] for x in self.agent_data])

        # Plotting
        self.colors = self.get_spaced_colors(total_agents)
        plt.rcParams['axes.facecolor'] = (0.9,0.9,0.9)

        self.timestep = 0.4 # Don't change! Discretization of the dataset
        self.episode_time = int(self.episode_time/self.timestep)

        # Setup observation and action spaces
        n = int(3 + 2*self.num_agents*(self.agent_hist+1) + 2*self.num_path_points)
        self.low = np.zeros(n)
        self.high = np.ones(n)
        self.observation_space = spaces.Box(np.float32(self.low),np.float32(self.high))
        self.actions = [('L',v) for v in np.arange(dv,max_vel+dv,dv)] + [('S',v) for v in np.arange(0,max_vel+dv,dv)] + [('R',v) for v in np.arange(dv,max_vel+dv,dv)]
        self.action_space = spaces.Discrete(len(self.actions))
        self.time = 0
        self.start_time = 0

        # Reset simulation to populate env data
        _ = self.reset()

    def parse_data(self,filenames):
        '''
        Given filenames, parse the data and return a list of dictionaries of agent movements
        '''

        # Get common ranges among all datasets
        xrange,yrange = float('inf'),float('inf')
        for f in filenames:
            data = np.loadtxt(f)
            minx,maxx = data[:,2].min(),data[:,2].max()
            miny,maxy = data[:,3].min(),data[:,3].max()
            if maxx-minx < xrange:
                xrange = maxx-minx
            if maxy-miny < yrange:
                yrange = maxy-miny

        self.agent_data = []
        for filename in filenames:
            data = np.loadtxt(filename)
            minid,maxid = int(data[:,1].min()),int(data[:,1].max())
            num_agents = int(data[:,1].max() - data[:,1].min())
            mint,maxt = data[:,0].min(),data[:,0].max()
            minx,maxx = data[:,2].min(),data[:,2].max()
            miny,maxy = data[:,3].min(),data[:,3].max()

            if maxx-minx > xrange:
                maxx = maxx - (maxx-minx-xrange)
            if maxy-miny > yrange:
                maxy = maxy - (maxy-miny-yrange)

            agents = {id:{} for id in range(num_agents+1)}
            agents['num_agents'] = num_agents
            # limits: [mint,maxt,minx,maxx,miny,maxy]
            agents['limits'] = [0,int((maxt-mint)/10),minx,maxx,miny,maxy]

            for x in data:
                id = int(x[1]-1) - (minid-1)
                t = int((x[0]-mint)/10)
                if x[2] > maxx or x[3] > maxy:
                    continue
                agents[id][t] = np.array([(x[2]-minx)/(maxx-minx),(x[3]-miny)/(maxy-miny)])

            self.agent_data.append(agents)

    def set_random_path(self):
        '''
        Generate random starting and goal positions, and a path that connects them
        '''
        valid_start_time = False
        while not valid_start_time:
            p1 = np.random.random(2)
            p2 = np.random.random(2)
            while np.linalg.norm(p1-p2) < self.min_path_len:
                p1 = np.random.random(2)
                p2 = np.random.random(2)

            self.start = p1
            self.goal = p2
            self.pose = np.array([self.start[0]*(self.maxx-self.minx)+self.minx,self.start[1]*(self.maxy-self.miny)+self.miny,np.arctan2((self.goal-self.start)[1],(self.goal-self.start)[0])%(2*np.pi)])

            valid_start_time = True
            ids = [id for id in self.agents.keys() if isinstance(id,int)]
            for id in ids:
                if self.start_time in self.agents[id]:
                    agent_pos = np.array([self.agents[id][self.time][0]*(self.maxx-self.minx) + self.minx,
                        self.agents[id][self.time][1]*(self.maxy-self.miny) + self.miny])
                    if np.linalg.norm(agent_pos - self.pose[0:2]) < 2*self.collision_dist:
                        valid_start_time = False
                        break

        self.path = np.linspace(p1,p2,self.num_path_points).flatten()
        theta = (np.arctan2((self.goal-self.start)[1],(self.goal-self.start)[0]) + np.random.normal(0,np.pi/8) ) % (2*np.pi)
        self.pose = np.array([self.start[0]*(self.maxx-self.minx)+self.minx,self.start[1]*(self.maxy-self.miny)+self.miny,theta])
        self.pose_history = np.array([self.pose])

    def set_fixed_path(self,start,goal):
        self.start = start
        self.goal = goal
        self.pose = np.array([self.start[0]*(self.maxx-self.minx)+self.minx,self.start[1]*(self.maxy-self.miny)+self.miny,np.arctan2((self.goal-self.start)[1],(self.goal-self.start)[0])%(2*np.pi)])
        self.path = np.linspace(start,goal,self.num_path_points).flatten()
        self.pose = np.array([self.start[0]*(self.maxx-self.minx)+self.minx,self.start[1]*(self.maxy-self.miny)+self.miny,np.arctan2((self.goal-self.start)[1],(self.goal-self.start)[0])%(2*np.pi)])
        self.pose_history = np.array([self.pose])

    def get_agent_states(self,time):
        '''
        Return a sorted list of agent poses and histories, where the closest
        agent appears first. Only agents within self.prox_rad are included
        '''

        states = None
        n = 0
        ids = [id for id in self.agents.keys() if isinstance(id,int)]
        for id in ids:
            if time in self.agents[id]:
                # If this agent is outside of our proximity radius, don't include it
                p = self.agents[id][time] - self.pose[0:2]
                if np.dot(p,p) >= self.prox_rad**2: continue
                if states is None:
                    states = np.zeros((1,2*(self.agent_hist+1)))
                else:
                    states = np.vstack([states,np.zeros(2*(self.agent_hist+1))])
                states[n,0:2] = self.agents[id][time]
                for i in range(1,self.agent_hist+1):
                    if time-i in self.agents[id]:
                        states[n,2*i:2*i+2]=self.agents[id][time-i]
                    else:
                        states[n,2*i:] = np.array([self.agents[id][time-i+1] for _ in range(self.agent_hist-i+1)]).flatten()
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
        Reset the entire simulation, including new dataset, start and goal
        '''

        # Set new dataset
        idx = np.random.randint(len(self.agent_data))
        self.agents = self.agent_data[idx]
        self.mint,self.maxt = 0,self.agents['limits'][1]-self.agents['limits'][0]
        self.minx,self.maxx = self.agents['limits'][2],self.agents['limits'][3]
        self.miny,self.maxy = self.agents['limits'][4],self.agents['limits'][5]

        # Set new path, start and goal
        if self.maxt - self.episode_time > 0:
            self.time = np.random.randint(low=0,high=self.maxt-self.episode_time)
        else:
            self.time = 0
        self.start_time = self.time

        self.set_random_path()

        robot_pos = np.array([(self.pose[0]-self.minx)/(self.maxx-self.minx),(self.pose[1]-self.miny)/(self.maxy-self.miny),(self.pose[2]%(2*np.pi))/(2*np.pi)])
        return np.concatenate((robot_pos,self.get_agent_states(self.time),self.path))

    def distance_to_path(self,pose):
        '''
        Given some robot pose, return the distance to our linear path
        '''
        a = self.start[1]-self.goal[1]
        b = self.goal[0]-self.start[0]
        c = self.goal[1]*(self.start[0]-self.goal[0]) + self.goal[0]*(self.goal[1]-self.start[1])
        return abs(a*pose[0]+b*pose[1]+c)/np.sqrt(a*a+b*b)

    def step(self,action):
        ''' 
        Main function called at each timestep. Given some action, in self.actions
        return our new observed state, reward given and whether we've terminated
        '''
        a = self.actions[action]

        prev_pose = np.copy(self.pose)

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

        self.pose[2] = self.pose[2]%(2*np.pi)
        # Update pose history for rendering
        self.pose_history = np.insert(self.pose_history,0,np.copy(self.pose),0)
        self.pose_history = self.pose_history[0:self.agent_hist]

        # Update observation
        self.time += 1
        robot_pos = np.array([(self.pose[0]-self.minx)/(self.maxx-self.minx),(self.pose[1]-self.miny)/(self.maxy-self.miny),self.pose[2]/(2*np.pi)])
        prev_robot_pos = np.array([(prev_pose[0]-self.minx)/(self.maxx-self.minx),(prev_pose[1]-self.miny)/(self.maxy-self.miny),prev_pose[2]/(2*np.pi)])
        state = np.concatenate((robot_pos,self.get_agent_states(self.time),self.path))

        # Calculate change in distance to goal
        d0 = np.linalg.norm(prev_robot_pos[0:2] - self.goal)
        d1 = np.linalg.norm(robot_pos[0:2] - self.goal)
        delta_goal = -d1+d0

        # Calculate change in distance from path
        p0 = self.distance_to_path(prev_robot_pos)
        p1 = self.distance_to_path(robot_pos)
        delta_path = -p1+p0

        # Calculate reward
        reward = self.alpha*delta_path + self.beta*delta_goal
        done = False

        # Did we collide with a human agent?
        ids = [id for id in self.agents.keys() if isinstance(id,int)]
        for id in ids:
            if self.time in self.agents[id]:
                agent_pos = np.array([self.agents[id][self.time][0]*(self.maxx-self.minx) + self.minx,
                    self.agents[id][self.time][1]*(self.maxy-self.miny) + self.miny])
                if np.linalg.norm(agent_pos-self.pose[:2]) < self.collision_dist:
                    reward = -self.collision_penalty
                    done = True
                    return state,reward,done,{}

        # Did we reach the goal?
        if np.linalg.norm(robot_pos[0:2]-self.goal) < self.goal_thresh:
            done = True
            reward = self.goal_reward
            return state,reward,done,{}

        # Are we out of time?
        if self.time >= self.start_time + self.episode_time: 
            done = True
            return state,reward,done,{}

        # Are we out of bounds?
        if robot_pos[0] < 0 or robot_pos[0] > 1 or robot_pos[1] < 0 or robot_pos[1] > 1: 
            done = True
            reward = -self.out_of_bounds_penalty
            return state,reward,done,{}

        return state,reward,done,{}

    def render(self):
        '''
        Render the current timestep using pyplot
        '''
        plt.clf()
        plt.axis([0,1,0,1])
        ids = [id for id in self.agents.keys() if isinstance(id,int)]
        for id in ids:
            # Is this agent in the scene during this timestep?
            if self.time in self.agents[id]:
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
