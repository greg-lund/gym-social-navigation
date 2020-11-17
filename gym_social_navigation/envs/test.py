import sys,os,time,colorsys
import matplotlib.pyplot as plt
import numpy as np

class human_agents:
    def __init__(self,datafile,num_agents=None,agent_hist=0,start=(1,6),goal=(14,6)):
        self.filename = datafile
        data = np.loadtxt(datafile)

        # Map image
        self.minx,self.maxx = data[:,2].min(),data[:,2].max()
        self.miny,self.maxy = data[:,3].min(),data[:,3].max()

        # How many agents to consider in our simulation?
        # If left as None, we use all agents
        if num_agents is None:
            self.num_agents = int(data[:,1].max())
        else:
            self.num_agents = num_agents

        self.agent_hist = agent_hist

        self.colors = self.get_spaced_colors(self.num_agents)
        self.agents = {id:{} for id in range(self.num_agents)}
        for x in data:
            id = x[1]-1
            t = x[0]
            if id < self.num_agents:
                self.agents[id][t] = (x[2],x[3])

        # Start and goal for the robot
        self.start = start
        self.goal = goal

        self.robot_pos = start
        plt.rcParams['axes.facecolor'] = 'black'

    # Display current timestep using pyplot
    def render(self,t):
        plt.clf()
        for id in self.agents:
            # Is this agent in the scene during this timestep?
            if t in self.agents[id]:
                plt.axis([self.minx,self.maxx,self.miny,self.maxy])
                plt.scatter(self.agents[id][t][0],self.agents[id][t][1],color=tuple(self.colors[id]))
                for i in range(1,self.agent_hist+1):
                    if t-10*i in self.agents[id]:
                        plt.scatter(self.agents[id][t-10*i][0],self.agents[id][t-10*i][1],facecolors='none',color=tuple(self.colors[id]))
        plt.plot([self.start[0],self.goal[0]],[self.start[1],self.goal[1]],color=(0,1,0))
        plt.scatter(self.robot_pos[0],self.robot_pos[1],s=80,marker=(5,1),color=(1.0,1.0,0))
        plt.pause(0.05)

    # Create a list of n "equally spaced" rgb colors
    @staticmethod
    def get_spaced_colors(n):
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

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: %s <datafile_path> [<num_agents>]"%sys.argv[0])
        quit()

    num_agents = None
    if len(sys.argv) > 2:
        num_agents = int(sys.argv[2])

    a = human_agents(sys.argv[1],num_agents=num_agents,agent_hist=3)
    for t in np.arange(0,1000,10):
        a.render(t)
    plt.show()
