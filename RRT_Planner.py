import numpy as np
from random import randint
import time
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pickle

class RRTplanner:
    def __init__(self, env, set_joints_fn, set_velocity, get_action_fn, collision_fn, obs_fn, xyz_obs_fn, dist_fn,\
                 d_thres, min_dist, coll_dist, box_granularity, start, bbox_start, bbox_end):
        '''
        This class takes in many functions to run RRT planner in joint angle space. Thus it moves the robot in joint angle space
        while also storing xyz,quarternion coordinates corresponding to each joint angle state.
        
        Parameters
        --------------------------
        env: Object of class mujoco env
        
        set_joints_fn: A function
             Will be called with env and the state to be set as argument. The dimension of the state will be same as that of
             bbox_start and bbox_end
             
        set_velocity: A function
            Will be called with env and the state velocity to be set as argument. Will be used to set velocity to zero.
            The dimension of the state velocity will be same as that of bbox_start and bbox_end
        
        get_action_fn: A function
            Will be called with env to get an action for the desired change in the joint angle. The returned value will be directly
            given to the env.step function and thus any manipulation of dimensions must happen here.
        
        collision_fn: A function
            Will be called after each env.step() function to see in collision happened.
        
        obs_fn: A function
            Will be called with env as argument. Must return the joint angles, of the same dimension as that of bbox_start.
        
        xyz_obs_fn: A function
            Will be called with env as argument. Must return the current xyz-quarternion coordinates of the environment.
        
        dist_fn: A function
            Need to take two joint state values as parameters and must return the distance between two joint states.
        
        d_thres: Float value
            The maximum distance between two RRT nodes. Distance is compared with the value returned by dist_fn
        
        min_dist: Float value
            The minimum distance between two RRT nodes. Distance is compared with the value returned by dist_fn
        
        coll_dist: 
            The minimum gap to be maintained from any environment objects
            
        box_granularity: Float value
            The sampling gap used when sampling a state space
        
        bbox_start: Array of float
            The start coordinates of the bounding box of the state space
            
        bbox_end: Array of float
            The end coordinates of the bounding box of the state space
        '''
        self.env = env
        self.set_joints_fn = set_joints_fn
        self.set_velocity = set_velocity
        self.get_action_fn = get_action_fn
        self.collision_check = collision_fn
        self.obs_fn = obs_fn
        self.xyz_obs_fn = xyz_obs_fn
        self.dist_fn = dist_fn
        self.d_thres = d_thres
        self.min_dist = min_dist
        self.coll_dist = coll_dist
        self.box_granularity = box_granularity
        
        self.start = start
        self.bbox_start = bbox_start
        self.bbox_end = bbox_end
        self.bbox_index = np.array(bbox_start) # Index we will use to keep track of which next box to sample on
        self.dof = len(bbox_start)
        
        self.nodes = []
        self.node_cost = {}
        self.nodes_parent = {}
        self.nodes_children = {}
        self.xyz_to_nodes = {}
        self.nodes_to_xyz = {}
        
        # Add the first node to the tree
        env.reset()
        self.set_joints_fn( self.env, self.start)
        w = self.obs_fn(self.env)
        self.nodes.append(w)
        self.nodes_parent[tuple(w)] = None
        self.nodes_children[tuple(w)] = []
        self.node_cost[tuple(w)] = 0
        
        self.xyz_to_nodes[tuple(self.xyz_obs_fn(self.env))] = [w]
        self.nodes_to_xyz[tuple(w)] = self.xyz_obs_fn(self.env)
        
        # Generate a set of boxes, which we will use to generate random samples to sample 
        
        
    def rand_sample(self):
        '''
        Generate a random sample in the state space. It divides the state space into hyper cubes and each time it is called,
        it takes the next hyper cube and samples a point from it. 
        '''
        r1 = self.bbox_index
        r2 = self.bbox_index + self.box_granularity
        for i in range(len(self.bbox_index)):
            if r2[i] > self.bbox_end[i]:
                r2[i] = self.bbox_end[i]
        sample = np.random.uniform(r1,r2)
        
        #increment the bbox_index so that next time we call rand_sample, we will sample from a different cube
        for i in range(len(self.bbox_index)):
            self.bbox_index[i] = self.bbox_index[i] + self.box_granularity
            if (self.bbox_index[i] >= self.bbox_end[i]):
                self.bbox_index[i] = self.bbox_start[i]
            else:
                break
        return sample
    
    def nearest(self, x_rand, nodes):
        '''
        Finds the nearest point to x_rand from the list of state space nodes. Uses dist_fn to get the distance metric.
        
        Parameters:
        ----------------
        x_rand: array of Float
            A state space value
        nodes: List of array of Float
            The list of state space nodes already visited
        '''
        min_dist = np.inf
        x_near = None
        for n in nodes:
            d = self.dist_fn(x_rand,n)#*2 + self.node_cost[tuple(n)]
            if (d < min_dist):
                x_near = np.copy(n)
                min_dist = d
        if (min_dist < self.min_dist):
            return None
        return x_near
    
    def steer(self,x_near,x_rand, scale=1):
        '''
        From x_near, tries to move the robot to x_rand using the action_fn. If a collision occurs or if it has travelled more than
        d_thres distance, it will search for a point farthest from x_near, atleast coll_dist from the stopping point (to maintain
        a minimum distance from object collision) and atleast min_dist away from x_near.
        
        Parameters:
        ----------------
        x_near: array of Float
            A state space node which we have already visited
        x_rand: array of Float
            A state space value, which we want to reach
        scale: Float
            How much to scale the action returned by action_fn
        '''
        self.set_joints_fn(self.env, x_near)
        self.set_velocity(self.env, np.zeros(len(x_near)))
        
        x_current = np.copy(x_near)
        collision = False
        direction = 0
        
        done = False
        buffer = []
        while True:
            action = self.get_action_fn(env=self.env, w_cur= x_current, w_next=x_rand)
            self.env.step(action*scale)
            x_current = self.obs_fn(self.env)
            
            # Check for collision. When collision occurs, we want to get last successful point which was
            # min_dist away from collision point in the path travelled
            if (self.collision_check(self.env) == True):
                print('collision')
                self.env.reset()
                collision = True
                break
             
            # Add the current point to buffer
            buffer.append(x_current)
            
            # Check if the current point is far away enough or close enough to random point
            if (self.dist_fn(x_current,x_near) > self.d_thres+self.coll_dist or \
                    self.dist_fn(x_current,x_rand) < self.min_dist/4 or \
                    len(buffer) > 500):
                break
            
        # We have a set of points in the buffer based on the path travelled
        # Select a point which is atleast min_distance/2 away from stopped point
        # This is useful to avoid obstacles
        for i in range(1, len(buffer)+1):
            if (self.dist_fn(x_current, buffer[-i]) > self.coll_dist and
               self.dist_fn(x_near, buffer[-i]) > self.min_dist):
                return buffer[-i], collision, direction
            if (self.dist_fn(x_near, buffer[-i]) < self.min_dist):
                break
        return None, collision, direction
    
    def run_rrt(self, num_cycles:int = 1):
        '''
        This function runs rrt algorithm to add more points to the rrt planner object. It will try
        to add points for the num_cycles mentioned. There is no guarantee that num_cycles points would
        be added to the graph. This is bult this way so that each run of this function is guaranteed to end.
        
        Parameters:
        ------------
        num_cycles : The number of times RRT algorithm is run, with each run sampling a random point and
                    trying to add one point to the graph
        '''
        start_time = time.perf_counter()
        for cycle in range(num_cycles):
            x_rand = self.rand_sample()
            x_near = self.nearest(x_rand, self.nodes)
            if x_near is None:
                continue
            
            xc, collision, direction = self.steer(x_near, x_rand)
            
            if (xc is None):
                continue
            
            self.set_joints_fn(self.env, xc)
            xyz_xc = self.xyz_obs_fn(self.env)
            
            self.nodes.append(xc)
            try:
                self.xyz_to_nodes[tuple(xyz_xc)].append(xc)
            except:
                self.xyz_to_nodes[tuple(xyz_xc)] = [xc]
            
            self.nodes_to_xyz[tuple(xc)] = xyz_xc

            self.nodes_parent[tuple(xc)] = x_near
            self.nodes_children[tuple(xc)] = []
            self.node_cost[tuple(xc)] = self.dist_fn(xc,x_near) + self.node_cost[tuple(x_near)]
            
            self.nodes_children[tuple(x_near)].append(xc)
        print('Took Time: %.3f sec for %.d cycles'%(time.perf_counter()-start_time,num_cycles) )
        
    def display_xyz_graph(self,x_lim, y_lim, z_lim, start=1, end = -1):
        
        #Get all the edges in the graph
        edge_list = []
        
        fig = plt.figure(figsize=(10,10))
        ax = fig.add_subplot(111, projection='3d')
        for n in self.nodes[start:end]:
            parent_n = self.nodes_parent[tuple(n)]
            x = self.nodes_to_xyz[tuple(n)][0:3]
            parent_x = self.nodes_to_xyz[tuple(parent_n)][0:3]
            
            ax.plot([x[0],parent_x[0]], [x[1],parent_x[1]], [x[2],parent_x[2]], c='r')
        ax.set_xlim(x_lim)
        ax.set_ylim(y_lim)
        ax.set_zlim(z_lim)
        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')
        plt.show()

    def save_planner(self, filename):
        dic = {}
        dic['start'] = self.start
        dic['bbox_start'] = self.bbox_start
        dic['bbox_end'] = self.bbox_end
        dic['bbox_index'] = self.bbox_index 
        dic['dof'] = self.dof

        dic['nodes'] = self.nodes
        dic['nodes_parent'] = self.nodes_parent
        dic['nodes_children'] = self.nodes_children
        dic['xyz_to_nodes'] = self.xyz_to_nodes
        dic['nodes_to_xyz'] = self.nodes_to_xyz

        with open(filename,'wb') as f:
            pickle.dump(dic,f)

    def load_planner(self, filename):

        with open(filename,'rb') as f:
            dic = pickle.load(f)
        self.start = dic['start']
        self.bbox_start = dic['bbox_start'] 
        self.bbox_end = dic['bbox_end']
        self.bbox_index = dic['bbox_index'] 
        self.dof = dic['dof']

        self.nodes = dic['nodes']
        self.nodes_parent = dic['nodes_parent']
        self.nodes_children = dic['nodes_children']
        self.xyz_to_nodes = dic['xyz_to_nodes']
        self.nodes_to_xyz = dic['nodes_to_xyz']

            