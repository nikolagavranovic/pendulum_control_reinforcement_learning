import numpy as np
import math
import random as rand
from pendulum_plot import PendulumPlot
from model import Model
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt


class Agent():
    def __init__(self):
        # setting up model constants
        self.T = 0.1
        self.gamma = 0.9  # discount factor
        self.model = Model(9.81, 1, 0.0, [0.0, 0.0], self.T)
        self.p_plot = PendulumPlot(speed = 50)

        # Discretising state variables: anglular position and anglular velocity
        self.disc_angle = np.linspace(0, 2*math.pi, 65)
        self.disc_angle_rate = np.linspace(-math.pi, math.pi, 33)
        # descrete actions (force intensity): [-3.0, 0.0, 3.0]
        # or push to one direction, push to oposite direction and do nothing
        # can be discretised more smoothly, for example: [-3, -2.5, -2.0 .... 3]
        # which leads to more dimensions in QA matrix and training is more complex. 
        # For simple purpose of controling pendulum in up state, 
        self.disc_action = np.linspace(-3, 3, 3)  
        self.initialize_q_matrix()

    def initialize_q_matrix(self):
        # q matrix is initialized with rewards of each state regardles of action for first iteration
        # this is better than initializing all with zeros, although it should work both ways
        self.q_table = np.zeros(shape = (len(self.disc_angle), len(self.disc_angle_rate), len(self.disc_action)))
        for i in range(len(self.disc_angle)):
            for j in range(len(self.disc_angle_rate)):
                for k in range(len(self.disc_action)):
                    self.q_table[i, j, k] = self.cost_function(self.disc_angle[i], self.disc_angle_rate[j]) 

    def cost_function(self, x, dx):     
        # penality is less if pendulum is closer to pi (up position) and if velocity is 0
        # although is more important that direction is upwards
        cost = (math.pi - x) ** 2 + 0.25*(dx)**2        
        return cost

    def play_episode(self, epsilon = 0.05, niter = 2000, plot_pendulum = False, bonus_flag = False, show_info_flag = False):
        """ Simulates one continous episode of training.

        Args:
            epsilon (float): Probability of choosing random action, due to exploration. Defaults to 0.05.
            niter (int): Number of iterations (with sample time T) in episode. Defaults to 2000.
            plot_pendulum (bool): Show animation. Defaults to False.
            bonus_flag (bool): Award bonus if upward state with 0 velocity. Defaults to False.
            show_info_flag (bool): Show info after 500th iteration. Defaults to False.
        """
        self.model.states = [0.0, 0.0]  # resets both angular position and velocity to 0
        
        for i in range(niter):
            
            x, dx = self.model.get_states()
            # find the discrete state sistem is closest to
            x_disc_ind = np.argmin(abs(self.disc_angle - x))   # smallest difference of current state to the closest discrete state
            dx_disc_ind = np.argmin(abs(self.disc_angle_rate - dx))

            reward = self.cost_function(self.disc_angle[x_disc_ind], self.disc_angle_rate[dx_disc_ind])
            
            # epsilon is probability that random action is choosen
            c = np.random.choice([0, 1], p = [epsilon, 1 - epsilon])

            if c == 0:
                action_ind = np.random.choice(range(len(self.disc_action))) # choose random action
            else:
                # finding index of minimal action for given states value (it can be changed so that the best action is for maximum
                # if cost function is defined differently)
                action_ind = int(np.argmin(self.q_table[x_disc_ind, dx_disc_ind, :])) 

            action = self.disc_action[action_ind] 

            self.model.action = action

            xnew, dxnew = self.model.next_states()
            xnew_disc_ind = np.argmin(abs(self.disc_angle - xnew))   # smallest difference of current state to the closest discrete state
            dxnew_disc_ind = np.argmin(abs(self.disc_angle_rate - dxnew))
            
            bonus = 0
            if bonus_flag:
                # reward for the state in which velocity is 0 and angle is pi
                # computed state variables has to be in some (discrete) range - there is some tolerance  
                if  self.disc_angle_rate[dxnew_disc_ind] == 0:
                    if abs(self.disc_angle[xnew_disc_ind] - math.pi) < 0.02:  # due to numerical error, condition is defined this way
                        bonus = -10 # setting big negative reward for state and action that led to this state
                        # so that all other states would be motivated to get here

            # updating values for action in state is done according to Belmann's equation. 
            self.q_table[x_disc_ind, dx_disc_ind, action_ind] = reward + self.gamma * min(self.q_table[xnew_disc_ind, dxnew_disc_ind, :]) + bonus

            if plot_pendulum:
                self.p_plot.plot_pendulum_state(xnew)

            if show_info_flag:
                if (i + 1)% 2000 == 0:
                    print(f"Iteration {i}, epsilon {epsilon}")
                    if sum(sum(sum(self.q_table < 0))) == 0:
                        min_value = min(self.q_table[self.q_table > 0])
                    else:
                        min_value = min(self.q_table[self.q_table < 0])

                    x, dx, a = np.where(self.q_table == min_value)
                    print(f"Minimum q value: {min_value} is for\n angle {self.disc_angle[x]}, angle rate {self.disc_angle_rate[dx]} and action {self.disc_action[a]}")
                    print("-----------------------------------")
                    print(f"number of unexplored fields is {len(self.q_table[self.q_table == 0])}")  # works only if all fields are initilized to zero, not when cost function is apllied


    def learn(self, n_of_epizodes = 100, episode_to_plot_q = 100):
        epsilon_start = 0.1
        epsilon_stop = 0.0
        epsilon_step = (epsilon_start - epsilon_stop)/(n_of_epizodes)
        epsilon = 0.1
        for i in range(n_of_epizodes):
            self.play_episode(epsilon=epsilon) 
            print(f"Episode {i + 1} over...")
            epsilon -= epsilon_step
            print(f"Average value of q matrix is: {np.average(self.q_table)}")
            if (i + 1) %  episode_to_plot_q == 0:
                self.plot_q_table()
                self.plot_optimal_policy()


    def control(self, simulation_length = 1000):
        for i in range(simulation_length):
            x, dx = self.model.get_states()
            x_disc_ind = np.argmin(abs(self.disc_angle - x))   # smallest difference of current state to the closest discrete state
            dx_disc_ind = np.argmin(abs(self.disc_angle_rate - dx))
            action_ind = int(np.argmin(self.q_table[x_disc_ind, dx_disc_ind, :])) # nalazimo indeks minimalne vrednosti stanja (one koja je najmanje kaznjiva, ovo se moze promeniti kasnije da bude nagrada umesto kazne)
            action = self.disc_action[action_ind] 

            self.model.action = action

            xnew, dxnew = self.model.next_states()
            self.p_plot.plot_pendulum_state(xnew)

    def plot_q_table(self):
        q_to_plot = pd.DataFrame(np.min(self.q_table, 2))  ## minimum po dimenziji koja se odnosi na akcija, odnosno nalazimo vrednosti svakog stanja
        
        s = sns.heatmap(q_to_plot, annot=True, fmt = ".0f", cmap='Blues', xticklabels=np.round(self.disc_angle_rate, 3), yticklabels=np.round(self.disc_angle, 4))
        s.set(xlabel='angular speed', ylabel='angular position')
        plt.show()
    
    def plot_optimal_policy(self):
        q_to_plot = pd.DataFrame(np.argmin(self.q_table, axis = 2))  ## minimum po dimenziji koja se odnosi na akcija, odnosno nalazimo vrednosti svakog stanja
        
        s = sns.heatmap(q_to_plot, annot=True, fmt = ".2f", cmap='Blues', xticklabels=np.round(self.disc_angle_rate, 3), yticklabels=np.round(self.disc_angle, 4))
        s.set(xlabel='angular speed', ylabel='angular position')
        plt.show()

    def plot_table(self, table):       
        s = sns.heatmap( pd.DataFrame(table) , annot=True, fmt = ".2f", cmap='Blues', xticklabels=np.round(self.disc_angle_rate, 3), yticklabels=np.round(self.disc_angle, 4))
        s.set(xlabel='angular speed', ylabel='angular position')
        plt.show()   
            
        

