
import numpy as np
import math
import random as rand

class Model():

    def __init__(self, g, l, k_fr, initial_conditions, T):
        """ Initializes system parameters 

        Args:
            g (float): Gravity constant
            l (float): Pendulum length
            k_fr (float): Friction coefitient
            initial_conditions (list): Initial conditions of system
            T (float): Time period of discrete system
        """
        if len(initial_conditions) != 2:
            print("System is second order so two initial conditions must be passed!")
            exit(1)

        # setting properties
        self.g = g
        self.l = l
        self.k_fr = k_fr
        self.T = T
        self.states = initial_conditions
        self.action = 0       
        

    def get_action(self):        

        return self.action

    def get_states(self):
        return self.states
        
    def next_states(self):
        """Simulates next state of state variables (angular position and angular velocity). 
        Model is given with reccurence equation, and for discretisation is used Euler I
        """
                
        # For computing next state, multiple steps procedure is used (fixed to 5, but can be modified). 
        # In case of not doing this, force (input action) will not affect angular position immediatly, because it apears only in second
        # equation (angular position is only affected by angular velocity which is affected by action/force).
        # This will lead to confusion in Q/A matrix, because some force apliance will not affect position change.

        for i in range(5):
            T_temp = (self.T/5)
            self.states[0] +=  T_temp* self.states[1]
            self.states[1] += T_temp*(-self.g/self.l*math.sin(self.states[0]) - self.k_fr * self.states[1] + self.get_action())

        # if angle is not in (0, 2pi) then rescaling is done
        if (self.states[0] < 0):
            self.states[0] = self.states[0] + 2*math.pi
        elif (self.states[0] > 2*math.pi):
            self.states[0] = self.states[0] - 2*math.pi
            

        return self.states


