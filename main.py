from pendulum_plot import PendulumPlot
from agent import Agent
from model import Model
import math
import matplotlib.pyplot as plt

if __name__ == '__main__':
    a = Agent()
    a.learn(n_of_epizodes=100, episode_to_plot_q=1000)
    
    print("Initial condition 1")
    a.model.states = [0.0, 0.0]
    a.control()

    print("Initial condition 2")
    a.model.states = [3.0, -0.5]
    a.control()
