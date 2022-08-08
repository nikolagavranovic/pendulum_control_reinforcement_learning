import pygame.gfxdraw
import pygame
from enum import Enum
import math

class Color(Enum):
    WHITE = (255, 255, 255)
    BLACK = (0, 0, 0)
    RED = (200, 0, 0)
    YELLOW = (170, 180, 0)


class PendulumPlot():

    def __init__(self, speed = 10):
        # window and pendulum characteristics
        self.width = 360
        self.height = 360
        self.speed = speed
        self.surface = pygame.Surface((self.width, self.height))
        self.center = (self.width/2, self.height/2)
        self.pendulum_width = 15  # pendulum width in pixels
        self.pendulum_len = 150   # pendulum length in pixels
        
        pygame.init()
        self.display = pygame.display.set_mode(size = (self.width, self.height))
        self.clock = pygame.time.Clock()   
        

    def plot_pendulum_state(self, angle):
        """Plots pendulum state on windiow

        Args:
            angle (float): Angle of pendulum in radians
        """
        # initial position is when pendulum is positioned vertically downwards
        upper_left_init = (self.center[0] - self.pendulum_width/2, self.center[1])
        upper_right_init = (self.center[0] + self.pendulum_width/2, self.center[1])
        down_left_init = (self.center[0] + self.pendulum_width/2, self.center[1] + self.pendulum_len)
        down_right_init = (self.center[0] - self.pendulum_width/2, self.center[1] + self.pendulum_len)
        
        # rotate for angle
        upper_left = self.rotate_point(upper_left_init, self.center, angle)
        upper_right = self.rotate_point(upper_right_init, self.center, angle)
        down_left = self.rotate_point(down_left_init, self.center, angle)
        down_right = self.rotate_point(down_right_init, self.center, angle)

        pendulum_coordinates = [upper_left, upper_right, down_left, down_right]

        self.update_ui(pendulum_coordinates)
        self.clock.tick(self.speed)


    def rotate_point(self, p1, p2, angle):
        """_summary_

        Args:
            p1 (tuple): Point that rotates
            p2 (tuple): Point around p1 is rotated
            angle (float): Rotation angle, in radians

        Returns:
            tuple: Rotated point.
        """
        
        px, py = p1
        ox, oy = p2
        rot_x = math.cos(angle) * (px-ox) - math.sin(angle) * (py-oy) + ox
        rot_y = math.sin(angle) * (px-ox) + math.cos(angle) * (py-oy) + oy

        return (rot_x, rot_y)

    def update_ui(self, pendulum_coordinates):
        for event in pygame.event.get():  
            if event.type == pygame.QUIT:  
                quit()

        self.display.fill(Color.BLACK.value) 
        # firstly, filled polygon is drawn, and than its edges are drawn with antialiaser
        pygame.gfxdraw.filled_polygon(self.display, pendulum_coordinates, Color.YELLOW.value)
        pygame.gfxdraw.aapolygon(self.display, pendulum_coordinates, Color.YELLOW.value)      
        pygame.display.update() # updates changes 


