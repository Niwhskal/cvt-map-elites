#!/usr/bin/env python

import sys
import math
import time
import numpy as np
from pyfastsim import *

class Environment():

    def __init__(self):
        self.N_timesteps = 3000
        self.init_x = 500
        self.init_y = 900

        self.goal_x = 500
        self.goal_y = 500

        self.GoalReachedDistance = 10
        self.ObstacleTooClose = 13

    def initialize(self):
        self.settings = Settings('worlds/environ.xml')
        self.env_map = self.settings.map()
        self.robot = self.settings.robot()

        # self.d = Display(self.env_map, self.robot)

        self.startT = time.time()
        self.ts = 0
        self.distance_toGoal_list = []

    def simulate(self, nn):
        self.initialize()
        while self.ts < self.N_timesteps:
           # self.d.update()

            self.env_map.update(self.robot.get_pos())
            pos = self.robot.get_pos()
            dist2goal = math.sqrt((pos.x()-self.goal_x)**2+(pos.y()-self.goal_y)**2)
            self.distance_toGoal_list.append(dist2goal)
            if (dist2goal<self.GoalReachedDistance): # 30
              print('***** REWARD REACHED *****')
              print(f"Simulation completed in {time.time()- self.startT} s | Distance To Goal = {np.min(self.distance_toGoal_list)}")
              break

            lasers = self.robot.get_lasers()
            laserRanges = []
            for l in lasers:
              laserRanges.append(l.get_dist())

            radar = self.robot.get_radars()[0].get_activated_slice()

            bumperL = self.robot.get_left_bumper()
            bumperR = self.robot.get_right_bumper()


            inp = self.buildInputFromSensors(laserRanges, radar)

            motor_l, motor_r = nn.activate(inp)
            self.ts += 1
            self.robot.move(motor_l, motor_r, self.env_map, sticky_walls=False)
            time.sleep(0.01)

        return np.min(self.distance_toGoal_list), (pos.x()/1000, pos.y()/1000)


#--------------------------------------
    def buildInputFromSensors(self,laserRanges,radar):
        lrs = np.array(laserRanges)
        slice_sensors = np.zeros(4)
        slice_sensors[radar] = 1

        inps = np.concatenate((lrs,slice_sensors))

        return inps

#--------------------------------------

if __name__ == '__main__':
  random.seed()
  env = Environment()
  perf = env.simulate(nn)
