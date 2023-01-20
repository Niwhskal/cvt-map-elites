#!/usr/bin/env python

import sys
import math
import time
import numpy as np
from pyfastsim import *

class Environment():

    def __init__(self, dims, todisplay=False):
        self.N_timesteps = 3000
        self.init_x = 500
        self.init_y = 900

        self.goal_x = 500
        self.goal_y = 500

        self.GoalReachedDistance = 10
        self.ObstacleTooClose = 13

        self.todisplay = todisplay
        self.split_value = int(3000/dims)

    def initialize(self):

        if self.todisplay:
            self.settings = Settings('worlds/environ.xml')
            self.env_map = self.settings.map()
            self.robot = self.settings.robot()

            self.d = Display(self.env_map, self.robot)

        else:

            self.env_map = Map('worlds/original_maze.pbm', 1000)

            self.env_map.add_goal(Goal(500,500,30,0))

            self.robot = Robot(20.0, Posture(500, 920, 0))

            self.robot.add_laser(Laser(45, 1000.0))
            self.robot.add_laser(Laser(-45, 1000.0))
            self.robot.add_laser(Laser(0, 1000.0))

            self.robot.add_radar(Radar(0,4))

        self.startT = time.time()
        self.ts = 0
        self.distance_toGoal_list = []
        self.quitcounter = 0
        self.pos_list = []

    def simulate(self, nn):
        self.initialize()
        while self.ts < self.N_timesteps:
            if self.todisplay:
                self.d.update()
            pos = self.robot.get_pos()
            self.pos_list.append((pos.x()/1000, pos.y()/1000))
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

            if pos.x() == self.robot.get_pos().x() and pos.y() == self.robot.get_pos().y():
                self.quitcounter += 1
            if (pos.x() == self.robot.get_pos().x() and pos.y() == self.robot.get_pos().y()) and (self.quitcounter>50):
                break
            time.sleep(0.01)

        _lx = [i[0] for i in self.pos_list]
        _ly = [i[1] for i in self.pos_list]

        x_es = np.split(np.array(_lx), self.split_value)
        x_es = [np.mean(j) for j in x_es]

        y_es = np.split(np.array(_ly), self.split_value)
        y_es = [np.mean(j) for j in y_es]

        com_list = [(i,j) for i, j in zip(x_es, y_es)]

        return np.min(self.distance_toGoal_list), com_list


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
