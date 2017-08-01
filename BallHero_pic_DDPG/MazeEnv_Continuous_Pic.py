# Obstacle avoidence environment
# Continuous Action
# 7/24/2017

import math
import matplotlib.pyplot as plt
import numpy as np
import random
import time
import cv2

from collections import defaultdict
from euclid import Circle, Point2, Vector2, LineSegment2

import svg
from IPython.display import clear_output, display, HTML

class GameObject(object):
    def __init__(self, position, speed, obj_type, settings):
        """Esentially represents circles of different kinds, which have
        position and speed."""
        self.settings = settings
        self.radius = self.settings["object_radius"]
        
        # give hero different radius
        if obj_type == 'hero':
            self.radius = self.settings["hero_radius"]

        self.obj_type = obj_type
        self.position = position
        self.speed    = speed
        self.angle = 0    # only for continuous agent
        self.bounciness = 1.0
        self.dt = self.settings['sim_dt']

    def wall_collisions(self):
        """Update speed upon collision with the wall."""
        world_size = self.settings["world_size"]

        for dim in range(2):
            if self.position[dim] - self.radius       <= 0               and self.speed[dim] < 0:
                self.speed[dim] = - self.speed[dim] * self.bounciness
            elif self.position[dim] + self.radius + 1 >= world_size[dim] and self.speed[dim] > 0:
                self.speed[dim] = - self.speed[dim] * self.bounciness

    def move(self):
        """Move as if dt seconds passed"""
        self.position += self.dt * self.speed
        self.position = Point2(*self.position)

    def step(self):
        """Move and bounce of walls."""
        # Enable obstacles bounce wall
        self.wall_collisions()
        ################################################
        self.move()

    def as_circle(self):
        return Circle(self.position, float(self.radius))

    def draw(self):
        """Return svg object for this item."""
        color = self.settings["colors"][self.obj_type]
        return svg.Circle(self.position + Point2(10, 10), self.radius, color=color)

class MazeSimulator(object):
    def __init__(self, settings):
        """Initiallize game simulator with settings"""
        # to make it into episodic
        self.GameOver = False
        self.settings = settings
        self.size  = self.settings["world_size"]
        
        # walls
        self.walls = [LineSegment2(Point2(0,0),                        Point2(0,self.size[1])),
                      LineSegment2(Point2(0,self.size[1]),             Point2(self.size[0], self.size[1])),
                      LineSegment2(Point2(self.size[0], self.size[1]), Point2(self.size[0], 0)),
                      LineSegment2(Point2(self.size[0], 0),            Point2(0,0))]
        
        
        # Random hero inital position
        self.hero = GameObject(Point2(*self.settings["hero_initial_position"]),
                               Vector2(*self.settings["hero_initial_speed"]),
                               "hero",
                               self.settings)
        if not self.settings["hero_bounces_off_walls"]:
            self.hero.bounciness = 0.0

        # Initialize other Objects, OK
        self.objects = []
        for obj_type, number in settings["num_objects"].items():
            for _ in range(number):
                # This function used to Initialize other objects once a time
                self.spawn_object(obj_type)

        self.object_reward = 0
        self.collision_obj = None

    def perform_action1(self, action_id):
        """Change speed to one of hero vectors"""
        assert 0 <= action_id < self.num_actions
        self.hero.speed *= 0.7
        self.hero.speed += self.directions[action_id] * self.settings["delta_v"]
        
    def perform_action2(self, v_angle, v_line):
        dt = self.settings['sim_dt']
        # recover v_angle and v_line from normalized -1,1
        v_angle = v_angle * self.settings['max_angular_speed']
        v_line =  v_line  * self.settings['max_line_speed']
        self.hero.angle += dt * v_angle
        Xspeed = v_line * math.cos(self.hero.angle)
        Yspeed = v_line * math.sin(self.hero.angle)
        new_speed = Vector2(Xspeed, Yspeed)
        self.hero.speed = new_speed
    
    def perform_action3(self, x_acc, y_acc):
        dt = self.settings['sim_dt']
        limit = self.settings['max_line_acc']
        Xspd = self.hero.speed[0] + dt * x_acc * limit
        Yspd = self.hero.speed[1] + dt * y_acc * limit
        if Xspd > limit: Xspd = limit
        if Xspd < -limit: Xspd = -limit
        if Yspd > limit: Yspd = limit
        if Yspd < -limit: Yspd = -limit
            
        new_speed = Vector2(Xspd, Yspd)
        self.hero.speed = new_speed
    
    def perform_action4(self, Xspd, Yspd):
        spdAmp = self.settings('max_line_speed')
        new_speed = Vector2(Xspd, Yspd) * spdAmp
        self.hero.speed = new_speed

    def spawn_object(self, obj_type):
        """Spawn object of a given type and add it to the objects array"""
        radius = self.settings["object_radius"]
        hero_radius = self.settings["hero_radius"]
        # Randomly initialize object position, don't go out side of the wall
        # Keep a distance from the hero when do it
        spawn_clearance_squared = ((radius + hero_radius)*2)**2
        for i in range(100):
            position = np.random.uniform([radius, radius], np.array(self.size) - radius)
            if self.squared_distance(position, self.hero.position) > spawn_clearance_squared:
                break
        position = Point2(float(position[0]), float(position[1]))
        max_speed = np.array(self.settings["enemy_max_speed"])
        speed    = np.random.uniform(-max_speed, max_speed).astype(float)
        
        # move or not
        if self.settings['enemy_move']:
            speed = Vector2(float(speed[0]), float(speed[1]))
        else:
            speed = Vector2(0.0, 0.0)
        self.objects.append(GameObject(position, speed, obj_type, self.settings))
        
  
    def step(self, a1, a2):
        """Simulate all the objects for a given ammount of time.
           This step did everything
           1. perform action
           2. move objects
           3. make observation
           4. collect reward
           5. check done
        """
        # 1. perform action
        self.perform_action3(a1, a2)
        
        # 2. object movement and check
        for obj in self.objects + [self.hero] :
            obj.step()

        # 3. make observation
        observation = self.observe()
        
        # 4. collect reward
        self.resolve_collisions_eat()
        reward = self.collect_reward()
        
        # 5. check game over
        done = self.GameOver
        
        return observation, reward, done


    def resolve_collisions(self):
        """ hero collision into obstacle or wall, reward gets updated, and make a stop """
        ''' collision 1, if hero touch obstacle, episode terminate '''
        collision_distance = self.settings["object_radius"] + self.settings["hero_radius"]
        collision_distance2 = collision_distance**2
        
        # resolve collision with obstacles
        for obj in self.objects:
            if self.squared_distance(self.hero.position, obj.position) < collision_distance2:
                self.object_reward = self.settings['object_reward']['obstacle']
                # stop the hero
                self.hero.speed = Vector2(0.0, 0.0)
                # game over
                self.GameOver = True
                self.collision_obj = obj
                break
        
        # resolve collision into wall
        wall_collision_EPS = 1e-4
        if self.distance_to_walls() < wall_collision_EPS:
            self.object_reward = self.settings['object_reward']['wall']
            # The speed is resovled in GameObject class, no need to do it here
            self.GameOver = True
    
    def resolve_collisions_eat(self):
        ''' collision 2, eat, if hero touch, obstacle eat, not terminate  '''
        collision_distance = self.settings["object_radius"] + self.settings["hero_radius"]
        collision_distance2 = collision_distance**2
        
        # resolve collision with obstacles
        remove_list = []
        remove_counter = 0
        eat = False
        for obj in self.objects:
            if self.squared_distance(self.hero.position, obj.position) < collision_distance2:
                self.object_reward = self.settings['object_reward']['obstacle']
                # stop the hero
                self.hero.speed = Vector2(0.0, 0.0)
                # add to remove list
                remove_list.append(obj)
                remove_counter += 1
                #self.collision_obj = obj
                eat = True
                
        # resolve collision into wall
        wall_collision_EPS = 1e-4
        if self.distance_to_walls() < wall_collision_EPS:
            self.object_reward = self.settings['object_reward']['wall']
            
        # object remove and re-add
        if eat:
            for obj in remove_list:
                self.objects.remove(obj)
            for i in range(remove_counter):
                self.spawn_object('enemy')
                
                
    def observe0(self):
        # return low dimensional 
        heroPos = [float(self.hero.position[0]), float(self.hero.position[1])]
        objPos = []
        for obj in self.objects:
            x = float(obj.position[0])
            y = float(obj.position[1])
            objPos.append([x,y])
        return objPos
    
    def observe(self):
        # return a picture
        dim = self.settings['num_state']
        img = np.zeros((500, 500, 3), np.uint8)
        # draw obstacles
        for obj in self.objects:
            x = int(obj.position[0])
            y = int(obj.position[1])
            cv2.circle(img, (x, y), 15, (0,0,255), -1)
        # draw hero
        x = int(self.hero.position[0])
        y = int(self.hero.position[1])
        cv2.circle(img, (x, y), 10, (255,255,255), -1)
        
        img = cv2.cvtColor( img, cv2.COLOR_RGB2GRAY )        
        img = cv2.resize(img, (dim, dim))
        img = img/255.0
        img = np.reshape(img, [1, -1])
        
        return img

    def inside_walls(self, point):
        """Check if the point is inside the walls"""
        EPS = 1e-4
        return (EPS <= point[0] < self.size[0] - EPS and
                EPS <= point[1] < self.size[1] - EPS)
    
    def squared_distance(self, p1, p2):
        return (p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2
    
    def distance_to_walls(self):
        """Returns distance of a hero to walls"""
        res = float('inf')
        for wall in self.walls:
            res = min(res, self.hero.position.distance(wall))
        # distance to wall is the hero to wall
        return res - self.settings["hero_radius"]
        
        
    def collect_reward(self):
        """Return accumulated object eating score + current distance to walls score"""
        
        total_reward = self.object_reward 
        self.object_reward = 0
        return total_reward

    def _repr_html_(self):
        return self.to_html()

    def to_html(self, stats=[]):
        """Return svg representation of the simulator"""

        # draw stats
        stats = stats[:]
        
        stats.extend([
            "position = %.1f, %.1f" % (self.hero.position[0], self.hero.position[1],),    
        ])
        
        # Init Scene
        scene = svg.Scene((self.size[0] + 20, self.size[1] + 20 + 20 * len(stats)))
        
        # Draw Walls
        #scene.add(svg.Rectangle((10, 10), self.size))
        for wall in self.walls:
            scene.add(svg.Line(wall.p1 + Point2(10,10), wall.p2 + Point2(10,10)))

        # draw objects and hero
        for obj in self.objects + [self.hero] :
            scene.add(obj.draw())

        offset = self.size[1] + 15
        for txt in stats:
            scene.add(svg.Text((10, offset + 20), txt, 15))
            offset += 20

        return scene


    def draw(self, stats=[]):
        """
        An optional method to be triggered in simulate(...) to render the simulated environment.
        It is repeatedly called in each simulated iteration.
        simulate(...) will run with/without this method declared in the simulation class.
        """
        clear_output(wait=True)
        # to_html return the scene
        svg_html = self.to_html(stats)
        # display scene
        display(svg_html)
