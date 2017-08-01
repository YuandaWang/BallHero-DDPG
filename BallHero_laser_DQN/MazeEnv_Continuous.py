# Obstacle avoidence environment
# Continuous Action
# 7/24/2017

import math
import matplotlib.pyplot as plt
import numpy as np
import random
import time

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
        self.GameSuccess = False
        ####################################
        self.settings = settings
        self.size  = self.settings["world_size"]
        # walls
        # maze walls
        walloffset = 0
        self.mazewalls = [LineSegment2(Point2(250, 0), Point2(250, 300-walloffset)),
                          LineSegment2(Point2(300, 0), Point2(300, 300-walloffset)),
                          LineSegment2(Point2(250, 300-walloffset), Point2(300, 300-walloffset)),
                          
                          LineSegment2(Point2(600, 200+walloffset), Point2(600, 500)),
                          LineSegment2(Point2(650, 200+walloffset), Point2(650, 500)),
                          LineSegment2(Point2(600, 200+walloffset), Point2(650, 200+walloffset)),]
                                  
        self.walls = [LineSegment2(Point2(0,0),                        Point2(0,self.size[1])),
                      LineSegment2(Point2(0,self.size[1]),             Point2(self.size[0], self.size[1])),
                      LineSegment2(Point2(self.size[0], self.size[1]), Point2(self.size[0], 0)),
                      LineSegment2(Point2(self.size[0], 0),            Point2(0,0))]
        
        # no maze walls in this environment
        #self.walls.extend(self.mazewalls)
        
        
        # Random hero inital position
        # never mind, just put it one the middle
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
        
        # Draw laser lines, no measurement
        self.observation_lines = self.generate_observation_lines()

        self.object_reward = 0
        self.collected_rewards = []
        self.scroll_times = 0
        self.hero_forward_distance = self.hero.position[0]
        self.maze_grid = np.zeros([3,2])
        self.maze_grid[0][0] = 1
        self.collision_obj = None
        
        # best distance to goal
        self.min_distance_exit = 1000000
        self.entrance_position = Point2(*self.settings["hero_initial_position"])
        self.exit_position = Point2(*self.settings["end_position"])

        # laser_size, new observation size
        self.laser_size = len(self.observation_lines)
        self.laser_buffer_size = len(self.observation_lines) * self.settings["laser_buffer_length"]
        ###########################################################################################

        # initalize a laser buffer
        self.laser_buffer = np.zeros([self.laser_size, self.settings["laser_buffer_length"]])
        #####################################################################################

        self.directions = [Vector2(*d) for d in [[1,0], [0,1], [-1,0], [0,-1], [0.0,0.0]]]
        self.num_actions      = len(self.directions)

        
    def laserbuffer_update(self, new_observation):
        laser_buffer_length = self.settings["laser_buffer_length"]
        temp = self.laser_buffer[:, 1:laser_buffer_length]
        new_laser_array = np.array([new_observation]).T
        self.laser_buffer = np.hstack((temp, new_laser_array))
        return self.laser_buffer.flatten()
    

    def perform_action(self, action_id):
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
        # make other objects not move, to become obstacles
        #####################################################
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
        #self.resolve_collisions()
        self.resolve_collisions_eat()
        reward = self.collect_reward(observation)
        
        # 5. check game over
        done = self.GameOver
        
        return observation, reward, done

    def squared_distance(self, p1, p2):
        return (p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2

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
                

    def inside_walls(self, point):
        """Check if the point is inside the walls"""
        EPS = 1e-4
        return (EPS <= point[0] < self.size[0] - EPS and
                EPS <= point[1] < self.size[1] - EPS)

    
    def observe(self):
        """Return observation vector. For all the observation directions it returns representation
        of the closest object to the hero - might be nothing, another object or a wall.
        Representation of observation for all the directions will be concatenated.
        """
        num_obj_types = len(self.settings["objects"]) + 1 # and wall
        max_speed_x, max_speed_y = self.settings["maximum_speed"]

        observable_distance = self.settings["observation_line_length"]
        
        # generate new observation lines, directions
        self.observation_lines = self.generate_observation_lines()
        

        # only consider relavent objects, can inside the view
        relevant_objects = [obj for obj in self.objects
                            if obj.position.distance(self.hero.position) < observable_distance]
        # objects sorted from closest to furthest
        relevant_objects.sort(key=lambda x: x.position.distance(self.hero.position))

        observation        = np.zeros(len(self.observation_lines))

        observation_offset = 0
        observation_raw_offset = 0
        
        # for every observation line
        for i, observation_line in enumerate(self.observation_lines):
            # shift to hero position
            observation_line = LineSegment2(self.hero.position + Vector2(*observation_line.p1),
                                            self.hero.position + Vector2(*observation_line.p2))
            observed_object = None
            # New observation code from here
            best_candidate = None
            # Deal with walls
            for wall in self.walls:
                candidate = observation_line.intersect(wall)
                if candidate is not None:
                    if (best_candidate is None or
                                best_candidate.distance(self.hero.position) >
                                candidate.distance(self.hero.position)):
                            best_candidate = candidate
                if best_candidate is None:
                    # assume it is due to rounding errors
                    # and wall is barely touching observation line
                    proximity_wall = observable_distance
                else:
                    proximity_wall = best_candidate.distance(self.hero.position)
                    
            # Deal with obstacles
            # find out observed_objects
            #print(relevant_objects)
            proximity_obj = observable_distance
            if relevant_objects is not None:
                for obj in relevant_objects:
                    if observation_line.distance(obj.position) < self.settings["object_radius"]:
                        observed_object = obj
                        break
                # find distance from hero to observed object
                if observed_object is not None:
                    intersection_segment = obj.as_circle().intersect(observation_line)
                    assert intersection_segment is not None
                    try:
                        # yes, one observation_line maybe intersect a circle twice
                        proximity_obj = min(intersection_segment.p1.distance(self.hero.position),
                                        intersection_segment.p2.distance(self.hero.position))
                    except AttributeError:
                        proximity_obj = observable_distance
                
            # nearer is what hero see
            proximity = min(proximity_wall, proximity_obj)
            
            observation[i] = proximity / observable_distance
            
        # after each observation, generate laser distance lines
        self.laser_lines = self.generate_laser_lines(observation)

        return observation

    def distance_to_walls(self):
        """Returns distance of a hero to walls"""
        res = float('inf')
        for wall in self.walls:
            res = min(res, self.hero.position.distance(wall))
        # distance to wall is the hero to wall
        return res - self.settings["hero_radius"]
        
        
    def collect_reward(self, observation):
        """Return accumulated object eating score + current distance to walls score"""
        
        # obstacle penalty
        nearest_obstacle = min(observation)
        if nearest_obstacle < self.settings['obstacle_tolerance']:
            norm_distance = nearest_obstacle / self.settings['obstacle_tolerance']
            obstacle_reward = -(1 - norm_distance)**2 
        else:
            obstacle_reward = 0.0
        
        obstacle_reward = 0
        lasting_reward = 0
        
        total_reward = self.object_reward + obstacle_reward + lasting_reward
        
        self.object_reward = 0
        self.collected_rewards.append(total_reward)
        return total_reward

    def generate_observation_lines(self):
        """Generate observation segments in settings["num_observation_lines"] directions"""
        # This function just generate the direction of observation, no distance measurement
        # for hero has angle, the observation lines should be generated before every observation
        result = []
        start = Point2(0.0, 0.0)
        end   = Point2(self.settings["observation_line_length"],
                       self.settings["observation_line_length"])
        #for angle in np.linspace(0, 2*np.pi, self.settings["num_observation_lines"], endpoint=False):
        start_angle = self.hero.angle + np.pi
        for angle in np.linspace(start_angle, 2*np.pi + start_angle, self.settings["num_observation_lines"], endpoint=False):
            rotation = Point2(math.cos(angle), math.sin(angle))
            current_start = Point2(start[0] * rotation[0], start[1] * rotation[1])
            current_end   = Point2(end[0]   * rotation[0], end[1]   * rotation[1])
            result.append( LineSegment2(current_start, current_end))
        return result

    
    """ New function: """
    def generate_laser_lines(self, distance):
        """ Generate laser lines, same structure as func:generate_generate_observation_lines """
        result = []
        line_K = self.settings["observation_line_length"]
        start = Point2(0.0, 0.0)
        start_angle = self.hero.angle + np.pi
        laser_index = 0
        #for angle in np.linspace(0, 2*np.pi, self.settings["num_observation_lines"], endpoint=False):
        for angle in np.linspace(start_angle, 2*np.pi+start_angle, self.settings["num_observation_lines"], endpoint=False):
            # different distance from distance
            line_length = distance[laser_index]*line_K
            laser_index += 1
            # Here should be float or Error
            end = Point2(float(line_length), float(line_length))
            rotation = Point2(math.cos(angle), math.sin(angle))
            current_start = Point2(start[0] * rotation[0], start[1] * rotation[1])
            current_end   = Point2(end[0]   * rotation[0], end[1]   * rotation[1])
            result.append( LineSegment2(current_start, current_end))
        return result
    ##################################################################################################
        
    def _repr_html_(self):
        return self.to_html()

    def to_html(self, stats=[]):
        """Return svg representation of the simulator"""

        # draw stats
        stats = stats[:]
        recent_reward = self.collected_rewards[-100:] + [0]
        
        stats.extend([
            #"nearest wall = %.1f" % (self.distance_to_walls(),),
            #"forward dist = %.1f"  % (self.hero_forward_distance),
            #"reward       = %.1f" % (self.collected_rewards[-1]),
            "position = %.1f, %.1f" % (self.hero.position[0], self.hero.position[1],),    
        ])
        
        # Init Scene
        scene = svg.Scene((self.size[0] + 20, self.size[1] + 20 + 20 * len(stats)))
        
        # Draw Walls
        #scene.add(svg.Rectangle((10, 10), self.size))
        for wall in self.walls:
            scene.add(svg.Line(wall.p1 + Point2(10,10), wall.p2 + Point2(10,10)))
        
        # Draw laser lines
        # for line in self.laser_lines:
        for idx, line in enumerate(self.laser_lines):
            # two params, two end points of the line
            
            if self.squared_distance(line.p1, line.p2) < (self.settings['observation_line_length']-1)**2 :
                scene.add(svg.Line(line.p1 + self.hero.position + Point2(10,10),
                                   line.p2 + self.hero.position + Point2(10,10)))
            # record heading line
            if idx == self.settings["num_observation_lines"]/2:
                headline_p1 = line.p1 + self.hero.position + Point2(10,10)
                headline_p2 = line.p2 + self.hero.position + Point2(10,10)
                
        # draw heading line that indicate moving direction
        scene.add(svg.Line2(headline_p1, headline_p2, color='red', width=5))
            
  
        # draw start and end point
        start_pos = self.settings["hero_initial_position"]
        end_pos = self.settings["end_position"]
        #scene.add(svg.Circle(Point2(*start_pos) + Point2(10,10), radius = 10, color='green'))
        scene.add(svg.Circle(Point2(*end_pos) + Point2(10,10), radius = 10, color='blue'))
        
        # draw objects and hero
        for obj in self.objects + [self.hero] :
            scene.add(obj.draw())
        
        # draw collision, can save files
        '''
        if self.GameOver and self.collision_obj is not None:
            scene.add(svg.Circle(self.collision_obj.position + Point2(10,10), radius = 30, color='yellow'))
        '''

        offset = self.size[1] + 15
        for txt in stats:
            scene.add(svg.Text((10, offset + 20), txt, 15))
            offset += 20

        return scene

    def setup_draw(self):
        """
        An optional method to be triggered in simulate(...) to initialise
        the figure handles for rendering.
        simulate(...) will run with/without this method declared in the simulation class
        As we are using SVG strings in KarpathyGame, it is not curently used.
        """
        pass

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
