import pygame
import sys
import numpy as np
from pygame.locals import K_t, K_g, K_w, K_s, K_o, K_l, K_x, KEYDOWN
from shapely.geometry import LineString 
from shapely.geometry import Point 
import random, time
import matplotlib.pyplot as plt
from keras.models import Sequential,Model
from keras.layers import Dense,LSTM,Input

FPS = 100 #Frames per second
margin = 30
SIZE_PANEL = 100
SIZE_SCREEN = width, height = 1000+SIZE_PANEL, 780
NUM_SENSORS = 12
THRESHOLD = 200

#Colours
WHITE = 0xffffff
BLACK = 0x000000
GREEN = 0,255,0
RED = 255,0,0
BLUE = 0,0,255
GREY = 90,90,90
LIGHT_GREY = 150,150,150
LIGHT_YELLOW = 0xfff5c2
COLOUR_CONT = BLACK
LINE_COLOUR = WHITE
DUST_COLOUR = LIGHT_YELLOW
COLOUR_ROBOT = GREY
COLLISION_COLOUR = RED
LIMIT_SPEED=10

pygame.init() #Initializing library
screen = pygame.display.set_mode(SIZE_SCREEN) # Initializing screen
FPSCLOCK = pygame.time.Clock() #Refreshing screen rate
fontObj = pygame.font.Font("C:/Windows/Fonts/comicbd.ttf", 10) #Font of the messages
fontObj2 = pygame.font.Font("C:/Windows/Fonts/comicbd.ttf", 15) #Font of the messages


class Robot():
    def __init__(self, init_pos=[100, 100], length=30, brain='simple', init_direct=0, limit_speed=LIMIT_SPEED):
        self.position = np.asarray(init_pos).astype(int)
        self.length = int(length) # The size will affect how the robot behaves (separation between wheels)
        self.direction = init_direct
        self.speed_right = 0
        self.speed_left = 0
        self.distance_sensors = THRESHOLD
        self.value_sensors = [0]*NUM_SENSORS
        if brain=='LSTM':
            self.brain = NeuralNet2()
        else:
            self.brain = NeuralNet()
        self.limit_speed = limit_speed
        # fitness function parameters
        self.dust_count = 0       # see clean_dust function
        self.collision_count = 0  # see draw_robot function
        self.sensor_reward = 0    # see use_sensors function
        self.trajectory = []


    def use_brain(self):
        # Neural Network move

        self.speed_left = int(self.brain.feed_forward(self.value_sensors)[0][0]*10)
        self.speed_right = int(self.brain.feed_forward(self.value_sensors)[0][1]*10)

        if self.speed_right > self.limit_speed:
            self.speed_right = self.limit_speed
        if self.speed_right < -self.limit_speed:
            self.speed_right = -self.limit_speed
        if self.speed_left > self.limit_speed:
            self.speed_left = self.limit_speed
        if self.speed_left < -self.limit_speed:
            self.speed_left = -self.limit_speed

    def use_sensors(self, lines_env):
        angle = self.direction
        sensors = [None]*NUM_SENSORS
        for i in range(NUM_SENSORS):
            # Draw lines
            sensor_x = int(self.position[0] + (self.length) * np.cos(angle)) # self.length indicates how long has to be the sensor outside of the circle
            sensor_y = int(self.position[1] + (self.length) * np.sin(angle))
            start_x = int(self.position[0] + (self.distance_sensors+self.length) * np.cos(angle))
            start_y = int(self.position[1]) + (self.distance_sensors+self.length) * np.sin(angle)
            sensors[i]=(pygame.draw.line(screen, LINE_COLOUR, (start_x, start_y), (sensor_x, sensor_y), 1))
            angle += np.pi/6

            # Check intersections
            line_sensor = LineString([[start_x, start_y], [sensor_x, sensor_y]])
            min_distance = THRESHOLD
            for j in range(len(lines_env)):
                # Creating the environment line
                line_env=LineString([lines_env[j][0], lines_env[j][1]])
                # If collision -> Take value
                if str(line_sensor.intersection(line_env))!="LINESTRING EMPTY":
                    point = Point(self.position)
                    if int(np.round(point.distance(line_sensor.intersection(line_env))-self.length)) < min_distance:
                        min_distance = int(np.round(point.distance(line_sensor.intersection(line_env))-self.length))
            self.value_sensors[i] = min_distance

            # for each sensor distance that is less than 30 (but not too close), give a reward
            if self.value_sensors[i] < 30 and self.value_sensors[i] > 10:
                self.sensor_reward += 1


    def draw_robot(self, coll_flag):
        #Colours for collision
        if coll_flag:
            colour_robot  = COLLISION_COLOUR
            self.collision_count +=1
        else: colour_robot = COLOUR_ROBOT

        #Body of the robot
        pygame.draw.circle(screen, WHITE, (int(self.position[0]), int(self.position[1])), self.length) #Background
        coord_robot = pygame.draw.circle(screen, colour_robot, (int(self.position[0]), int(self.position[1])), self.length, 2) #Out line
        #Head of the robot
        head_x = self.position[0]+(self.length/1.3)*np.cos(self.direction)
        head_y = self.position[1]+(self.length/1.3)*np.sin(self.direction)
        pygame.draw.line(screen, GREY, self.position, [head_x, head_y], 2)
        #Sensors of the robot
        angle = self.direction
        for val in self.value_sensors:
            pos_x = self.position[0]+(self.length/0.7)*np.cos(angle)
            pos_y = self.position[1]+(self.length/0.7)*np.sin(angle)
            textSurfaceObj = fontObj.render(str(val), True, LIGHT_GREY, WHITE) #
            textRectObj = textSurfaceObj.get_rect()                             #
            textRectObj.center = (pos_x, pos_y)
            screen.blit(textSurfaceObj, textRectObj)
            angle += np.pi/6
        #Velocities of the motors
        pos_x = self.position[0]+(self.length/2)*np.cos(self.direction-np.pi/2)
        pos_y = self.position[1]+(self.length/2)*np.sin(self.direction-np.pi/2)
        textSurfaceObj = fontObj2.render(str(self.speed_right), True, LIGHT_GREY, WHITE) # Left motor
        textRectObj = textSurfaceObj.get_rect()                            #
        textRectObj.center = (pos_x, pos_y)                                #
        screen.blit(textSurfaceObj, textRectObj)                           #
        pos_x = self.position[0]+(self.length/2)*np.cos(self.direction+np.pi/2)
        pos_y = self.position[1]+(self.length/2)*np.sin(self.direction+np.pi/2)
        textSurfaceObj = fontObj2.render(str(self.speed_left), True, LIGHT_GREY, WHITE) # Right motor
        textRectObj = textSurfaceObj.get_rect()                            #
        textRectObj.center = (pos_x, pos_y)                                #
        screen.blit(textSurfaceObj, textRectObj)                           #
        return coord_robot

    def update_velocities(self, inc_right, inc_left):
        self.speed_right += inc_right
        self.speed_left += inc_left
        # Limit in the speed
        if self.speed_right > self.limit_speed:
            self.speed_right = self.limit_speed
        if self.speed_right < -self.limit_speed:
            self.speed_right = -self.limit_speed
        if self.speed_left > self.limit_speed:
            self.speed_left = self.limit_speed
        if self.speed_left < -self.limit_speed:
            self.speed_left = -self.limit_speed

    def update_pos(self, lines_env):

        #Change speed
        speed = (self.speed_right + self.speed_left)/2
        try:
            R = (1/2)*((self.speed_left+self.speed_right)/(self.speed_right-self.speed_left))
        except:
            R = 32767
        w = (self.speed_right-self.speed_left)/(self.length*2)

        # Timesteps
        param = 1
        new_position = [self.position[0]+speed*np.cos(self.direction),self.position[1]+speed*np.sin(self.direction)]

        # Calculating ICC
        ICC_x = new_position[0]-R*np.sin(self.direction)
        ICC_y = new_position[1]+R*np.cos(self.direction)
        # Matrixes involved
        rot_mat = np.asarray([[np.cos(w*param), -np.sin(w*param), 0],
                            [np.sin(w*param), np.cos(w*param), 0],
                            [0, 0, 1]])
        second_mat = [new_position[0]-ICC_x, new_position[1]-ICC_y, self.direction]
        third_mat = [ICC_x, ICC_y, w*param]
        new_position[0], new_position[1], temp_direction = np.dot(rot_mat, second_mat)+third_mat

        coll_flag = self.collision_handling_complex(lines_env,new_position,speed)
        self.direction = temp_direction
        return coll_flag

    def collision_handling_complex(self, lines_env,new_position,speed):
        index_walls = []
        coll_flag = False
        for index,line in enumerate(lines_env):

            start = Point([line[0][0], line[0][1]])
            end = Point([line[1][0], line[1][1]])
            line_string = LineString([start,end])

            center = Point(new_position)
            circle = center.buffer(self.length)
            old_center = Point(self.position)
            if (circle.intersects(line_string) and center.distance(line_string) < self.length-0.01) or LineString([old_center,center]).intersects(line_string):
                index_walls.append(index)
                coll_flag = True

        if len(index_walls) == 0:
            self.position = new_position
            return

        elif len(index_walls) == 2:
            dists = []
            for i in range(2):
                start = Point([lines_env[index_walls[i]][0][0], lines_env[index_walls[i]][0][1]])
                end = Point([lines_env[index_walls[i]][1][0], lines_env[index_walls[i]][1][1]])
                try:
                    line_angle = np.arctan((start.y-end.y)/(start.x-end.x))
                except:
                    line_angle = np.pi/2
                if start.distance(Point(self.position)) < end.distance(Point(self.position)):
                    dists.append(Point(start.x+20*np.cos(line_angle),start.y+20*np.sin(line_angle)).distance(Point(self.position)))
                else:
                    dists.append(Point(end.x+20*np.cos(line_angle),end.y+20*np.sin(line_angle)).distance(Point(self.position)))

            if dists[0] > dists[1]:
                temp = index_walls[0]
                index_walls[0] = index_walls[1]
                index_walls[1] = temp

        #for index in index_walls:
        for i in range(len(index_walls)):
            index = index_walls[i]
            start = Point([lines_env[index][0][0], lines_env[index][0][1]])
            end = Point([lines_env[index][1][0], lines_env[index][1][1]])

            line_string = LineString([start,end])

            if (circle.intersects(line_string) and center.distance(line_string) < self.length-0.01) or LineString([old_center,center]).intersects(line_string):
                try:
                    line_angle = np.arctan((end.y-start.y)/(start.x-end.x))
                except:
                    line_angle = np.pi/2

                collision_angle = np.pi+line_angle-self.direction
                parallel_speed = speed*np.cos(collision_angle)
                perpendicular_speed = speed*np.sin(collision_angle)
                i = 0
                while Point(self.position).distance(line_string.interpolate(line_string.project(Point(self.position))))-self.length > 0.88:
                    self.position[0] = self.position[0] + np.cos(np.pi/2+line_angle)*(1/perpendicular_speed)
                    self.position[1] = self.position[1] - np.sin(np.pi/2+line_angle)*(1/perpendicular_speed)
                    i += 1
                    if i >= perpendicular_speed:
                        break

                new_x = self.position[0] - np.cos(line_angle)*parallel_speed
                new_y = self.position[1] - np.sin(line_angle)*parallel_speed

                center = Point([new_x,self.position[1]])
                circle = center.buffer(self.length)
                intersect = False
                for index2,line2 in enumerate(lines_env):
                    start2 = Point([line2[0][0], line2[0][1]])
                    end2 = Point([line2[1][0], line2[1][1]])
                    line_string2 = LineString([start2,end2])
                    if circle.intersects(line_string2) and index2 is not index:
                        coll_flag = True
                        intersect = True
                if not intersect:
                    self.position[0] = new_x

                center = Point([self.position[0],new_y])
                circle = center.buffer(self.length)
                intersect = False
                for index2,line2 in enumerate(lines_env):
                    start2 = Point([line2[0][0], line2[0][1]])
                    end2 = Point([line2[1][0], line2[1][1]])
                    line_string2 = LineString([start2,end2])
                    if circle.intersects(line_string2) and index2 is not index:
                        coll_flag = True
                        intersect = True
                if not intersect:
                    self.position[1] = new_y
                center = Point(self.position)
                circle = center.buffer(self.length)
                return coll_flag

    def stop_robot(self):
        self.speed_right = 0
        self.speed_left = 0


class Environment():
    def __init__(self, points,margin):
        self.points = points
        self.lines = []
        self.dust = np.asarray([[DUST_COLOUR]*height]*width)
        self.margin = margin

    def draw_env(self):
        for i in range(width):
            pygame.PixelArray(screen)[i] = self.dust[i].tolist()
        pygame.draw.rect(screen, WHITE, [0,0,self.margin,height])
        pygame.draw.rect(screen, WHITE, [0,0,width,self.margin])
        pygame.draw.rect(screen, WHITE, [0,height-self.margin,width,height])
        pygame.draw.rect(screen, WHITE, [width-(self.margin+SIZE_PANEL),0+self.margin,width,height])

        self.lines = []
        for point_list in self.points:
            prev = None
            for point in point_list:
                if prev is not None:
                    pygame.draw.line(screen, GREY, prev, point, 2)
                    self.lines.append([prev, point])
                prev = point

    def clean_dust(self, coords, robot):
        mask = self.create_circular_mask(height, width, [coords[0]+robot.length, coords[1]+robot.length], robot.length)
        temp = self.dust[mask]
        self.dust[mask] = WHITE

        for i in range(len(temp)):
            if temp[i] != WHITE:
                robot.dust_count += 1

    def create_circular_mask(self, w, h, center=None, radius=None):
        if center is None: # use the middle of the image
            center = (int(w/2), int(h/2))
        if radius is None: # use the smallest distance between the center and image walls
            radius = min(center[0], center[1], w-center[0], h-center[1])
        X,Y = np.ogrid[:h, :w]
        dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)
        mask = dist_from_center <= radius
        return mask

class NeuralNet:
    def __init__(self,num_inputs=NUM_SENSORS, num_outputs=2, num_biases=1):
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.weights = 0
        self.biases = np.random.rand(num_biases,self.num_outputs)

    def feed_forward(self, sensors):
        normalized = np.asarray(sensors) / THRESHOLD
        return np.dot(normalized,self.weights) + self.biases

    def set_weights_from_EA(self, weights):
        self.weights = np.reshape(weights, (NUM_SENSORS, 2))


class NeuralNet2:
    def __init__(self, num_inputs=12, num_hidden=4, num_outputs=2):
        self.model = Sequential()
        self.model.add(LSTM(num_hidden, input_shape=(num_inputs, 1)))
        self.model.add(Dense(num_outputs, activation='linear'))
        self.model.compile(loss='mean_squared_error', optimizer='adam')
        self.model.summary()

    def feed_forward(self, sensors):
        sensors = np.reshape(sensors, (1, 12, 1))
        return self.model.predict(sensors)

    def set_weights_from_EA(self, weights):
        weights = np.asarray(weights)
        inp = np.reshape(weights[:16], (1, 16))
        lstm = np.reshape(weights[16:80], (4,16))
        lstm_bias = np.asarray([0., 0., 0., 0., 1., 1., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0.])
        output = np.reshape(weights[-10:-2], (4, 2))
        output_bias = np.asarray([0., 0.])
        
        layer1 = [inp, lstm, lstm_bias]
        layer2 = [output, output_bias]
        
        #Replacing new weights
        self.model.layers[0].set_weights(layer1)
        self.model.layers[1].set_weights(layer2)


# Evoluationary Algorithm class
class EvolutionaryAlgorithm():
    def __init__(self, population_size, NN, min_gen=-1, max_gen=1):
        if NN=='LSTM':
            number_genes=106
        else:
            number_genes=NUM_SENSORS*2
        self.number_genes = number_genes
        self.range_genes = [min_gen, max_gen]
        self.individuals = [[np.round(random.uniform(min_gen, max_gen), 3)  for _  in range(number_genes)] for _ in range(population_size)]
        self.evaluations = [None]*population_size
        self.rank = [None]*population_size


    def calc_diversity(self):
        distance = 0
        for indiv1 in self.individuals:
            for indiv2 in self.individuals:
                distance += np.linalg.norm(np.asarray(indiv1)-np.asarray(indiv2))
        distance = distance/len(self.individuals)
        return distance


    #Save weights of NN of individuals
    def save_individuals(self, name='weightsEA', path=''):
        np.save(path+name, np.asarray(self.individuals), allow_pickle=False)

    #Load weights of NN of individuals
    def load_individuals(self, name='weightsEA.npy', path=''):
        self.individuals=np.load(path+name).tolist()

    def fitnessFuction(self, robot):
        score = 0
        
        #  > No collision
        score -= robot.collision_count*1000
        #  > Close to the walls
        score += robot.sensor_reward*10
        #  > Clean dust
        score += robot.dust_count*50

        # Negative value, as the selection function picks the smallest values
        return -score

    # You have to indicate the function to asses the good/bad performance 
    def evaluation(self,robots):
        assert len(robots) == len(self.evaluations)
        for i in range(len(robots)):
            self.evaluations[i] = self.fitnessFuction(robots[i])
        #Maximum and mean values
        maxim = -np.asarray(self.evaluations).min()
        mean = -np.asarray(self.evaluations).mean()
        return maxim, mean

    # using truncate rank-based selection (number of individuals as input)
    def selection(self, number_individuals):
        # Taking the n best individuals
        evalu, indiv = zip(*sorted(zip(self.evaluations, self.individuals))) # Sorting
        best_inidividuals = indiv[:int(number_individuals)]
        best_values = evalu[:int(number_individuals)]
        #Keeping only the best ones -> The others = None
        for i in range(len(self.individuals)):
            try:
                self.individuals[i] = best_inidividuals[i]
                self.evaluations[i] = best_values[i]
            except:
                self.individuals[i] = None
                self.evaluations[i] = None


    def reproduction(self, crossover='Arithmetic'):
        # Types of crossover:
        #   - One point
        #   - Uniform
        #   - Arithmetic
        #   - Sequences

        # Excluding None individuals
        current_individuals = [i for i in self.individuals if i is not None]
        idx = 0
        # Creating new individuals
        for i in range(len(self.individuals)):
            if self.individuals[i] is not None: continue
            # Picking parents in order -> To individuals have the same number of offsprings
            try: dad = current_individuals[idx]
            except:
                idx = 0
                dad = current_individuals[idx]
            idx += 1
            try: mom = current_individuals[idx]
            except:
                idx = 0
                mom = current_individuals[idx]
            if crossover=='Arithmetic':
                self.individuals[i] = ((np.asarray(dad)+np.asarray(mom))/2).tolist() # ARITHMETIC CREATION
            elif  crossover=='Uniform':
                self.individuals[i] = []
                for j in range(len(dad)):
                    if j%2==0: self.individuals[i].append(dad[j])
                    else: self.individuals[i].append(mom[j])

    def mutation(self, mutation_range=1, mutation_rate='0.05'):
        for indiv in self.individuals:  # Iterating individuals
            for i in range(len(indiv)): # Iterating genes
                if random.random() <= mutation_rate: # Probability of mutation
                    indiv[i] += random.uniform(-mutation_range, mutation_range)
                    if indiv[i] < self.range_genes[0]: indiv[i] = self.range_genes[0]
                    elif indiv[i] > self.range_genes[1]: indiv[i] = self.range_genes[1]


def main(num_room, num_robots, NN, num_generations):
    # nn = NeuralNet2(num_inputs=12, num_hidden=4, num_outputs=2)
    # nn.set_weights_from_EA(np.ones((106,)))
    # print("DONE!!")
    # Initial checking
    if num_robots<2:
        print("\n\n# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #")
        print("#                                                                     #")
        print("#   To use the Evolutionary Algorithm you must use 2 or more robots   #")
        print("#                                                                     #")
        print("# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #\n\n")
        sys.exit(1)

    points = []

    #### ROOMS ####
    points.append([[0+margin,0+margin], [width-margin-SIZE_PANEL,0+margin], [width-margin-SIZE_PANEL,height-margin], [0+margin,height-margin],[0+margin,0+margin]])
    if num_room == 0:
        points.append([[0+200,0+200], [width-200-SIZE_PANEL,0+200], [width-200-SIZE_PANEL,height-200], [0+200,height-200],[0+200,0+200]])
    elif num_room == 1:
        points.append([[0+margin,0+150], [width-200-SIZE_PANEL,0+150], [width-200-SIZE_PANEL,0+300], [0+margin,0+300]])
        points.append([[width-margin-SIZE_PANEL,0+450],[0+margin+200,0+450],[0+margin+200,0+600],[width-margin-SIZE_PANEL,0+600]])
    elif num_room == 2:
        points.append([[0+margin+150,150+margin], [0+margin+150,height-margin-150], [0+margin+300,height-margin-150], [0+margin+300,150+margin], [150+margin, 150+margin]])
        points.append([[width-margin-SIZE_PANEL,0+150],[0+margin+500,0+150],[0+margin+500,0+600],[width-margin-SIZE_PANEL,0+600]])
    elif num_room == 3:
        points.append([[0+margin,0+200], [0+margin+400,0+200], [0+margin+400,0+300], [0+margin,0+300]])
        points.append([[width-margin-SIZE_PANEL,0+200], [width-margin-SIZE_PANEL-400,0+200], [width-margin-SIZE_PANEL-400,0+300], [width-margin-SIZE_PANEL,0+300]])
        points.append([[0+margin+200,0+450], [width-margin-SIZE_PANEL-200,0+450], [width-margin-SIZE_PANEL-200,height-margin-150], [0+margin+200,height-margin-150],[0+margin+200,0+450]])
   

    ea = EvolutionaryAlgorithm(num_robots, NN=NN)
    max_gen = []
    avg_gen = []
    diversity = []

    ## Creating robots
    robots = []
    init_pos = [100,100] # Initial position first robot
    radius = 30 # Radius robot
    for n in range(num_robots):
        robots.append(Robot(init_pos=init_pos,length=radius, brain=NN, limit_speed=10))
        robots[n].brain.set_weights_from_EA(ea.individuals[n])


    #The life of each generation will be larger as the time goes by
    time_limit = 15
    for i in range(num_generations):
        print("\nLaunching generation",i)
        #The life of each generation will be 5 seconds larger each 10 generations
        if i%10==0 and i!=0: time_limit += 5 
        #Save performance and diversity plots each 5 generations
        if i%5==0 and i!=0: 
            #Performance
            plt.plot(range(i), max_gen, label="Max fitness")
            plt.plot(range(i), avg_gen, label="Average fitness")
            plt.legend(loc="upper left")
            plt.xlabel('Generations')
            plt.ylabel('Score')
            plt.savefig('gen_'+str(i)+'_performance.png')
            plt.close()
            #Diversity
            plt.plot(range(i), diversity, label="Diversity")
            plt.legend(loc="upper left")
            plt.xlabel('Generations')
            plt.ylabel('Euclidean distance diversity')
            plt.savefig('gen_'+str(i)+'_diversity.png')
            plt.close()

        ### ENVIRONMENT ###
        env = Environment(points,margin) # Creating the environment
        env.draw_env() # Drawing the environment

        ### ROBOTS ###
        for n in range(num_robots):
            robots[n].draw_robot(False)

        ### MAIN LOOP SIMULATOR ###
        start = time.time()
        while True:
            screen.fill(WHITE) # Background screen
            for event in pygame.event.get():  # Event observer
                if event.type == pygame.QUIT: # Exit
                    pygame.quit()
                    sys.exit(1)
            # Moving robot
            for robot in robots:
                robot.use_brain()

            # Environment drawing
            env.draw_env()
            # Robots drawing & dust cleaning
            for robot in robots:
                robot.use_sensors(env.lines)
                coll_flag = robot.update_pos(env.lines)
                coords_robot = robot.draw_robot(coll_flag)
                env.clean_dust(coords_robot, robot)


            pygame.display.update()
            FPSCLOCK.tick(FPS)

            # each generation run for 15 seconds
            if time.time()-start > time_limit:
                break

        maxim, mean = ea.evaluation(robots) #Evaluating scores individuals with fitnessFunction & some parameters
        
        # Calculating best and average fitness values in the generation
        max_gen.append(maxim)
        avg_gen.append(sum(ea.evaluations)/len(ea.evaluations))
        #Calculating diversity in the generation
        diversity.append(ea.calc_diversity())
        
        ea.selection(num_robots/2)     #Selecting n best ones
        ea.reproduction(crossover='Arithmetic') #Letting them reproduce
        ea.mutation(mutation_range=0.5, mutation_rate=0.05) #Mutating individuals


        # Creating robots of next generation
        # Setting new weights from EA
        del robots #Saving memory
        robots = []
        for n in range(num_robots):
            robots.append(Robot(init_pos=init_pos,length=radius, limit_speed=10))
            robots[n].brain.set_weights_from_EA(ea.individuals[n])

        # Saving weights of NN from individuals at the end of each generation
        ea.save_individuals()


if __name__ == '__main__':
    main(num_room=0, num_robots=6, NN='simple', num_generations=200)

