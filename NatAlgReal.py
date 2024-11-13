#################################################################################
#### PLEASE READ ALL COMMENTS BELOW AND MAKE SURE YOU FOLLOW MY INSTRUCTIONS ####
#################################################################################

# This is the skeleton program 'NatAlgReal.py' around which you should build your implementation.
# Please read through this program and follow the instructions given.

# There are no input or output files, with the results printed to the standard output.

# As regards the two values to be entered below
# - make sure that the first two values appear within double quotes
# - make sure that no comments are inserted after you have entered the values.

# Ensure that your implementation works for *arbitrary* hard-coded functions of arbitrary
# dimension and arbitray min- and max-ranges!

##############################
#### ENTER YOUR USER-NAME ####
##############################

username = "ttrq46"

###############################################################
#### ENTER THE CODE FOR THE ALGORITHM YOU ARE IMPLEMENTING ####
###############################################################

alg_code = "WO"

################################################################
#### DO NOT TOUCH ANYTHING BELOW UNTIL I TELL YOU TO DO SO! ####
####      THIS INCLUDES IMPORTING ADDITIONAL MODULES!       ####
################################################################

import time
import random
import math
import sys
import os
import datetime

def compute_f(point):
    f = 40 + (point[0]**2 - 10*math.cos(2*math.pi*point[0])) + \
        (point[1]**2 - 10*math.cos(2*math.pi*point[1])) + (point[2]**2 - 10*math.cos(2*math.pi*point[2])) + \
        (point[3]**2 - 10*math.cos(2*math.pi*point[3]))
    return f

n = 4

min_range = [-5.12, -5.12, -5.12, -5.12]
max_range = [5.12, 5.12, 5.12, 5.12]

start_time = time.time()

#########################################################################################
#### YOU SHOULDN'T HAVE TOUCHED *ANYTHING* UP UNTIL NOW APART FROM SUPPLYING VALUES  ####
####                 FOR 'username' and 'alg_code' AS REQUESTED ABOVE.               ####
####                        NOW READ THE FOLLOWING CAREFULLY!                        ####
#########################################################################################

# The function 'f' is 'n'-dimensional and you are attempting to MINIMIZE it.
# To compute the value of 'f' at some point 'point', where 'point' is a list of 'n' integers or floats,
# call the function 'compute_f(point)'.
# The ranges for the values of the components of 'point' are given above. The lists 'min_range' and
# 'max_range' above hold the minimum and maximum values for each component and you should use these
# list variables in your code.

# On termination your algorithm should be such that:
#   - the reserved variable 'min_f' holds the minimum value that you have computed for the
#     function 'f' 
#   - the reserved variable 'minimum' is a list of 'n' entries (integer or float) holding the point at which
#     your value of 'min_f' is attained.

# Note that the variables 'username', 'alg_code', 'f', 'point', 'min_f', 'n', 'min_range', 'max_range' and
# 'minimum' are all reserved.

# FOR THE RESERVED VARIABLES BELOW, YOU MUST ENSURE THAT ON TERMINATION THE TYPE
# OF THE RESPECTIVE VARIABLE IS AS SHOWN.

#  - 'min_f'                int or float
#  - 'minimum'              list of int or float

# You should ensure that your code works on any function hard-coded as above, using the
# same reserved variables and possibly of a dimension different to that given above. I will
# run your code with a different such function/dimension to check that this is the case.

# The various algorithms all have additional parameters (see the lectures). These parameters
# are detailed below and are referred to using the following reserved variables.
#
# AB (Artificial Bee Colony)
#   - 'n' = dimension of the optimization problem       int
#   - 'num_cyc' = number of cycles to iterate           int
#   - 'N' = number of employed bees / food sources      int
#   - 'M' = number of onlooker bees                     int
#   - 'lambbda' = limit threshold                       float or int
#
# FF (Firefly)
#   - 'n' = dimension of the optimization problem       int
#   - 'num_cyc' = number of cycles to iterate           int
#   - 'N' = number of fireflies                         int
#   - 'lambbda' = light absorption coefficient          float or int
#   - 'alpha' = scaling parameter                       float or int
#
# CS (Cuckoo Search)
#   - 'n' = dimension of optimization problem           int
#   - 'num_cyc' = number of cycles to iterate           int
#   - 'N' = number of nests                             int
#   - 'p' = fraction of local flights to undertake      float or int
#   - 'q' = fraction of nests to abandon                float or int
#   - 'alpha' = scaling factor for Levy flights         float or int
#   - 'beta' = parameter for Mantegna's algorithm       float or int
#
# WO (Whale Optimization)
#   - 'n' = dimension of optimization problem           int
#   - 'num_cyc' = number of cycles to iterate           int
#   - 'N' = number of whales                            int
#   - 'b' = spiral constant                             float or int
#
# BA (Bat)
#   - 'n' = dimension of optimization problem           int
#   - 'num_cyc' = number of cycles to iterate           int
#   - 'N' = number of bats                              int
#   - 'sigma' = scaling factor                          float or int
#   - 'f_min' = minimum frequency                       float or int
#   - 'f_max' = maximum frequency                       float or int

# These are reserved variables and need to be treated as such, i.e., use these names for these
# parameters and don't re-use the names. Don't forget to ensure that on termination all the above
# variables have the stated type. In particular, if you use specific numpy types then you'll need
# to ensure that they are changed prior to termination (this is checked).

# INITIALIZE THE ACTUAL PARAMETERS YOU USE FOR YOUR ALGORITHM BELOW. ENSURE THAT YOU INITIALIZE
# *ALL* OF THE PARAMETERS REQUIRED APPROPRIATELY (SEE ABOVE) FOR YOUR CHOSEN ALGORITHM.

# In summary, before you input the bulk of your code, ensure that you:
# - import any (legal) modules you wish to use in the space provided below 
# - initialize your parameters in the space provided below
# - ensure that reserved variables have the correct type on termination.

###########################################
#### NOW YOU CAN ENTER YOUR CODE BELOW ####
###########################################
####################################################
#### FIRST IMPORT ANY MODULES IMMEDIATELY BELOW ####
####################################################



import numpy as np


##########################################################
#### NOW INITIALIZE YOUR PARAMETERS IMMEDIATELY BELOW ####
##########################################################

n = 4
num_cyc = 500
N = 50
b = 2


###########################################
#### NOW INCLUDE THE REST OF YOUR CODE ####
###########################################
# ENHANCED VERSION
# TO ACCESS VANILLA VERSION UNCOMMENT HIGHLIGHTED SECTION IN THE MAIN LOOP OF THE ALGORITHM

class WhaleOptimization:
    def __init__(self, n=n, N=N, num_cyc=num_cyc, b=b, min_range=min_range, max_range=max_range, compute_f=compute_f):
        """
        Initialize the Whale Optimization algorithm parameters.
        :param n: Dimension of the optimization problem.
        :param N: Number of whales.
        :param num_cyc: Number of cycles to iterate.
        :param b: Spiral constant.
        :param min_range: Minimum range of the search space.
        :param max_range: Maximum range of the search space.
        :param compute_f: The function to compute the minimum for.
        """
        self.n = n
        self.N = N
        self.num_cyc = num_cyc
        self.b = b
        self.min_range = min_range
        self.max_range = max_range
        self.compute_f = compute_f
        self.whales = np.random.uniform(low=min_range, high=max_range, size=(N, n)) # randomly generate N whales {Xi: i = 1, 2, .., N} and set t = 1
        self.fitness = np.array([compute_f(whale) for whale in self.whales]) # calculate the fitness of each X_i (whale) and store the fittest as X^* (best_whale)
        self.best_whale = self.whales[np.argmin(self.fitness)] # store the fittest X_i as X^* 

    def euclidean_dist(self, a, b):
        """
        Find euclidean distance(affinity) between points
        """
        # return math.sqrt(sum([(x - y) ** 2 for x, y in zip(a, b)]))
        # use numpy to optimise using vectorisation
        return np.linalg.norm(a - b)

    def update_a(self, t):
        """
        Update the algorithm 'a' parameter for each to update the encircling behaviour to shrink.
        :param t: Current iteration number.
        :return a: Influences the encircling behaviour, decreases linearly from 2 to 0 over  cycle iterations - controls the exploration and exploitation behavior.
        """
        a = 2 - 2 * (t / self.num_cyc)
        return a

    def update_parameters(self, t, a):
        """
        Update the algorithm parameters for each cycle based on the encircling value.
        :param t: Current iteration number.
        :param a: Influences the encircling behaviour, decreases linearly from 2 to 0 over cycle iterations - controls the exploration and exploitation behavior.
        :return A: Vector (a function of 'a' and 'r') that controls the intensity of exploration or exploitation - a high absolute value of 'A' (>1) encourages exploration (global search), while a low absolute value (<1)  facilitates exploitation (local search).
        :return C: Random vector [0, 2) for each whale to influence the random distance of the whale from the prey (or best solution so far) during exploration and exploitation.
        :return l: Random number between [-1, 1] for the spiral-shaped path towards the prey during the exploitation phase.
        :return p: Random probability [0, 1) for each whale to decides whether the whale should follow the  shrinking encircling mechanism (p<0.5) or the spiral-shaped path (p>=0.5) for bubble-net attacking
        """
        r = np.random.uniform(0, 1, (self.N,))  # 'r' is a random vector [0, 1) for each whale
        A = 2 * a * r - a
        C = 2 * r
        l = np.random.uniform(-1, 1, (self.N,))
        p = np.random.uniform(0, 1, (self.N,))
        return A, C, l, p

    def encircle(self, i, A, C):
        """
        Encircle behavior of the whale.
        Simulates the whale encircling the prey during hunting, represents exploitation.
        :param i: Index of the current whale.
        :param A: Coefficient A for whale i, controlling the intensity of encircling.
        :param C: Coefficient C for whale i, influencing the random distance from the prey.
        """
        d = np.abs(C[i] * self.best_whale - self.whales[i])
        self.whales[i] = self.best_whale - A[i] * d

    def explore(self, i, A, C):
        """
        Exploration behavior of the whale.
        Mimics the random search for prey, representing global exploration.
        :param i: Index of the current whale.
        :param A: Coefficient A for whale i, controlling the intensity of exploration.
        :param C: Coefficient C for whale i, influencing the random distance during exploration.
        """
        j = np.random.choice([x for x in range(self.N) if x != i])
        d = np.abs(C[i] * self.whales[j] - self.whales[i])
        self.whales[i] = self.whales[j] - A[i] * d

    def bubblenet_attack(self,i,l):
        """
        Bubble-net attacking behavior of the whale. 
        Simulates the unique hunting method of humpback whales using a spiral-shaped path.
        :param i: Index of the current whale.
        :param l: Spiral coefficient for whale i, determining the shape of the spiral movement.
        """
        d_star = np.abs(self.best_whale - self.whales[i])
        self.whales[i] = d_star * np.exp(self.b * l[i]) * np.cos(2 * np.pi * l[i]) + self.best_whale

    def update_whales_vanilla(self, t, i, a, A, C, l, p):
        """
        Update the position of each whale.
        Decides the movement strategy: encircle, explore, bubble-net attack - based on probability.
        :param t: Current cycle number.
        :param i: Index of the current whale.
        :param a: Coefficient a influencing the encircling behavior.
        :param A: Coefficient A for whale i, controlling the intensity of exploration or exploitation.
        :param C: Coefficient C for whale i, influencing the random distance during exploration or exploitation.
        :param l: Spiral coefficient for whale i, determining the shape of the spiral movement.
        :param p: Probability for deciding the movement strategy.
        """
        if p[i] < 0.5:
            if np.abs(A[i]) < 1:
                self.encircle(i, A, C)
            else:
                self.explore(i, A, C)
        else:
            self.bubblenet_attack(i, l)

    def modified_update_whales_exploration(self, i, A, C, p):
        """
        Update the position of each whale.
        Decides the movement strategy: encircle, explore, bubble-net attack - based on probability.
        :param i: Index of the current whale.
        :param A: Coefficient A for whale i, controlling the intensity of exploration or exploitation.
        :param C: Coefficient C for whale i, influencing the random distance during exploration or exploitation.
        :param p: Probability for deciding the movement strategy.
        """
        def update_solution(present_solution, random_solution):
            d = np.abs(C[i] * random_solution - present_solution)
            updated_solution = random_solution - A[i] * d
            if self.compute_f(updated_solution) < self.compute_f(present_solution):
                self.whales[i] = updated_solution
            else:
                pass 
        present_solution = self.whales[i] 
        random_solution_1 = self.whales[np.random.randint(0, len(self.whales))]
        random_solution_2 = self.whales[np.random.randint(0, len(self.whales))]
        if p[i] < 0.5:
            if self.euclidean_dist(random_solution_1, present_solution) < self.euclidean_dist(random_solution_2, present_solution):
                update_solution(present_solution, random_solution_1)
            elif self.euclidean_dist(random_solution_1, present_solution) > self.euclidean_dist(random_solution_2, present_solution):
                update_solution(present_solution, random_solution_2)
            else: #adds randomness in the cases when solutions are equally distant from the present solution - not covered by the paper referenced
                x = random.uniform(0, 1)
                if x <= 0.5:
                    update_solution(present_solution, random_solution_1)
                else:
                    update_solution(present_solution, random_solution_2)
            
        else:
            mean_solution = np.mean([random_solution_1, random_solution_2], axis=0)
            update_solution(present_solution, mean_solution)

    def modified_update_whales_exploitation(self, i, A, C, l, p):
        """
        Update the position of each whale.
        Decides the movement strategy: encircle, explore, bubble-net attack - based on probability.
        :param t: Current cycle number.
        :param i: Index of the current whale.
        :param a: Coefficient a influencing the encircling behavior.
        :param A: Coefficient A for whale i, controlling the intensity of exploration or exploitation.
        :param C: Coefficient C for whale i, influencing the random distance during exploration or exploitation.
        :param l: Spiral coefficient for whale i, determining the shape of the spiral movement.
        :param p: Probability for deciding the movement strategy.
        """
        def update_solution_encircling(present_solution):
            d = np.abs(C[i] * self.best_whale - present_solution)
            updated_solution = self.best_whale - A[i] * d
            if self.compute_f(updated_solution) < self.compute_f(present_solution):
                self.whales[i] = updated_solution
            else:
                pass 
        def update_solution_attacking(present_solution):
            d_star = np.abs(self.best_whale - present_solution)
            updated_solution = d_star * np.exp(self.b * l[i]) * np.cos(2 * np.pi * l[i]) + self.best_whale
            if self.compute_f(updated_solution) < self.compute_f(present_solution):
                self.whales[i] = updated_solution
            else:
                pass 
        # group of solutions - 1/10 of the all solutions. selected at random?
        solutions_indexes = random.sample(range(len(self.whales)), 10)
        solutions = []
        for j in solutions_indexes:
            solutions.append(self.whales[j])
        solutions = np.array(solutions)
        if p[i] > 0.5:
            for solution in solutions:
                update_solution_encircling(solution)
        else:
            for solution in solutions:
                update_solution_attacking(solution)

    def find_minimum(self):
        """
        Find the minimum of the compute_f function
         :return: A tuple containing the position of the best whale (minimum) and its corresponding fitness value.
        """
        t = 1
        while t <= self.num_cyc:
            a = self.update_a(t)
            for i in range(self.N):
                A, C, l, p = self.update_parameters(t, a)

                # UNCOMMENT THE BELOW TO GET THE VANILLA VERSION OF WHALE OPTIMISATION IMPLEMENTATION

                # self.update_whales_vanilla(t, i, a, A, C, l, p)
                # continue

                # UNCOMMENT THE ABOVE TO GET THE VANILLA VERSION OF WHALE OPTIMISATION IMPLEMENTATION
                
                if t <= self.num_cyc /2:
                    self.modified_update_whales_exploration(i, A, C, p)
                else:
                    self.modified_update_whales_exploitation(i, A, C, l, p)
            self.fitness = np.array([self.compute_f(whale) for whale in self.whales])
            self.best_whale = self.whales[np.argmin(self.fitness)]
            t += 1

        return list(self.best_whale), float(compute_f(self.best_whale))

WO = WhaleOptimization()
minimum, min_f = WO.find_minimum()






























    

#########################################################
#### YOU SHOULD HAVE NOW FINISHED ENTERING YOUR CODE ####
####     DO NOT TOUCH ANYTHING BELOW THIS COMMENT    ####
#########################################################

# At this point in the execution, you should have computed your minimum value for the function 'f' in the
# variable 'min_f' and the variable 'minimum' should hold a list containing the values of the point 'point'
# for which function 'f(point)' attains your minimum.

now_time = time.time()
elapsed_time = round(now_time - start_time, 1)

error = []

try:
    n
    try:
        y = n
    except:
        error.append("*** error: 'n' has not been initialized")
        n = -1
except:
    error.append("*** error: the variable 'n' does not exist\n")
    n = -1
try:
    num_cyc
    try:
        y = num_cyc
    except:
        error.append("*** error: 'num_cyc' has not been initialized")
        num_cyc = -1
except:
    error.append("*** error: the variable 'num_cyc' does not exist")
    num_cyc = -1

if alg_code == "AB":
    try:
        N
        try:
           y = N
        except:
            error.append("*** error: 'N' has not been initialized")
            N = -1
    except:
        error.append("*** error: the variable 'N' does not exist")
        N = -1
    try:
        M
        try:
           y = M
        except:
            error.append("*** error: 'M' has not been initialized")
            M = -1
    except:
        error.append("*** error: the variable 'M' does not exist")
        M = -1
    try:
        lambbda
        try:
           y = lambbda
        except:
            error.append("*** error: 'lambbda' has not been initialized")
            lambbda = -1
    except:
        error.append("*** error: the variable 'lambbda' does not exist")
        lambbda = -1
if alg_code == "FF":
    try:
        N
        try:
           y = N
        except:
            error.append("*** error: 'N' has not been initialized")
            N = -1
    except:
        error.append("*** error: the variable 'N' does not exist")
        N = -1
    try:
        alpha
        try:
           y = alpha
        except:
            error.append("*** error: 'alpha' has not been initialized")
            alpha = -1
    except:
        error.append("*** error: the variable 'alpha' does not exist")
        alpha = -1
    try:
        lambbda
        try:
           y = lambbda
        except:
            error.append("*** error: 'lambbda' has not been initialized")
            lambbda = -1
    except:
        error.append("*** error: the variable 'lambbda' does not exist")
        lambbda = -1
if alg_code == "CS":
    try:
        N
        try:
           y = N
        except:
            error.append("*** error: 'N' has not been initialized")
            N = -1
    except:
        error.append("*** error: the variable 'N' does not exist")
        N = -1
    try:
        p
        try:
           y = p
        except:
            error.append("*** error: 'p' has not been initialized")
            p = -1
    except:
        error.append("*** error: the variable 'p' does not exist")
        p = -1
    try:
        q
        try:
           y = q
        except:
            error.append("*** error: 'q' has not been initialized")
            q = -1
    except:
        error.append("*** error: the variable 'q' does not exist")
        q = -1
    try:
        alpha
        try:
           y = alpha
        except:
            error.append("*** error: 'alpha' has not been initialized")
            alpha = -1
    except:
        error.append("*** error: the variable 'alpha' does not exist")
        alpha = -1
    try:
        beta
        try:
           y = beta
        except:
            error.append("*** error: 'beta' has not been initialized")
            beta = -1
    except:
        error.append("*** error: the variable 'beta' does not exist")
        beta = -1
if alg_code == "WO":
    try:
        N
        try:
           y = N
        except:
            error.append("*** error: 'N' has not been initialized")
            N = -1
    except:
        error.append("*** error: the variable 'N' does not exist")
        N = -1
    try:
        b
        try:
           y = b
        except:
            error.append("*** error: 'b' has not been initialized")
            b = -1
    except:
        error.append("*** error: the variable 'b' does not exist")
        b = -1
if alg_code == "BA":
    try:
        sigma
        try:
           y = sigma
        except:
            error.append("*** error: 'sigma' has not been initialized")
            sigma = -1
    except:
        error.append("*** error: the variable 'sigma' does not exist")
        sigma = -1
    try:
        f_max
        try:
           y = f_max
        except:
            error.append("*** error: the variable 'f_max' has not been initialized")
            f_max = -1
    except:
        error.append("*** error: the variable 'f_max' does not exist")
        f_max = -1
    try:
        f_min
        try:
           y = f_min
        except:
            error.append("*** error: 'f_min' has not been initialized")
            f_min = -1
    except:
        error.append("*** error: the variable 'f_min' does not exist")
        f_min = -1

if type(n) != int:
    error.append("*** error: 'n' is not an integer: it is {0} and it has type {1}".format(n, type(n)))
if type(num_cyc) != int:
    error.append("*** error: 'num_cyc' is not an integer: it is {0} and it has type {1}".format(num_cyc, type(num_cyc)))

if alg_code == "AB":
    if type(N) != int:
        error.append("*** error: 'N' is not an integer: it is {0} and it has type {1}".format(N, type(N)))
    if type(M) != int:
        error.append("*** error: 'M' is not an integer: it is {0} and it has type {1}".format(M, type(M)))
    if type(lambbda) != int and type(lambbda) != float:
        error.append("*** error: 'lambbda' is not an integer or a float: it is {0} and it has type {1}".format(lambbda, type(lambbda)))

if alg_code == "FF":
    if type(N) != int:
        error.append("*** error: 'N' is not an integer: it is {0} and it has type {1}".format(N, type(N)))
    if type(lambbda) != int and type(lambbda) != float:
        error.append("*** error: 'lambbda' is not an integer or a float: it is {0} and it has type {1}".format(lambbda, type(lambbda)))
    if type(alpha) != int and type(alpha) != float:
        error.append("*** error: 'alpha' is not an integer or a float: it is {0} and it has type {1}".format(alpha, type(alpha)))

if alg_code == "CS":
    if type(N) != int:
        error.append("*** error: 'N' is not an integer: it is {0} and it has type {1}".format(N, type(N)))
    if type(p) != int and type(p) != float:
        error.append("*** error: 'p' is not an integer or a float: it is {0} and it has type {1}".format(p, type(p)))
    if type(q) != int and type(q) != float:
        error.append("*** error: 'q' is not an integer or a float: it is {0} and it has type {1}".format(q, type(q)))
    if type(alpha) != int and type(alpha) != float:
        error.append("*** error: 'alpha' is not an integer or a float: it is {0} and it has type {1}".format(alpha, type(alpha)))
    if type(beta) != int and type(beta) != float:
        error.append("*** error: 'beta' is not an integer or a float: it is {0} and it has type {1}".format(beta, type(beta)))

if alg_code == "WO":
    if type(N) != int:
        error.append("*** error: 'N' is not an integer: it is {0} and it has type {1}\n".format(N, type(N)))
    if type(b) != int and type(b) != float:
        error.append("*** error: 'b' is not an integer or a float: it is {0} and it has type {1}".format(b, type(b)))

if alg_code == "BA":
    if type(sigma) != int and type(sigma) != float:
        error.append("*** error: 'sigma' is not an integer or a float: it is {0} and it has type {1}".format(sigma, type(sigma)))
    if type(f_min) != int and type(f_min) != float:
        error.append("*** error: 'f_min' is not an integer or a float: it is {0} and it has type {1}".format(f_min, type(f_min)))
    if type(f_max) != int and type(f_max) != float:
        error.append("*** error: 'f_max' is not an integer or a float: it is {0} and it has type {1}".format(f_max, type(f_max)))

if type(min_f) != int and type(min_f) != float:
    error.append("*** error: there is no real-valued variable 'min_f'")
if type(minimum) != list:
    error.append("*** error: there is no tuple 'minimum' giving the minimum point")
elif type(n) == int and len(minimum) != n:
    error.append("*** error: there is no {0}-tuple 'minimum' giving the minimum point; you have a {1}-tuple".format(n, len(minimum)))
elif type(n) == int:
    for i in range(0, n):
        if not "int" in str(type(minimum[i])) and not "float" in str(type(minimum[i])):
            error.append("*** error: the value for component {0} (ranging from 1 to {1}) in the minimum point is not numeric\n".format(i + 1, n))

if error != []:
    print("\n*** ERRORS: there were errors in your execution:")
    length = len(error)
    for i in range(0, length):
        print(error[i])
    print("\n Fix these errors and run your code again.\n")
else:
    print("\nYou have found a minimum value of {0} and a minimum point of {1}.".format(min_f, minimum))
    print("Your elapsed time was {0} seconds.\n".format(elapsed_time))
    
