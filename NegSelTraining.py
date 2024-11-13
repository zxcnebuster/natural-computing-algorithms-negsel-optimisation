#################################################################################
#### PLEASE READ ALL COMMENTS BELOW AND MAKE SURE YOU FOLLOW MY INSTRUCTIONS ####
#################################################################################

# This is the skeleton program 'NegSelTraining.py' around which you should build your implementation.

# The training set should be in a file 'self_training.txt' (in the same folder as this program).

# The output is a detector set that is in the file 'detector_<timestamp>.txt' where '<timestamp>' is a timestamp
# so that you do not overwrite previously produced detector sets. You can always rename these files.

# In summary, it is assumed that 'NegSelTraining.py' and 'self_training.txt' are in the same folder
# and that the file containing the detector set is written in this folder.

# As regards the four values to be entered below
# - make sure that no comments are inserted after you have entered the values
# - make sure that the first two values appear within double quotes
# - make sure that the type of 'threshold' is int or float
# - make sure that the type of 'num_detectors' is int.

# Ensure that your implementation works for data of *general* dimension n and not just for the
# particular dimension of the given data sets!

##############################
#### ENTER YOUR USER-NAME ####
##############################

username = "ttrq46"

###############################################################
#### ENTER THE CODE FOR THE ALGORITHM YOU ARE IMPLEMENTING ####
###############################################################

alg_code = "RV"

#####################################################################################################################
#### ENTER THE THRESHOLD: IF YOU ARE IMPLEMENTING VDETECTOR THEN SET THE THRESHOLD AS YOUR CHOICE OF SELF-RADIUS ####
#####################################################################################################################

threshold = 0.089370286583804

######################################################
#### ENTER THE INTENDED SIZE OF YOUR DETECTOR SET ####
######################################################

num_detectors = 775

################################################################
#### DO NOT TOUCH ANYTHING BELOW UNTIL I TELL YOU TO DO SO! ####
####      THIS INCLUDES IMPORTING ADDITIONAL MODULES!       ####
################################################################

import time
import os.path
import random
import math
import sys
    
def get_a_timestamp_for_an_output_file():
    local_time = time.asctime(time.localtime(time.time()))
    timestamp = local_time[4:7] + local_time[8:10] + local_time[11:13] + local_time[14:16] + local_time[17:19]
    timestamp = timestamp.replace(" ", "0") 
    return timestamp

def read_points_only(f, point_length, num_points, file):
    list_of_points = []
    count = 0
    error = []
    the_line = f.readline()
    while the_line != "":
        points = the_line.split("[")
        points.pop(0)
        how_many = len(points)
        for i in range(0, how_many):
            if points[i][len(points[i]) - 1] == ",":
                points[i] = points[i][0:len(points[i]) - 2]
            elif points[i][len(points[i]) - 1] == "\n":
                points[i] = points[i][0:len(points[i]) - 3]
            else:
                points[i] = points[i][0:len(points[i]) - 1]
            split_point = points[i].split(",")
            if len(split_point) != point_length:
                error.append("\n*** error: point {0} has the wrong number of components\n".format(i + 1))
                return list_of_points, error
            numeric_point = []
            for j in range(0, point_length):
                numeric_point.append(float(split_point[j]))
            list_of_points.append(numeric_point[:])
            count = count + 1
        the_line = f.readline()
    if count != num_points:
        error.append("\n*** error: there should be {0} points in {1} but there are {2}\n".format(num_points, file, count))
    return list_of_points, error
 
location_of_self = "self_training.txt"

if not os.path.exists(location_of_self):
    print("\n*** error: {0} does not exist\n".format(location_of_self))
    sys.exit()

f = open(location_of_self, "r")

self_or_non_self = f.readline()
if self_or_non_self != "Self\n":
    print("\n*** error: the file " + location_of_self + " is not denoted as a Self-file\n")
    f.close()
    sys.exit()
dim = f.readline()
length_of_dim = len(dim)
dim = dim[len("n = "):length_of_dim - 1]
n = int(dim)
num_points = f.readline()
length_of_num_points = len(num_points)
num_points = num_points[len("number of points = "):length_of_num_points - 1]
Self_num_points = int(num_points)

list_of_points, error = read_points_only(f, n, Self_num_points, location_of_self)
Self = list_of_points[:]

f.close()

if error != []:
    length = len(error)
    for i in range(0, length):
        print(error[i])
    sys.exit()

detectors = []

start_time = time.time()

intended_num_detectors = num_detectors

#########################################################################################
#### YOU SHOULDN'T HAVE TOUCHED *ANYTHING* UP UNTIL NOW APART FROM SUPPLYING VALUES  ####
#### FOR 'username', 'alg_code', 'threshold' and 'num_detectors' AS REQUESTED ABOVE. ####
####                        NOW READ THE FOLLOWING CAREFULLY!                        ####
#########################################################################################

# The training data has now been read with the following reserved variables:
#   - 'n' = the dimension of the points in the training set                 int
#   - 'threshold' = the threshold or self-radius, as appropriate            int or float
#   - 'Self_num_points' = the number of points in the training set
#   - 'Self' = the list of points in the training set.
# These are reserved variables and their names should not be changed.

# You also have the reserved variables
#   - 'user_name', 'alg_code', 'threshold', 'num_detectors', 'intended_num_detectors' and 'start_time'.
# Remember: if 'alg_code' = 'VD' then 'threshold' denotes your chosen self-radius.

# You need to initialize any other parameters (if you are implementing 'Real-valued Negative Selection'
# or 'VDetector') yourself in your code below.

# The list of detectors that your code generates needs to be stored in the variable 'detectors'.
# This is a reserved variable and has just been initialized as empty above. You need to ensure that
# your computed detector set is stored in 'detectors' as a list of points, i.e., as a list of lists-of-
# floats-of-length-'n' for NS and RV and 'n' + 1 for VD (remember: a detector for VD is a point plus
# its individual radius - see Lecture 4).

# FOR ALL OF THE RESERVED VARIABLES BELOW, YOU MUST ENSURE THAT ON TERMINATION THE TYPE
# OF THE RESPECTIVE VARIABLE IS AS SHOWN.

#  - 'n'                int
#  - 'threshold'        int or float

# Finally, if you choose to use numpy then import it below (and don't forget to ensure that 
# variables are of the correct type on termination).

###########################################
#### NOW YOU CAN ENTER YOUR CODE BELOW ####
###########################################


import numpy as np


class RealValuedNegativeSelection:
    def __init__(self, S, r, n, N, max_it, T, movement_decay_0, movement_decay_rate, k):
        """
        Initialise parameters
        :param S: Self set
        :param r: Threshold distance
        :param N: Number of detectors of length n
        :param n: Dimensionality of the data space
        :param max_it: Maximum number of full iterations
        :param T: Maturity age of detectors
        :param movement_decay_0: Initial value of updating rate, i.e. learning rate of the algorithm
        :param movement_decay_rate: Decay rate parameter -  influences the magnitude of decay over time i.e. smaller the value, faster decay, decay becomes small in a short period of time
        :param k: Number of k-nearest neighbors
        """
        self.S = np.array(S)
        self.r = r
        self.n = n
        self.N = N
        self.max_it = max_it
        self.T = T
        self.movement_decay_0 = movement_decay_0
        self.movement_decay_rate = movement_decay_rate
        self.k = k
        self.D = self.rand_gen_detectors()
        # self.D = self.calculate_init_antibody_set(S=np.array(S), n=n)
        self.ages = np.zeros(N)

    def rand_gen_detectors(self):
        """
        Randomly generated detectors of length n and age 0
        """
        return np.random.rand(self.N, self.n)

    def euclidean_dist(self, a, b):
        """
        Find euclidean distance(affinity) between points
        """
        return np.linalg.norm(a - b)

    def median_dist(self, d):
        """
        Median distance to k nearest neighbours of d in 
        """
        distances = np.linalg.norm(self.S - d, axis=1)
        knn_indices = np.argpartition(distances, self.k)[:self.k]
        knn_dist = [(distances[i], self.S[i]) for i in knn_indices]
        return np.median(distances[knn_indices]), knn_dist

    def movement_decay(self, t):
        """
        Find movement decay
        """
        return self.movement_decay_0 * np.exp(-t / self.movement_decay_rate)

    def replace_detector(self, i):
        """
        Кeplace d with randomly generated age-0 detector
        """
        self.D[i] = np.random.rand(self.n)
        self.ages[i] = 0

    def move_detector(self, movement_decay_t, d, i, knn_dist):
        """
        Move d out of Self via d → d + movement
        """
        knn_dist_np = np.array([d - s for dist, s in knn_dist])
        movement = movement_decay_t * np.sum(knn_dist_np, axis=0) / self.k
        # self.D[i] = np.clip(self.D[i] + movement, 0, 1) ### NOT AN ENHANCEMENT
        self.D[i] += movement
        for dim in range(len(self.D[i])):
            if self.D[i][dim] < 0 or self.D[i][dim] > 1:
                self.D[i][dim] = self.D[i][dim] - 2 * (self.D[i][dim] - round(self.D[i][dim]))
        self.ages[i] += 1

    def separate_detector(self, movement_decay_t, d, i):
        """
        If d does not match with S, set the age of d to be 0 and “move d away from” other existing detectors
        """
        D_diff = self.D - d
        not_same = ~np.all(D_diff == 0, axis=1)
        D_diff = D_diff[not_same]
        matching_degrees = np.exp(-np.linalg.norm(D_diff, axis=1)**2 / (2 * self.r**2))
        sum_1 = np.sum(matching_degrees[:, None] * D_diff, axis=0)
        sum_2 = np.sum(matching_degrees)
        movement = movement_decay_t * sum_1 / sum_2
        # self.D[i] = np.clip(self.D[i] + movement, 0, 1) ### NOT AN ENHANCEMENT
        self.D[i] += movement
        for dim in range(len(self.D[i])):
            if self.D[i][dim] < 0 or self.D[i][dim] > 1:
                self.D[i][dim] = self.D[i][dim] - 2 * (self.D[i][dim] - round(self.D[i][dim]))
        self.ages[i] = 0

    def update_detectors(self, t):
        """
        For some d:
            - while it is “close” to S, it is moved via move_detector(self, movement_decay_t, d, i)
            - when it is “away” from S, it is “separated” from D via separate_detector(self, movement_decay_t, d, i)
            - when d is not moving away from S, suppose it is in the center of Self space, it is replaced at age > T
        """
        movement_decay_t = self.movement_decay(t)
        for i, d in enumerate(self.D):
            median, knn_dist = self.median_dist(d)
            if median < self.r:
                if self.ages[i] > self.T:
                    self.replace_detector(i)
                else:
                    self.move_detector(movement_decay_t, d, i, knn_dist)
            else:
                self.separate_detector(movement_decay_t, d, i)
    def find_detectors(self):
        """
        Update detectors for max_it iterations and then output the list of detectors
        :return detectors
        """
        for t in range(0, self.max_it + 1):
            self.update_detectors(t)            
        return self.D

S=Self
threshold=threshold
n=n
N=num_detectors
max_it=65
T=1
movement_decay_0=1e-05
movement_decay_rate=0.5181587951916901
k=3
RNS = RealValuedNegativeSelection(S, threshold, n, N, max_it, T, movement_decay_0, movement_decay_rate, k)
detectors = RNS.find_detectors()





#########################################################
#### YOU SHOULD HAVE NOW FINISHED ENTERING YOUR CODE ####
####     DO NOT TOUCH ANYTHING BELOW THIS COMMENT    ####
#########################################################

# At this point in the execution, you should have computed
# - the list 'detectors' of your detector set.

now_time = time.time()
training_time = round(now_time - start_time, 1)

timestamp = get_a_timestamp_for_an_output_file()
detector_set_location = "detector_" + timestamp + ".txt"

f = open(detector_set_location, "w")

f.write("username = {0}\n".format(username))
f.write("detector set\n")
f.write("algorithm code = {0}\n".format(alg_code))
f.write("dimension = {0}\n".format(n))
if alg_code != "VD":
    f.write("threshold = {0}\n".format(threshold))
else:
    f.write("self-radius = {0}\n".format(threshold))
num_detectors = len(detectors)
f.write("number of detectors = {0} (from an intended number of {1})\n".format(num_detectors, intended_num_detectors))
f.write("training time = {0}\n".format(training_time))
detector_length = n
if alg_code == "VD":
    detector_length = n + 1
for i in range(0, num_detectors):
    f.write("[")
    for j in range(0, detector_length):
        if j != detector_length - 1:
            f.write("{0},".format(detectors[i][j]))
        else:
            f.write("{0}]".format(detectors[i][j]))
            if i == num_detectors - 1:
                f.write("\n")
            else:
                f.write(",\n")
f.close()

print("detector set saved as {0}\n".format(detector_set_location))















    
