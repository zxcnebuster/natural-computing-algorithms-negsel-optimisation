#################################################################################
#### PLEASE READ ALL COMMENTS BELOW AND MAKE SURE YOU FOLLOW MY INSTRUCTIONS ####
#################################################################################

# This is the skeleton program 'NatAlgDiscrete.py' around which you should build your implementation.

# On input 'GCGraphA.txt', say, the output is a witness set that is in the file 'WitnessA_<username>_<timestamp>.txt' where
# '<timestamp>' is a timestamp so that you do not overwrite previously produced witnesses. You can always
# rename these files.

# It is assumed that all graph files are in a folder called 'GraphFiles' that lies in the same folder as
# this program.

# As regards the four values to be entered below
# - make sure that the all four values appear within double quotes
# - make sure that no comments are inserted after you have entered the values.

##############################
#### ENTER YOUR USER-NAME ####
##############################

username = "ttrq46"

###############################################################
#### ENTER THE CODE FOR THE ALGORITHM YOU ARE IMPLEMENTING ####
###############################################################

alg_code = "WO"

#################################################################
#### ENTER THE CODE FOR THE GRAPH PROBLEM YOU ARE OPTIMIZING ####
#################################################################

problem_code = "GC"

#############################################################
#### ENTER THE DIGIT OF THE INPUT GRAPH FILE (A, B OR C) ####
#############################################################

graph_digit = "B"

################################################################
#### DO NOT TOUCH ANYTHING BELOW UNTIL I TELL YOU TO DO SO! ####
####      THIS INCLUDES IMPORTING ADDITIONAL MODULES!       ####
################################################################

import time
import os
import random
import math
import sys

def location_of_GraphFiles(problem_code, graph_digit):
    input_file = os.path.join("GraphFiles", problem_code + "Graph" + graph_digit + ".txt")
    return input_file

def location_of_witness_set(username, graph_digit, timestamp):
    witness_set = "Witness" + graph_digit + "_" + username + "_" + timestamp + ".txt"
    return witness_set

def get_a_timestamp_for_an_output_file():
    local_time = time.asctime(time.localtime(time.time()))
    timestamp = local_time[4:7] + local_time[8:10] + local_time[11:13] + local_time[14:16] + local_time[17:19]
    timestamp = timestamp.replace(" ", "0") 
    return timestamp

def read_the_graph_file(problem_code, graph_digit):
    vertices_tag =          "number of vertices = "
    edges_tag =             "number of edges = "
    colours_to_use_tag =    "number of colours to use = "
    sets_in_partition_tag = "number of partition sets = "
    len_vertices_tag = len(vertices_tag)
    len_edges_tag = len(edges_tag)
    len_colours_to_use_tag = len(colours_to_use_tag)
    len_sets_in_partition_tag = len(sets_in_partition_tag)
    
    input_file = location_of_GraphFiles(problem_code, graph_digit)
    f = open(input_file, 'r')
    whole_line = f.readline()
    vertices = whole_line[len_vertices_tag:len(whole_line) - 1]
    v = int(vertices)
    whole_line = f.readline()
    edges = whole_line[len_edges_tag:len(whole_line) - 1]
    if problem_code == "GC":
        whole_line = f.readline()
        colours_to_use = whole_line[len_colours_to_use_tag:len(whole_line) - 1]
        colours = int(colours_to_use)
    if problem_code == "GP":
        whole_line = f.readline()
        sets_in_partition = whole_line[len_sets_in_partition_tag:len(whole_line) - 1]
        sets_in_partition = int(sets_in_partition)
    matrix = []
    for i in range(0, v - 1):
        whole_line = f.readline()
        if i != v - 2:
            splitline = whole_line.split(',')
            splitline.pop(v - 1 - i)
            for j in range(0, v - 1 - i):
                splitline[j] = int(splitline[j])
        else:
            splitline = whole_line[0:len(whole_line) - 1]
            splitline = [int(splitline)]
        splitline.insert(0, 0)
        matrix.append(splitline[:])            
    matrix.append([0])
    for i in range(0, v):
        for j in range(0, i):
            matrix[i].insert(j, matrix[j][i])
    f.close()

    edges = []
    for i in range(0, v):
        for j in range(i + 1, v):
            if matrix[i][j] == 1:
                edges.append([i, j])

    if problem_code == "GC":
        return v, edges, matrix, colours
    elif problem_code == "GP":
        return v, edges, matrix, sets_in_partition
    else:
        return v, edges, matrix
 
if problem_code == "GC":
    v, edges, matrix, colours = read_the_graph_file(problem_code, graph_digit)
elif problem_code == "GP":
    v, edges, matrix, sets_in_partition = read_the_graph_file(problem_code, graph_digit)
else:
    v, edges, matrix = read_the_graph_file(problem_code, graph_digit)

start_time = time.time()

#########################################################################################
#### YOU SHOULDN'T HAVE TOUCHED *ANYTHING* UP UNTIL NOW APART FROM SUPPLYING VALUES  ####
#### FOR 'username', 'alg_code', 'problem_code' and 'graph_digit' AS REQUESTED ABOVE ####
####                        NOW READ THE FOLLOWING CAREFULLY!                        ####
#########################################################################################

# FOR ALL OF THE RESERVED VARIABLES BELOW, YOU MUST ENSURE THAT ON TERMINATION THE TYPE
# OF THE RESPECTIVE VARIABLE IS AS SHOWN.

# For the problem GC, the graph data has now been read into the following reserved variables:
#   - 'v' = the number of vertices of the graph                                  int
#   - 'edges' = a list of the edges of the graph (just in case you need them)    list
#   - 'matrix' = the full adjacency matrix of the graph                          list
#   - 'colours' = the maximum number of colours to be used when colouring        int

# For the problem CL, the graph data has now been read into the following reserved variables:
#   - 'v' = the number of vertices of the graph                                  int
#   - 'edges' = a list of the edges of the graph (just in case you need them)    list
#   - 'matrix' = the full adjacency matrix of the graph                          list

# For the problem GP, the graph data has now been read into the following reserved variables:
#   - 'v' = the number of vertices of the graph                                  int
#   - 'edges' = a list of the edges of the graph (just in case you need them)    list
#   - 'matrix' = the full adjacency matrix of the graph                          list
#   - 'sets_in_partition' = the number of sets in any partition                  int

# These are reserved variables and need to be treated as such, i.e., use these names for these
# concepts and don't re-use the names.

# For the problem GC, you will produce a colouring in the form of a list of v integers called
# 'colouring' where the entries range from 1 to 'colours'. Note! 0 is disallowed as a colour!
# You will also produce an integer in the variable 'conflicts' which denotes how many edges
# are such that the two incident vertices are identically coloured (of course, your aim is to
# MINIMIZE the value of 'conflicts').

# For the problem CL, you will produce a clique in the form of a list of v integers called
# 'clique' where the entries are either 0 or 1. If 'clique[i]' = 1 then this denotes that the
# vertex i is in the clique with 'clique[i]' = 0 denoting otherwise.
# You will also produce an integer in the variable 'clique_size' which denotes how many vertices
# are in your clique (of course, your aim is to MAXIMIZE the value of 'clique_size').

# For the problem GP, you will produce a partition in the form of a list of v integers called
# 'partition' where the entries are in {1, 2, ..., 'sets_in_partition'}. Note! 0 is not the
# name of a partition set! If 'partition[i]' = j then this denotes that the vertex i is in the
# partition set j.
# You will also produce an integer in the variable 'conflicts' which denotes how many edges are
# incident with vertices in different partition sets (of course, your aim is to MINIMIZE the
# value of 'conflicts').

# In consequence, the following additional variables are reserved:
#   - 'colouring'                       list of int
#   - 'conflicts'                       int
#   - 'clique'                          list of int
#   - 'clique_size'                     int
#   - 'partition'                       list of int

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
# to ensure that they are changed prior to termination.

# INITIALIZE THE ACTUAL PARAMETERS YOU USE FOR YOUR ALGORITHM BELOW. ENSURE THAT YOU INITIALIZE
# *ALL* OF THE PARAMETERS REQUIRED APPROPRIATELY (SEE ABOVE) FOR YOUR CHOSEN ALGORITHM.

# In particular, if you are implementing, say, Artificial Bee Colony and you choose to encode
# your bees as tuples of length v (where v is the number of vertices in the input graph) then
# you must initialize n as v (of course, you might have some other encoding where n is different
# to v and if so then you would initialize n accordingly.)

# Also, you may introduce additional parameters if you wish (below) but they won't get written
# to the output file.

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




n = 425
num_cyc = 175
N = 1450
b = 2




###########################################
#### NOW INCLUDE THE REST OF YOUR CODE ####
###########################################



def update_a(t):
    a = 2 - 2 * (t / num_cyc)
    return a


def update_parameters(a):
    r = np.random.uniform(0, 1)
    A = 2 * a * r - a
    l = np.random.uniform(-1, 1)
    p = np.random.uniform(0, 1)
    return A, l, p


def compute_f(colouring, edges):
    color_vertex1 = colouring[edges[:, 0]]
    color_vertex2 = colouring[edges[:, 1]]
    conflicts = np.sum(color_vertex1 == color_vertex2)
    return conflicts


def encircle(t, best_whale, current_whale,):
    more_exploration = True
    n = len(current_whale)
    change_p = 1 - (t / num_cyc)
    change_mask = np.random.rand(n) < change_p
    new_whale = np.where(change_mask, best_whale, current_whale)

    # UNCOMMENT THE BELOW TO GET THE VANILLA VERSION OF WHALE OPTIMISATION IMPLEMENTATION

                 # more_exploration = False


    # UNCOMMENT THE ABOVE TO GET THE VANILLA VERSION OF WHALE OPTIMISATION IMPLEMENTATION

    if more_exploration:
        exploration_factor = 0.001 
        exploration_mask = np.random.rand(n) < exploration_factor
        random_colors = np.random.randint(1, colours + 1, n)
        new_whale = np.where(exploration_mask, random_colors, new_whale)
    return new_whale


def bubblenet_attack(best_whale, current_whale,l):
    D_star = np.where(best_whale != current_whale)[0]
    x = int((l + 1) / 2 * len(D_star))
    selected_vertices = np.random.choice(D_star, x, replace=False)

    change_mask = np.random.rand(len(selected_vertices)) < 0.5
    vertices_to_recolor = selected_vertices[change_mask]
    new_colors = np.random.randint(1, colours + 1, size=len(vertices_to_recolor))

    new_whale = current_whale.copy()
    new_whale[vertices_to_recolor] = new_colors
    return new_whale


def greedy_coloring(N=None):
    """
    Greedy coloring algorithm for an adjacency matrix.
    :param adjacency_matrix: 2D numpy array representing the graph.
    :return: Array representing the coloring of the graph.
    """
    n = len(matrix)  
    coloring = np.zeros(n, dtype=int)
    for vertex in range(n):
        adjacent_colors = {coloring[i] for i in range(n) if matrix[vertex][i] == 1 and coloring[i] > 0}

        color = 1
        while color in adjacent_colors:
            if color < colours:
                color += 1
            else:
                break
        coloring[vertex] = color

    return coloring


def WhaleOptimisation(n, N, num_cyc):
    greedy_color_init_whales = True
    global edges
    edges = np.array(edges)
    
    # UNCOMMENT THE BELOW TO GET THE VANILLA VERSION OF WHALE OPTIMISATION IMPLEMENTATION

                # greedy_color_init_whales = False

    # UNCOMMENT THE ABOVE TO GET THE VANILLA VERSION OF WHALE OPTIMISATION IMPLEMENTATION
    
    if greedy_color_init_whales:
         whales = np.array([greedy_coloring() for _ in range(N)])
    else:
        whales = np.array([np.random.randint(1, colours + 1, n) for _ in range(N)])
    t = 1

    fitness = np.array([compute_f(coloring, edges) for coloring in whales])
    best_whale = whales[np.argmin(fitness)]

    best_fitness = np.min(fitness)

    while t <= num_cyc: 
        a = update_a(t)
        for i in range(N):
            A,  l, p = update_parameters(a)
            if p < 0.5:
                if np.abs(A) < 1:
                    whales[i] = encircle(t, best_whale, whales[i],)

                else:
                    j = np.random.choice([x for x in range(N) if x != i])
                    whales[i] = encircle(t, whales[j], whales[i])

            else:
                whales[i] = bubblenet_attack(best_whale, whales[i], l)

        fitness = np.array([compute_f(colouring, edges) for colouring in whales])
        best_whale = whales[np.argmin(fitness)]
        t = t + 1
    best_fitness=compute_f(best_whale, edges)
    return best_whale.tolist(), int(best_fitness)

       

colouring, conflicts = WhaleOptimisation(n, N, num_cyc)



















#########################################################
#### YOU SHOULD HAVE NOW FINISHED ENTERING YOUR CODE ####
####     DO NOT TOUCH ANYTHING BELOW THIS COMMENT    ####
#########################################################

# At this point in the execution, you should have computed
# - the list 'colouring' and integer 'conflicts', if you are solving GC;
# - the list 'clique' and the integer 'clique_size', if you are solving CL; or
# - the list 'partition' and the integer 'conflicts', if you are solving GP.

# What follows is error-checking code. Your output file will be saved only if
# no errors are thrown. If there are errors, you'll be told what they are.

# ANY OUTPUT WILL NOT BE SAVED TO FILE UNTIL NO ERRORS ARE THROWN!

now_time = time.time()
elapsed_time = round(now_time - start_time, 1)

error_flag = False

if not problem_code in ["GC", "GP", "CL"]:
    print("*** error: 'problem_code' = {0} is illegal".format(problem_code))
    error_flag = True

if problem_code == "GC":
    if type(conflicts) != int:
        print("*** error: 'conflicts' is not an integer: it is {0} and it has type {1})".format(conflicts, type(conflicts)))
        error_flag = True
    elif conflicts < 0:
        print("*** error: 'conflicts' should be non-negative whereas it is {0}".format(conflicts))
        error_flag = True
    elif type(colouring) != list:
        print("*** error: 'colouring' is not a list (it has type {0})".format(type(colouring)))
        error_flag = True
    elif len(colouring) != v:
        print("*** error: 'colouring' is a list of length {0} whereas it should have length {1}".format(len(colouring), v))
        error_flag = True
    else:
        for i in range(0, v):
            if type(colouring[i]) != int:
                print("*** error: 'colouring[{0}]' = {1} is not an integer (it has type {2})".format(i, colouring[i], type(colouring[i])))
                error_flag = True
                break
            elif colouring[i] < 1 or colouring[i] > colours:
                print("*** error: 'colouring[{0}]' = {1} which is a bad colour (colours must range from 1 up to 'colours' = {2})".format(i, colouring[i], colours))
                error_flag = True
                break
    if error_flag == False:
        true_conflicts = 0
        for i in range(0, v):
            for j in range(i + 1, v):
                if matrix[i][j] == 1 and colouring[i] == colouring[j]:
                    true_conflicts = true_conflicts + 1
        if conflicts != true_conflicts:
            print("*** error: you claim {0} but there are actually {1} conflicts\n".format(conflicts, true_conflicts))
            error_flag = True

if problem_code == "GP":
    if type(conflicts) != int:
        print("*** error: 'conflicts' is not an integer: it is {0} and it has type {1})".format(conflicts, type(conflicts)))
        error_flag = True
    elif conflicts < 0:
        print("*** error: 'conflicts' should be non-negative where it is {0}".format(conflicts))
        error_flag = True
    elif type(partition) != list:
        print("*** error: 'partition' is not a list (it has type {0})".format(type(partition)))
        error_flag = True
    elif len(partition) != v:
        print("*** error: 'partition' is a list of length {0} whereas it should have length {1}".format(len(partition), v))
        error_flag = True
    else:
        for i in range(0, v):
            if type(partition[i]) != int:
                print("*** error: 'partition[{0}]' = {1} is not an integer (it has type {2})".format(i, partition[i], type(partition[i])))
                error_flag = True
                break
            elif partition[i] < 1 or partition[i] > sets_in_partition:
                print("*** error: 'partition[{0}]' = {1} which is a bad partite set (partite set names must range from 1 up to 'sets_in_partition' = {2})".format(i, partition[i], sets_in_partition))
                error_flag = True
                break
    if error_flag == False:
        true_conflicts = 0
        for i in range(0, v):
            for j in range(i + 1, v):
                if matrix[i][j] == 1 and partition[i] != partition[j]:
                    true_conflicts = true_conflicts + 1
        if conflicts != true_conflicts:
            print("*** error: you claim {0} but there are actually {1} conflicts\n".format(conflicts, true_conflicts))
            error_flag = True

if problem_code == "CL":
    if type(clique_size) != int:
        print("*** error: 'clique_size' is not an integer: it is {0} and it has type {1})".format(clique_size, type(clique_size)))
        error_flag = True
    elif clique_size < 0:
        print("*** error: 'clique_size' should be non-negative where it is {0}".format(clique_size))
        error_flag = True
    elif type(clique) != list:
        print("*** error: 'clique' is not a list (it has type {0})".format(type(clique)))
        error_flag = True
    elif len(clique) != v:
        print("*** error: 'clique' is a list of length {0} whereas it should have length {1}".format(len(clique), v))
        error_flag = True
    else:
        for i in range(0, v):
            if type(clique[i]) != int:
                print("*** error: 'clique[{0}]' = {1} is not an integer (it has type {2})".format(i, clique[i], type(clique[i])))
                error_flag = True
                break
            elif clique[i] < 0 or clique[i] > 1:
                print("*** error: 'clique[{0}]' = {1} where as all entries should be 0 or 1".format(i, colouring[i]))
                error_flag = True
                break
    if error_flag == False:
        true_size = 0
        for i in range(0, v):
            if clique[i] == 1:
                true_size = true_size + 1
        if clique_size != true_size:
            print("*** error: you claim a clique of size {0} but the list contains {1} ones\n".format(clique_size, true_size))
            error_flag = True
        else:
            bad_edges = 0
            bad_clique = False
            for i in range(0, v):
                for j in range(i + 1, v):
                    if clique[i] == 1 and clique[j] == 1 and matrix[i][j] != 1:
                        bad_edges = bad_edges + 1
                        bad_clique = True
            if bad_clique == True:
                print("*** error: the clique of claimed size {0} is not a clique as there are {1} missing edges\n".format(clique_size, bad_edges))
                error_flag = True

if not alg_code in ["AB", "FF", "CS", "WO", "BA"]:
    print("*** error: 'alg_code' = {0} is invalid".format(alg_code))
    error_flag = True

if type(n) != int:
    print("*** error: 'n' is not an integer: it is {0} and it has type {1})".format(n, type(n)))
if type(num_cyc) != int:
    print("*** error: 'num_cyc' is not an integer: it is {0} and it has type {1})".format(num_cyc, type(num_cyc)))

if alg_code == "AB":
    if type(N) != int:
        print("*** error: 'N' is not an integer: it is {0} and it has type {1})".format(N, type(N)))
        error_flag = True
    if type(M) != int:
        print("*** error: 'M' is not an integer: it is {0} and it has type {1})".format(M, type(M)))
        error_flag = True
    if type(lambbda) != int and type(lambbda) != float:
        print("*** error: 'lambbda' is not an integer or a float: it is {0} and it has type {1})".format(lambbda, type(lambbda)))
        error_flag = True

if alg_code == "FF":
    if type(N) != int:
        print("*** error: 'N' is not an integer: it is {0} and it has type {1})".format(N, type(N)))
        error_flag = True
    if type(lambbda) != int and type(lambbda) != float:
        print("*** error: 'lambbda' is not an integer or a float: it is {0} and it has type {1})".format(lambbda, type(lambbda)))
        error_flag = True
    if type(alpha) != int and type(alpha) != float:
        print("*** error: 'alpha' is not an integer or a float: it is {0} and it has type {1})".format(alpha, type(alpha)))
        error_flag = True

if alg_code == "CS":
    if  type(N) != int:
        print("*** error: 'N' is not an integer: it is {0} and it has type {1})".format(N, type(N)))
        error_flag = True
    if type(p) != int and type(p) != float:
        print("*** error: 'p' is not an integer or a float: it is {0} and it has type {1})".format(p, type(p)))
        error_flag = True
    if type(q) != int and type(q) != float:
        print("*** error: 'q' is not an integer or a float: it is {0} and it has type {1})".format(q, type(q)))
        error_flag = True
    if type(alpha) != int and type(alpha) != float:
        print("*** error: 'alpha' is not an integer or a float: it is {0} and it has type {1})".format(alpha, type(alpha)))
        error_flag = True
    if type(beta) != int and type(beta) != float:
        print("*** error: 'beta' is not an integer or a float: it is {0} and it has type {1})".format(beta, type(beta)))
        error_flag = True

if alg_code == "WO":
    if type(N) != int:
        print("*** error: 'N' is not an integer: it is {0} and it has type {1})".format(N, type(N)))
        error_flag = True
    if type(b) != int and type(b) != float:
        print("*** error: 'b' is not an integer or a float: it is {0} and it has type {1})".format(b, type(b)))
        error_flag = True

if alg_code == "BA":
    if type(sigma) != int and type(sigma) != float:
        print("*** error: 'sigma' is not an integer or a float: it is {0} and it has type {1})".format(sigma, type(sigma)))
        error_flag = True
    if type(f_min) != int and type(f_min) != float:
        print("*** error: 'f_min' is not an integer or a float: it is {0} and it has type {1})".format(f_min, type(f_min)))
        error_flag = True
    if type(f_max) != int and type(f_max) != float:
        print("*** error: 'f_max' is not an integer or a float: it is {0} and it has type {1})".format(f_max, type(f_max)))
        error_flag = True

if error_flag == False:
    
    timestamp = get_a_timestamp_for_an_output_file()
    witness_set = location_of_witness_set(username, graph_digit, timestamp)

    f = open(witness_set, "w")

    f.write("username = {0}\n".format(username))
    f.write("problem code = {0}\n".format(problem_code))
    f.write("graph = {0}Graph{1}.txt with (|V|,|E|) = ({2},{3})\n".format(problem_code, graph_digit, v, len(edges)))
    if problem_code == "GC":
        f.write("colours to use = {0}\n".format(colours))
    elif problem_code == "GP":
        f.write("number of partition sets = {0}\n".format(sets_in_partition))
    f.write("algorithm code = {0}\n".format(alg_code))
    if alg_code == "AB":
        f.write("associated parameters [n, num_cyc, N, M, lambbda] = ")
        f.write("[{0}, {1}, {2}, {3}, {4}]\n".format(n,num_cyc, N, M, lambbda))
    elif alg_code == "FF":
        f.write("associated parameters [n, num_cyc, N, lambbda, alpha] = ")
        f.write("[{0}, {1}, {2}, {3}, {4}]\n".format(n, num_cyc, N, lambbda, alpha))
    elif alg_code == "CS":
        f.write("associated parameters [n, num_cyc, N, p, q, alpha, beta] = ")
        f.write("[{0}, {1}, {2}, {3}, {4}, {5}, {6}]\n".format(n, num_cyc, N, p, q, alpha, beta))
    elif alg_code == "WO":
        f.write("associated parameters [n, num_cyc, N, b] = ")
        f.write("[{0}, {1}, {2}, {3}]\n".format(n, num_cyc, N, b))
    elif alg_code == "BA":
        f.write("associated parameters [n, num_cyc, sigma, f_max, f_min] = ")
        f.write("[{0}, {1}, {2}, {3}, {4}]\n".format(n, num_cyc, sigma, f_max, f_min))
    if problem_code == "GC" or problem_code == "GP":
        f.write("conflicts = {0}\n".format(conflicts))
    elif problem_code == "CL":
        f.write("clique size = {0}\n".format(clique_size))
    f.write("elapsed time = {0}\n".format(elapsed_time))
    
    len_username = len(username)
    user_number = 0
    for i in range(0, len_username):
        user_number = user_number + ord(username[i])
    alg_number = ord(alg_code[0]) + ord(alg_code[1])
    if problem_code == "GC":
        diff = abs(colouring[0] - colouring[v - 1])
        for i in range(0, v - 1):
            diff = diff + abs(colouring[i + 1] - colouring[i])
    elif problem_code == "GP":
        diff = abs(partition[0] - partition[v - 1])
        for i in range(0, v - 1):
            diff = diff + abs(partition[i + 1] - partition[i])       
    elif problem_code == "CL":
        diff = abs(clique[0] - clique[v - 1])
        for i in range(0, v - 1):
            diff = diff + abs(clique[i + 1] - clique[i])
    certificate = user_number + alg_number + diff
    f.write("certificate = {0}\n".format(certificate))

    if problem_code == "GC":
        for i in range(0, v):
            f.write("{0},".format(colouring[i]))
            if (i + 1) % 40 == 0:
                f.write("\n")
        if v % 40 != 0:
            f.write("\n")
    if problem_code == "GP":
        for i in range(0, v):
            f.write("{0},".format(partition[i]))
            if (i + 1) % 40 == 0:
                f.write("\n")
        if v % 40 != 0:
            f.write("\n")
    if problem_code == "CL":
        for i in range(0, v):
            f.write("{0},".format(clique[i]))
            if (i + 1) % 40 == 0:
                f.write("\n")
        if v % 40 != 0:
            f.write("\n")

    f.close()
        
    print("witness file 'Witness{0}_{1}.txt' saved for student {2}".format(graph_digit, timestamp, username))

else:

    print("\n*** ERRORS: the witness file has not been saved - fix your errors first!")


















    
