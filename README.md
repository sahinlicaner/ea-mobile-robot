# Evolutionary Algorithm on Mobile Robot Simulator

It is basically a simulation of genetic algorithm implementation on generations of robots that make use of their own neural net to decide the velocities of their two motors. Robots are equipped with 12 sensors to calculate the distance to surrondings, which are also used as inputs to neural network. The fitness function is based on cleaning the dust as much as possible while avoiding collusion with boundaries.

## Requirements
* Numpy
* Matplotlib
* Pygame
* Shapely
* Keras

## Screenshot from Simulation
![im1](/figs/simulation.png)

## Performance-Diversity Graphs of a Random Generation
![im1](/figs/performance_ex.png)
![im1](/figs/diversity_ex.png)

