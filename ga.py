import random
import csv
import copy
import sys
import numpy as np
import random

#from NB import Naive_Bayes
#from NB_int import *
#from sklearn.naive_bayes import GaussianNB

class GA_model(object):
    def __init__(self, population,  cross_over_rate, mutation_rate, no_of_parameters, fitness_callback):
        self.cross_over_rate = cross_over_rate
        self.mutation_rate = mutation_rate
        self.no_of_parameters = no_of_parameters
        self.fitness_callback = fitness_callback
        self.population = population
        self.pop_size = len(population)
        self.chromo_size = len(population[0])
        self.fitness = [0] * self.pop_size

    def fitness_fun(self, input, utility_matrix):
        accuracy=[]
        for i in range(1,11):
            accuracy.append(self.fitness_callback(i,944,utility_matrix,input))
        return max(accuracy)

    def compute_fitness(self, utility_matrix):
        for i in range(self.pop_size):
            self.fitness[i] = self.fitness_fun(self.population[i], utility_matrix)


        if max(self.fitness) == 1:
            print "Done, features are - ", self.population[self.fitness.index(min(self.fitness))]
            sys.exit()


    def roulette_selection(self):
        probability = []
        total_fitness = sum(self.fitness)

        # Individual probability
        for chromo_fit in self.fitness:
            probability.append(float(chromo_fit / total_fitness))

        # Cumulative Probability
        for i in range(1, self.pop_size):
            probability[i] += probability[i-1]

        updated_population = copy.deepcopy(self.population)

        for i in range(self.pop_size):
            r = random.random()

            if r < probability[0]:
                updated_population[i] = self.population[0]
            else:
                for j in range(1, self.pop_size):
                    if r >= probability[j-1] and r < probability[j]:
                        updated_population[i] = self.population[j]
                        break
        self.population = updated_population


    def cross_over(self):
        # Choose some chromosomes
        selected = []
        for i in range(self.pop_size):
            if random.random() < self.cross_over_rate:
                selected.append([self.population[i], i])

        # Perform cross-over to get children
        if len(selected) > 2:
            for i in range(len(selected) - 1):
                cross_point = random.randint(1, self.chromo_size - 1)
                selected[i][0][cross_point:], selected[i+1][0][cross_point:] = selected[i+1][0][cross_point:], selected[i][0][cross_point:]

            # Get child of first and last
            cross_point = random.randint(1, self.chromo_size - 1)
            selected[0][0][cross_point:], selected[len(selected)-1][0][cross_point:] = selected[len(selected)-1][0][cross_point:], selected[0][0][cross_point:]

        # Update population
        for chromo in selected:
            self.population[chromo[1]] = chromo[0]


    def mutation(self):
        total_genes = self.pop_size * self.chromo_size

        for i in range(int(self.mutation_rate * total_genes)):
            chromo = random.randint(0, self.pop_size - 1)
            gene = random.randint(0, self.chromo_size - 1)

            update = random.randint(0, self.no_of_parameters - 1)
            if update not in self.population[chromo]:
                self.population[chromo][gene] = update
