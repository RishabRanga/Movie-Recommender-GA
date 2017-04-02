import random
import csv
import copy
import sys
import numpy as np
import random
import itertools

from movielens import *
#from NB import Naive_Bayes
#from NB_int import *
#from sklearn.naive_bayes import GaussianNB

class GA_model(object):
    def __init__(self, population,  cross_over_rate, mutation_rate, no_of_parameters):
        self.cross_over_rate = cross_over_rate
        self.mutation_rate = mutation_rate
        self.no_of_parameters = no_of_parameters
        self.population = population
        self.pop_size = len(population)
        self.chromo_size = len(self.population[0])
        self.fitness = [0] * self.pop_size

    def fitness_fun(self, chromosome, utility_matrix, cluster, d_weight=0.5, a_weight=0.5):
        """
        Calculate the fitness of each chromosome
        :param chromosome:
        :param utility_matrix:
        :param cluster:
        :param d_weight:
        :param a_weight:
        :return weighted average of accuracy and diversity scores
        """

        # Accuracy
        uid = chromosome.uid
        a_score = 0        # a score out of 25
        for gene in chromosome:
            a_score += utility_matrix[uid][cluster.labels_[gene.id]]
        a_score /= 5*len(chromosome)

        # Diversity
        n_different_movie_cluster = len(set(cluster.labels_[gene.id] for gene in chromosome))
        total_n_movie_cluster = cluster.n_clusters
        d_score = n_different_movie_cluster / float(total_n_movie_cluster)

        # Weighted score
        score = (a_weight*(a_score) + d_weight*(d_score)) / ( 2 * (a_weight+d_weight) )

        return score


    def compute_fitness(self, utility_matrix, cluster):
        for i in range(self.pop_size):
            self.fitness[i] = self.fitness_fun(self.population[i], utility_matrix, cluster)

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
                selected.append(i)
        random.shuffle(self.population)
        for chromo1,chromo2 in itertools.combinations(selected, 2):
            for i in range(0, len(self.population[chromo1])):
                if self.population[chromo1][i] !=  self.population[chromo2][i]:
                    if random.choice([True, False]):
                        self.population[chromo1][i], self.population[chromo2][i] = self.population[chromo2][i], self.population[chromo1][i]





    def mutation(self):
        total_genes = self.pop_size * self.chromo_size
        item = []
        d = Dataset()
        d.load_items("data/u.item", item)
        for i in range(int(self.mutation_rate * total_genes)):
            chromo = random.randint(0, self.pop_size - 1)
            gene = random.randint(0, self.chromo_size - 1)

            update = random.randint(0, self.no_of_parameters - 1)
            if update not in self.population[chromo]:
                self.population[chromo][gene] = item[update]
