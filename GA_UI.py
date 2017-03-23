import pickle
import random
import sys
import time
import random
import csv
import copy

import numpy as np
from sklearn.cluster import KMeans

from movielens import *
from ga import GA_model

user = []
item = []

d = Dataset()
d.load_users("./data/u.user", user)
d.load_items("./data/u.item", item)

n_users = len(user)
n_items = len(item)

utility_matrix = pickle.load(open("./utility_matrix.pkl", "rb"))

# Find the average rating for each user and stores it in the user's object
for i in range(0, n_users):
    x = utility_matrix[i]
    user[i].avg_r = sum(a for a in x if a > 0) / sum(a > 0 for a in x)

# Find the Pearson Correlation Similarity Measure between two users


def pcs(x, y, ut):
    num = 0
    den1 = 0
    den2 = 0
    A = ut[x - 1]
    B = ut[y - 1]
    num = sum((a - user[x - 1].avg_r) * (b - user[y - 1].avg_r)
              for a, b in zip(A, B) if a > 0 and b > 0)
    den1 = sum((a - user[x - 1].avg_r) ** 2 for a in A if a > 0)
    den2 = sum((b - user[y - 1].avg_r) ** 2 for b in B if b > 0)
    den = (den1 ** 0.5) * (den2 ** 0.5)
    if den == 0:
        return 0
    else:
        return num / den


# Perform clustering on items
movie_genre = []
for movie in item:
    movie_genre.append([movie.unknown, movie.action, movie.adventure, movie.animation, movie.childrens, movie.comedy,
                        movie.crime, movie.documentary, movie.drama, movie.fantasy, movie.film_noir, movie.horror,
                        movie.musical, movie.mystery, movie.romance, movie.sci_fi, movie.thriller, movie.war, movie.western])

movie_genre = np.array(movie_genre)
cluster = KMeans(n_clusters=19)
cluster.fit_predict(movie_genre)

ask = random.sample(item, 10)
new_user = np.zeros(19)

print "Please rate the following movies (1-5):"

for movie in ask:
    print movie.title + ": "
    a = int(input())
    if new_user[cluster.labels_[movie.id - 1]] != 0:
        new_user[cluster.labels_[movie.id - 1]
                 ] = (new_user[cluster.labels_[movie.id - 1]] + a) / 2
    else:
        new_user[cluster.labels_[movie.id - 1]] = a

utility_new = np.vstack((utility_matrix, new_user))

user.append(User(944, 21, 'M', 'student', 110018))

pcs_matrix = np.zeros(n_users)

print "Finding users which have similar preferences."
for i in range(0, n_users + 1):
    if i != 943:
        pcs_matrix[i] = pcs(944, i + 1, utility_new)

user_index = []
for i in user:
    user_index.append(i.id - 1)
user_index = user_index[:943]
user_index = np.array(user_index)

top_100 = [x for (y, x) in sorted(zip(pcs_matrix, user_index),
                                  key=lambda pair: pair[0], reverse=True)]
top_100 = top_100[:10]

top_100_genre = []

for i in range(0, 10):
    maxi = 0
    maxe = 0
    for j in range(0, 19):
        if maxe < utility_matrix[top_100[i]][j]:
            maxe = utility_matrix[top_100[i]][j]
            maxi = j
    top_100_genre.append(maxi)

print "Movie genres you'd like:"

for i in top_100_genre:
    if i == 0:
        print "unknown"
    elif i == 1:
        print "action"
    elif i == 2:
        print "adventure"
    elif i == 3:
        print "animation"
    elif i == 4:
        print "childrens"
    elif i == 5:
        print "comedy"
    elif i == 6:
        print "crime"
    elif i == 7:
        print "documentary"
    elif i == 8:
        print "drama"
    elif i == 9:
        print "fantasy"
    elif i == 10:
        print "film_noir"
    elif i == 11:
        print "horror"
    elif i == 12:
        print "musical"
    elif i == 13:
        print "mystery"
    elif i == 14:
        print "romance"
    elif i == 15:
        print "science fiction"
    elif i == 16:
        print "thriller"
    elif i == 17:
        print "war"
    else:
        print "western"


ga_matrix = []

for i in range(0, 10):
    ga_temp = []
    for j in range(0, 19):
        # can add the new user values also???then utility_new instead of
        # utility_matrix
        ga_temp.append(utility_matrix[top_100[i]][j])
    ga_matrix.append(ga_temp)


def ga_pcs(x, y, ut, pop):
    num = 0
    den1 = 0
    den2 = 0
    A = ga_matrix[x - 1]
    B = ut[y - 1]
    zippop = []
    k = zip(A, B)
    for l in pop:
        zippop.append(k[l])
    for l in pop:

        num = sum((a - user[x - 1].avg_r) * (b - user[y - 1].avg_r)
                  for (a, b) in zippop if a > 0 and b > 0)
        den1 = sum((a[0] - user[x - 1].avg_r) ** 2 for a in zippop if a > 0)
        den2 = sum((a[1] - user[y - 1].avg_r) ** 2 for a in zippop if a > 0)
        den = (den1 ** 0.5) * (den2 ** 0.5)
        if den == 0:
            return 0
        else:
            return num / den


# change 5 to 9 for only Naive Bayes
m = GA_model(0.25, 0.1, 30, 6, 19, ga_pcs)
m.compute_fitness(utility_new)

for i in range(50):

    if i == 0:
        temp = copy.deepcopy(m)
    else:
        if (sum(temp.fitness) / float(len(temp.fitness))) >= (sum(m.fitness) / float(len(m.fitness))):
            m = copy.deepcopy(temp)
        else:
            temp = copy.deepcopy(m)

    m.roulette_selection()
    m.cross_over()
    m.mutation()
    m.compute_fitness(utility_new)

maximum_fit = max(m.fitness)

for i in range(len(m.fitness)):
    if m.fitness[i] == maximum_fit:
        print (m.fitness[i], m.population[i])
print "Movie genres you'd like chosen by genetic algorithm:"

for i in m.population[i]:
    if i == 0:
        print "unknown"
    elif i == 1:
        print "action"
    elif i == 2:
        print "adventure"
    elif i == 3:
        print "animation"
    elif i == 4:
        print "childrens"
    elif i == 5:
        print "comedy"
    elif i == 6:
        print "crime"
    elif i == 7:
        print "documentary"
    elif i == 8:
        print "drama"
    elif i == 9:
        print "fantasy"
    elif i == 10:
        print "film_noir"
    elif i == 11:
        print "horror"
    elif i == 12:
        print "musical"
    elif i == 13:
        print "mystery"
    elif i == 14:
        print "romance"
    elif i == 15:
        print "science fiction"
    elif i == 16:
        print "thriller"
    elif i == 17:
        print "war"
    else:
        print "western"
