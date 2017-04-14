"""
Clustering
"""
from movielens import *
from ga_movie import GA_model, Chromosome

import csv
import sys
import time
import copy
import pickle

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder

SIZE_OF_POPULATION = 50
CHROMOSOME_SIZE = 10


# Store data in arrays
user = []
item = []
rating = []
rating_test = []

# Load the movie lens dataset into arrays
d = Dataset()
d.load_users("data/u.user", user)
d.load_items("data/u.item", item)
d.load_ratings("data/u.base", rating)
d.load_ratings("data/u.test", rating_test)

n_users = len(user)
n_items = len(item)

# The utility matrix stores the rating for each user-item pair in the
# matrix form.
utility = np.zeros((n_users, n_items))
for r in rating:
    utility[r.user_id - 1][r.item_id - 1] = r.rating


test = np.zeros((n_users, n_items))
for r in rating_test:
    test[r.user_id - 1][r.item_id - 1] = r.rating

# Perform clustering on items
movie_genre = []
for movie in item:
    movie_genre.append([movie.unknown, movie.action, movie.adventure, movie.animation, movie.childrens, movie.comedy,
                        movie.crime, movie.documentary, movie.drama, movie.fantasy, movie.film_noir, movie.horror,
                        movie.musical, movie.mystery, movie.romance, movie.sci_fi, movie.thriller, movie.war, movie.western])

movie_genre = np.array(movie_genre)
cluster = KMeans(n_clusters=19)
cluster.fit_predict(movie_genre)
with open("cluster.pkl", "w") as fp:
    pickle.dump(cluster, fp)

utility_clustered = []
for i in range(0, n_users):
    average = np.zeros(19)
    tmp = []
    for m in range(0, 19):
        tmp.append([])
    for j in range(0, n_items):
        if utility[i][j] != 0:
            tmp[cluster.labels_[j] - 1].append(utility[i][j])
    for m in range(0, 19):
        if len(tmp[m]) != 0:
            average[m] = np.mean(tmp[m])
        else:
            average[m] = 0
    utility_clustered.append(average)

utility_clustered = np.array(utility_clustered)

# Find the average rating for each user and stores it in the user's object
for i in range(0, n_users):
    x = utility_clustered[i]
    user[i].avg_r = sum(a for a in x if a > 0) / sum(a > 0 for a in x)
    user[i].min_r = min(x)
    user[i].max_r = max(x)

# Find the Pearson Correlation Similarity Measure between two users
def similarity(x, y):
    num = 0
    den1 = 0
    den2 = 0
    A = utility_clustered[x - 1]
    B = utility_clustered[y - 1]
    num = sum((a - user[x - 1].avg_r) * (b - user[y - 1].avg_r) for a, b in zip(A, B) if a > 0 and b > 0)
    den1 = sum((a - user[x - 1].avg_r) ** 2 for a in A if a > 0)
    den2 = sum((b - user[y - 1].avg_r) ** 2 for b in B if b > 0)
    den = (den1 ** 0.5) * (den2 ** 0.5)
    if den == 0:
        return 0
    else:
        return num / den

def similarity_utility(x, y, ut, user):
    num = 0
    den1 = 0
    den2 = 0
    A = ut[x - 1]
    B = ut[y - 1]
    num = sum((a - user[x - 1].avg_r) * (b - user[y - 1].avg_r) for a, b in zip(A, B) if a > 0 and b > 0)
    den1 = sum((a - user[x - 1].avg_r) ** 2 for a in A if a > 0)
    den2 = sum((b - user[y - 1].avg_r) ** 2 for b in B if b > 0)
    den = (den1 ** 0.5) * (den2 ** 0.5)
    if den == 0:
        return 0
    else:
        return num / den

userData=[]
for i in user:
    userData.append([i.id, i.sex, i.age, i.occupation, i.zip])

le = LabelEncoder()
le.fit([a[3] for a in userData])
le2 = LabelEncoder()
le2.fit([a[1] for a in userData])
le3 = LabelEncoder()
le3.fit([a[4] for a in userData])

userData=[]
for i in user:
    userData.append([i.id, int(le2.transform([i.sex])), i.age, int(le.transform([i.occupation])), int(le3.transform([i.zip]))])


userCluster = KMeans(n_clusters=4)
userCluster.fit_predict(userData)

for i in range(n_users):
    userCluster.labels_[i]

#print len(userCluster.labels_)

similarity_matrix = np.zeros((n_users, n_users))
s = time.time()
for i in range(0, n_users):
    for j in range(0, n_users):
        if userCluster.labels_[i]!=userCluster.labels_[j]:
            similarity_matrix[i][j]=0
            continue
        if i != j:
            similarity_matrix[i][j] = similarity(i + 1, j + 1)
            sys.stdout.write("\rGenerating Similarity Matrix [%d:%d] = %f" % (i+1, j+1, similarity_matrix[i][j]))
            sys.stdout.flush()
            time.sleep(0.00005)
print "\rGenerating Similarity Matrix [%d:%d] = %f" % (i+1, j+1, similarity_matrix[i][j])
# print time.time()-s
# Guesses the ratings that user with id, user_id, might give to item with id, i_id.
# We will consider the top_n similar users to do this.
def norm():
    normalize = np.zeros((n_users, 19))
    for i in range(0, n_users):
        for j in range(0, 19):
            if utility_clustered[i][j] != 0:
                normalize[i][j] = utility_clustered[i][j] - user[i].avg_r
            else:
                normalize[i][j] = float('Inf')
    return normalize

def guess(user_id, i_id, top_n):
    similarity = []
    for i in range(0, n_users):
        if i+1 != user_id:
            similarity.append(similarity_matrix[user_id-1][i])
    temp = norm()
    temp = np.delete(temp, user_id-1, 0)
    top = [x for (y,x) in sorted(zip(similarity,temp), key=lambda pair: pair[0], reverse=True)]
    s = 0
    c = 0
    for i in range(0, top_n):
        if top[i][i_id-1] != float('Inf'):
            s += top[i][i_id-1]
            c += 1
    g = user[user_id-1].avg_r if c == 0 else s/float(c) + user[user_id-1].avg_r
    if g < 1.0:
        return 1.0
    elif g > 5.0:
        return 5.0
    else:
        return g

utility_copy = np.copy(utility_clustered)
for i in range(0, n_users):
    for j in range(0, 19):
        if utility_copy[i][j] == 0:
            sys.stdout.write("\rGuessing [User:Rating] = [%d:%d]" % (i, j))
            sys.stdout.flush()
            time.sleep(0.0001)
            utility_copy[i][j] = guess(i+1, j+1, 150)
print "\rGuessing [User:Rating] = [%d:%d]" % (i, j)

# print utility_copy
# Utility matrix is an n_users x n_movie_clusters(hybrid genres) matrix where utility_matrix[i][j] = average rating of user i to hybrid genre j
pickle.dump( utility_copy, open("utility_matrix.pkl", "wb"))

# Predict ratings for u.test and find the mean squared error
y_true = []
y_pred = []
f = open('test.txt', 'w')
for i in range(0, n_users):
    for j in range(0, n_items):
        if test[i][j] > 0:
            f.write("%d, %d, %.4f\n" % (i+1, j+1, utility_copy[i][cluster.labels_[j]-1]))
            y_true.append(test[i][j])
            y_pred.append(utility_copy[i][cluster.labels_[j]-1])
f.close()
print "Mean Squared Error: %f" % mean_squared_error(y_true, y_pred)


selected_users = []
number_of_selected_users = 0
for i in user:
    if userCluster.labels_[i.id-1] == 0:
        selected_users.append(i)
        number_of_selected_users += 1
    if number_of_selected_users >= SIZE_OF_POPULATION:
        break

population = []
for selected_user in selected_users:
    print "The user we dealing with is:" + str(selected_user.id)
    similarity_list = []
    for i in range(0, n_users):
        if i != selected_user.id:
            similarity_list.append(similarity_utility(selected_user.id, i + 1, utility_copy, user))
    user_index = []
    for i in user:
        user_index.append(i.id - 1)
    del user_index[selected_user.id - 1]
    user_index = np.array(user_index)

    top_5 = [x for (y, x) in sorted(zip(similarity_list, user_index), key=lambda pair: pair[0], reverse=True)]
    top_5 = top_5[:5]

    top_5_cluster = []
    for i in range(0, 5):
        maxi = 0
        maxe = 0
        for j in range(0, 19):
            if maxe < utility_copy[top_5[i]][j]:
                maxe = utility_copy[top_5[i]][j]
                maxi = j
        top_5_cluster.append(maxi)
    print top_5_cluster
    k = 5

    res = {}
    for i in range(len(top_5_cluster)):
        if top_5_cluster[i] not in res.keys():
            res[top_5_cluster[i]] = len(top_5_cluster) - i
        else:
            res[top_5_cluster[i]] += len(top_5_cluster) - i

    top_cluster = res.keys()[0]
    movies_in_top_cluster = []

    for i in item:
        if cluster.labels_[i.id - 1] == top_cluster:
            movies_in_top_cluster.append(i)

    movie_dict = {movie.id: [0, 0, 0, 0, 0] for movie in movies_in_top_cluster}
    for movie in movies_in_top_cluster:
        for j in rating:
            if j.user_id in top_5 and j.item_id == movie.id:
                movie_dict[movie.id][j.rating - 1] += 1

    recommended_movies = None
    movie_sums = []
    for movie in movie_dict:
        total = 0
        for i, j in zip(range(0, CHROMOSOME_SIZE), movie_dict[movie]):
            total += i * j
        movie_sums.append(total)
    recommended_movies = sorted(zip(movie_dict.keys(), movie_sums), key=lambda x: x[1], reverse=True)

    chromosome = Chromosome()
    for i in item:
        if i.id in [recommended_movies[k][0] for k in range(CHROMOSOME_SIZE)]:
            chromosome.append(i)
    chromosome.setUID(selected_user.id)
    population.append(chromosome)


m = GA_model(population, 0.1, 0.1, len(item))
m.compute_fitness(utility_copy, cluster)

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
    m.compute_fitness(utility_copy, cluster)
    print m.fitness

maximum_fit = max(m.fitness)
for idx, fit in enumerate(m.fitness):
    if fit == maximum_fit:
        break
best_fit_chromosome = population[idx]

print "Evaluating..."
for movie in best_fit_chromosome:
    avg = []
    for r in rating:
        if userCluster.labels_[r.user_id-1] == 0 and movie.id == r.item_id:
            avg.append(((float(r.rating) - user[r.user_id-1].min_r)/(user[r.user_id-1].max_r - user[r.user_id-1].min_r))*4.0 + 1.0 )

    average_rating = 0
    if not avg:
        average_rating = 0
    else:
        average_rating = sum(avg)/float(len(avg))

    with open("recommendation_res.txt", "a") as fp:
        csvfile = csv.writer(fp)
        temp = [movie.id, movie.title, average_rating, len(avg)]
        temp.extend(avg)
        csvfile.writerow(temp)
    print movie.id, movie.title, len(avg), avg, average_rating


"""
# 1. Iterate through users to get `S` users from a cluster
2. Compute generes he might like
3. Do GA on it till convergence for the population with fitness of accuracy and diversity

A-metric:
    Ideally, the user will love our recommendations and rate it all 5
    So, as a measure of a-metric we guess how much the user will like it, ie to guess the ratings
    See, this guessing is a part of our recomendation engine, so is GA.
     So, it is alright to use this metric as we are interested in multiobjective optimisation not in evaluating the system

    To evaluate the system we should use some other metrics

"""
