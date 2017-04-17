from movielens import *

import numpy as np
import pickle
import easygui
import random
import os.path

NO_OF_RECOMMENDATIONS = 10

def load_from_dataset(utility_matrix):
    user = []
    item = []
    ratings = []
    d = Dataset()
    d.load_users("data/u.user", user)
    d.load_items("data/u.item", item)
    d.load_ratings("data/u.base", ratings)
    movie_genre = []
    for movie in item:
        movie_genre.append([movie.unknown, movie.action, movie.adventure, movie.animation, movie.childrens, movie.comedy,
                            movie.crime, movie.documentary, movie.drama, movie.fantasy, movie.film_noir, movie.horror,
                            movie.musical, movie.mystery, movie.romance, movie.sci_fi, movie.thriller, movie.war, movie.western])
    movie_genre = np.array(movie_genre)

    # Find the average rating for each user and stores it in the user's object
    for i in range(0, len(user)):
        x = utility_matrix[i]
        user[i].avg_r = sum(a for a in x if a > 0) / sum(a > 0 for a in x)

    return user, item, movie_genre, ratings

def pcs(x, y, ut, user):
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


def pop_movies(movies, movie_clusters):

	top_movies = []

	with open("avg_rating.pkl", "r") as fp:
		avg_rating = pickle.load(fp)

	for movie, rating, cluster in zip(movies, avg_rating, movie_clusters.labels_):
		
		# Threshold for average is low due to sparse dataset
		if rating > 0.1:
			top_movies.append(movie)
	return random.sample(top_movies, 100)


def update_recommendations(stored_rating, new_rating, alpha):
	# Update the average rating to include information abour latest preference
    # Alpha determines relative importance
	updated_rating = (alpha-1) * stored_rating + alpha * new_rating
	return updated_rating

def recomm_main(utility_matrix, avg_ratings, demographics, pcs_matrix):
    user, item, movie_genre, ratings = load_from_dataset(utility_matrix)
    n_users = len(user)
    n_items = len(item)

    with open("cluster.pkl", "r") as fp:
        cluster = pickle.load(fp)

    ask = pop_movies(item, cluster)
    
    print "Please rate the following movies (1-5):\nFill in 0 if you have not seen it:"
    k=0
    for movie in ask:
        print movie.title + ": "
        a = int(input())
        if a==0:
            continue
        if avg_ratings[cluster.labels_[movie.id - 1]] != 0:
            # avg_ratings[cluster.labels_[movie.id - 1]] = (avg_ratings[cluster.labels_[movie.id - 1]] + a) / 2
            avg_ratings[cluster.labels_[movie.id - 1]] = update_recommendations(avg_ratings[cluster.labels_[movie.id - 1]], a, 0.5)
        else:
            avg_ratings[cluster.labels_[movie.id - 1]] = a
        k = k+1
        if k == 10:
            break

    utility_new = np.vstack((utility_matrix, avg_ratings))

    user.append(demographics)

    print "Finding users which have similar preferences."
    for i in range(0, n_users + 1):
        if i != 943:
            pcs_matrix[i] = pcs(944, i + 1, utility_new, user)
    user_index = []
    for i in user:
        user_index.append(i.id - 1)
    user_index = user_index[:943]
    user_index = np.array(user_index)

    top_similar = [x for (y, x) in sorted(zip(pcs_matrix, user_index), key=lambda pair: pair[0], reverse=True)]
    top_5 = top_similar[:5]

    top_5_cluster = []

    for i in range(0, 5):
        maxi = 0
        maxe = 0
        for j in range(0, 19):
            if maxe < utility_matrix[top_5[i]][j]:
                maxe = utility_matrix[top_5[i]][j]
                maxi = j
        top_5_cluster.append(maxi)
    #print top_5_cluster

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
        for j in ratings:
            if j.user_id in top_5 and j.item_id == movie.id:
                movie_dict[movie.id][j.rating - 1] += 1

    recommended_movies = None
    movie_sums = []
    for movie in movie_dict:
        total = 0
        for i, j in zip(range(0, NO_OF_RECOMMENDATIONS), movie_dict[movie]):
            total += i * j
        movie_sums.append(total)
    recommended_movies = sorted(zip(movie_dict.keys(), movie_sums), key=lambda x: x[1], reverse=True)# print recommended_movies[:5]
    final_recommendations=[]
    for i in item:
        if i.id in [recommended_movies[k][0] for k in range(NO_OF_RECOMMENDATIONS)]:
            final_recommendations.append(i)
            print i.title
    return (avg_ratings, demographics, pcs_matrix, final_recommendations)

def rate_recomm(utility_matrix, avg_ratings, demographics, pcs_matrix, recommendations):
    with open("cluster.pkl", "r") as fp:
        cluster = pickle.load(fp)
    user, item, movie_genre, ratings = load_from_dataset(utility_matrix)
    user.append(demographics)
    n_users = len(user)
    n_items = len(item)
    c = 0
    for i in recommendations:
        print i.title
        r = input("Enter your rating\n")
        if r>3.5:
            c+=1

        avg_ratings[cluster.labels_[i.id - 1]] = update_recommendations(avg_rating[cluster.labels_[i.id - 1]], r, 0.5)

    print "Precision of predictions : ",c/5.0
    for i in range(0, n_users):
        if i!=943:
            utility_new = np.vstack((utility_matrix, avg_ratings))
            pcs_matrix[i] = pcs(944, i + 1, utility_new, user)
    return avg_ratings, pcs_matrix

def UI_main():
    username=raw_input("\nEnter username\n")
    if os.path.exists(username + ".pkl"):
        print "Old User"
        flag=True
        o=pickle.load(open(username+".pkl", "rb"))
        avg_ratings=o.avg_ratings
        demographics=o.demographics
        pcs_matrix=o.pcs
        recommendations=o.recommendations
    else:
        print "New User"
        flag=False
        age = input("Enter your age: ")
        gender = raw_input("M/F: ")
        pin = input("Enter pincode: ")
        profession = easygui.choicebox(msg="Select your profession",choices=["administrator",
"artist",
"doctor",
"educator",
"engineer",
"entertainment",
"executive",
"healthcare",
"homemaker",
"lawyer",
"librarian",
"marketing",
"none",
"other",
"programmer",
"retired",
"salesman",
"scientist",
"student",
"technician",
"writer",]
)
        avg_ratings = np.zeros(19)
        demographics= User(944, age,  profession, pin, gender)
        pcs_matrix = np.zeros(943)
        recommendations=[]
    utility_matrix = pickle.load(open("utility_matrix.pkl", "rb"))
    while(True):
        ch=input("\nEnter:\n 1 for getting recommendations\n 2 for rating past recommendations\n 3 to exit\n")
        if int(ch)==1:
            avg_ratings, demographics, pcs_matrix, recommendations = recomm_main(utility_matrix, avg_ratings, demographics,pcs_matrix)
        elif int(ch)==2:
            avg_ratings, pcs_matrix = rate_recomm(utility_matrix, avg_ratings, demographics, pcs_matrix, recommendations)
        else:
            break

    pickle.dump(NewUser(username, avg_ratings, demographics, pcs_matrix, recommendations), open(username+".pkl", "wb"))


if __name__=="__main__":
    UI_main()
