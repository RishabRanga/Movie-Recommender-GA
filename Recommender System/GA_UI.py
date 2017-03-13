import re

# Read data/README to get more info on these data structures
class User:
    def __init__(self, id, age, sex, occupation, zip):
        self.id = int(id)
        self.age = int(age)
        self.sex = sex
        self.occupation = occupation
        self.zip = zip
        self.avg_r = 0.0

# Read data/README to get more info on these data structures
class Item:
    def __init__(self, id, title, release_date, video_release_date, imdb_url, \
    unknown, action, adventure, animation, childrens, comedy, crime, documentary, \
    drama, fantasy, film_noir, horror, musical, mystery ,romance, sci_fi, thriller, war, western):
        self.id = int(id)
        self.title = title
        self.release_date = release_date
        self.video_release_date = video_release_date
        self.imdb_url = imdb_url
        self.unknown = int(unknown)
        self.action = int(action)
        self.adventure = int(adventure)
        self.animation = int(animation)
        self.childrens = int(childrens)
        self.comedy = int(comedy)
        self.crime = int(crime)
        self.documentary = int(documentary)
        self.drama = int(drama)
        self.fantasy = int(fantasy)
        self.film_noir = int(film_noir)
        self.horror = int(horror)
        self.musical = int(musical)
        self.mystery = int(mystery)
        self.romance = int(romance)
        self.sci_fi = int(sci_fi)
        self.thriller = int(thriller)
        self.war = int(war)
        self.western = int(western)

# Read data/README to get more info on these data structures
class Rating:
    def __init__(self, user_id, item_id, rating, time):
        self.user_id = int(user_id)
        self.item_id = int(item_id)
        self.rating = int(rating)
        self.time = time

# The dataset class helps you to load files and create User, Item and Rating objects
class Dataset:
    def load_users(self, file, u):
        f = open(file, "r")
        text = f.read()
        entries = re.split("\n+", text)
        for entry in entries:
            e = entry.split('|', 5)
            if len(e) == 5:
                u.append(User(e[0], e[1], e[2], e[3], e[4]))
        f.close()

    def load_items(self, file, i):
        f = open(file, "r")
        text = f.read()
        entries = re.split("\n+", text)
        for entry in entries:
            e = entry.split('|', 24)
            if len(e) == 24:
                i.append(Item(e[0], e[1], e[2], e[3], e[4], e[5], e[6], e[7], e[8], e[9], e[10], \
                e[11], e[12], e[13], e[14], e[15], e[16], e[17], e[18], e[19], e[20], e[21], \
                e[22], e[23]))
        f.close()

    def load_ratings(self, file, r):
        f = open(file, "r")
        text = f.read()
        entries = re.split("\n+", text)
        for entry in entries:
            e = entry.split('\t', 4)
            if len(e) == 4:
                r.append(Rating(e[0], e[1], e[2], e[3]))
        f.close()

        
import pickle

utility_matrix = pickle.load( open("C:/Users/rishab/Desktop/IR Project/Movie-Recommender-System-master/Movie-Recommender-System-master/utility_matrix.pkl", "rb") )

from sklearn.cluster import KMeans

import numpy as np
import pickle
import random
import sys
import time

user = []
item = []

d = Dataset()
d.load_users("C:/Users/rishab/Desktop/IR Project/Movie-Recommender-System-master/Movie-Recommender-System-master/data/u.user", user)
d.load_items("C:/Users/rishab/Desktop/IR Project/Movie-Recommender-System-master/Movie-Recommender-System-master/data/u.item", item)

n_users = len(user)
n_items = len(item)

utility_matrix = pickle.load( open("C:/Users/rishab/Desktop/IR Project/Movie-Recommender-System-master/Movie-Recommender-System-master/utility_matrix.pkl", "rb") )

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
    num = sum((a - user[x - 1].avg_r) * (b - user[y - 1].avg_r) for a, b in zip(A, B) if a > 0 and b > 0)
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
		new_user[cluster.labels_[movie.id - 1]] = (new_user[cluster.labels_[movie.id - 1]] + a) / 2
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

top_100 = [x for (y,x) in sorted(zip(pcs_matrix, user_index), key=lambda pair: pair[0], reverse=True)]
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
        

ga_matrix=[]

for i in range(0, 10):
    ga_temp=[]
    for j in range(0, 19):
        ga_temp.append(utility_matrix[top_100[i]][j])#can add the new user values also???then utility_new instead of utility_matrix
    ga_matrix.append(ga_temp)


#A=utility_new[942]
#B=new_user
#pop=[1,4,0,5,6]
#zippop=[]
#k=zip(A,B)
#for l in pop:
#zippop.append(k[l])
    

def ga_pcs(x, y, ut,pop):
    num = 0
    den1 = 0
    den2 = 0
    A = ga_matrix[x - 1]
    B = ut[y - 1]
    zippop=[]
    k=zip(A,B)
    for l in pop:
        zippop.append(k[l])
    for l in pop:
    
        num = sum((a - user[x - 1].avg_r) * (b - user[y - 1].avg_r) for (a, b) in zippop if a > 0 and b > 0)
        den1 = sum((a[0] - user[x - 1].avg_r) ** 2 for a in zippop if a > 0)
        den2 = sum((a[1] - user[y - 1].avg_r) ** 2 for a in zippop if a > 0)
        den = (den1 ** 0.5) * (den2 ** 0.5)
        if den == 0:
            return 0
        else:
            return num / den
    
    






















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
	
	def __init__(self, cross_over_rate, mutation_rate, pop_size, chromo_size, no_of_parameters):
		self.pop_size = pop_size
		self.chromo_size = chromo_size
		self.cross_over_rate = cross_over_rate
		self.mutation_rate = mutation_rate
		self.no_of_parameters = no_of_parameters
		self.population = [
			(random.sample(range(0, no_of_parameters), chromo_size)) for i in range(pop_size)
		]
		self.fitness = [0] * pop_size
		
		self.data = []
		#with open('Glass_New.csv') as f:
			#reader = csv.reader(f, delimiter = ',')
			#self.data = list(reader)
		#self.data = self.data[1:]
		self.data=ga_matrix

	def fitness_fun(self, input):

		#np_data = np.asarray(self.data)
		#y = np_data[:,-1]
		#i = 0
		#for n in input:
		#	if i == 0:
				#x = np_data[:,n]
				#i += 1
			#else:
				#x = np.vstack((x,np_data[:,n]))
		#x = x.T
		#x_new = np.zeros(shape=(x.shape[0], x.shape[1]+1))

		#for i in range(len(np_data)):
			#x_new[i] = np.append(x[i], y[i])

		
		

		# Option 2 native
		#splitRatio = 0.7
		#dataset = x_new
		#train, test = splitDataset(dataset, splitRatio)
		# prepare model
		#summaries = summarizeByClass(train)
		# test model
		#predictions = getPredictions(summaries, test)
		#accuracy = getAccuracy(test, predictions)
		accuracy=[]
		for i in range(1,11):
			accuracy.append(ga_pcs(i,944,utility_new,input))
		#print(max(accuracy))

		return max(accuracy)
	
	def compute_fitness(self):
		for i in range(self.pop_size):
			self.fitness[i] = self.fitness_fun(self.population[i])


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

def main():
	m = GA_model(0.25, 0.1, 30, 6, 19)#change 5 to 9 for only Naive Bayes
	m.compute_fitness()

	for i in range(50):

		if i == 0:
			temp = copy.deepcopy(m)
		else:
			if (sum(temp.fitness)/float(len(temp.fitness)))>=(sum(m.fitness)/float(len(m.fitness))):
				m = copy.deepcopy(temp)
			else:
				temp = copy.deepcopy(m)

		m.roulette_selection()
		# print zip(m.population, m.fitness)
		m.cross_over()

		m.mutation()
		m.compute_fitness
		# x = input()

	
	#	print x
	print("the chosen feature set and accuracy are:")
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

if __name__ == '__main__':
	main()