'''
Author : Adil Moujahid
Email : adil.mouja@gmail.com
Date : 10-Sep-2014
Description: Simulations of Schelling's seggregation model
http://www.binpress.com/tutorial/introduction-to-agentbased-models-an-implementation-of-schelling-model-in-python/144
'''

import matplotlib.pyplot as plt
import itertools
import random
import copy
import numpy as np

class Schelling:
    def __init__(self, width, height, empty_ratio, similarity_threshold, n_iterations, races = 2):
        self.width = width 
        self.height = height 
        self.races = races
        self.empty_ratio = empty_ratio
        self.similarity_threshold = similarity_threshold
        self.n_iterations = n_iterations
        self.unsatified = [[], []]
    
    def prepare(self):
        self.empty_houses = []
        self.agents = {}
        # need to add the border for easily matrix calculation
        self.empty_matrix = np.zeros((self.width+2, self.height+2))
        self.feature_matrix = np.zeros((self.width+2, self.height+2))

        self.all_houses = list(itertools.product(range(self.width),range(self.height)))
        # print(self.all_houses)
        random.shuffle(self.all_houses)

        self.n_empty = int( self.empty_ratio * len(self.all_houses) )
        self.empty_houses = self.all_houses[:self.n_empty]

        self.remaining_houses = self.all_houses[self.n_empty:]
        houses_by_race = [self.remaining_houses[i::self.races] for i in range(self.races)]
        # print(houses_by_race)
        for i in range(self.races):
            self.agents = {**self.agents, **dict(zip(houses_by_race[i], [i+1]*len(houses_by_race[i])))}
        for location, culture in self.agents.items():
            self.empty_matrix[location[0]+1][location[1]+1] = 1
            self.feature_matrix[location[0]+1][location[1]+1] = culture
        #print('empty: \n %s' % self.empty_matrix)
        #print('feature: \n %s' % self.feature_matrix)

    def calculate_similarity(self):
        feature = self.feature_matrix[1:-1,1:-1]
        # print('feature extract: \n %s' % feature)
        surroudings = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1), (2, 2)]
        # just for test
        # surroudings = [(0,0)]
        sum_differ = np.zeros((self.width, self.height))
        sum_neighbor = np.zeros((self.width, self.height))
        for x_context, y_context in surroudings:
            differ_matrix = self.feature_matrix[
                    x_context:x_context+self.width,
                    y_context:y_context+self.height]
            empty_matrix = self.empty_matrix[
                    x_context:x_context+self.width,
                    y_context:y_context+self.height]
            differ_matrix = np.abs(differ_matrix-feature)
            # print('after abs: \n%s' % differ_matrix)
            differ_matrix /= denominator(differ_matrix)
            # print('after to 1.0: \n%s' % differ_matrix)
            # change same value to -1.0, different value to 1.0
            differ_matrix = 2.0 * (differ_matrix - 0.5)
            # print('after differ: \n%s' % differ_matrix)
            # send empty value -> 0.0
            differ_matrix *= empty_matrix
            # print(differ_matrix)
            sum_differ += (differ_matrix + np.abs(differ_matrix))/2
            # -1 -> 1, all neighbors
            sum_neighbor += np.abs(differ_matrix)
        return sum_differ, sum_neighbor

    def calculate_homo(self):
        sum_differ, sum_neighbor = self.calculate_similarity()
        # the sum_same is for the old version of sum_differ, which differ -> -1.0
        # sum_same = (sum_neighbor - sum_differ)/2
        # sum_differ += sum_same
        #print('sum_differ: \n%s' % sum_differ)
        #print('sum_neighbor: \n%s' % sum_neighbor)
        # return np.sum(sum_same) / np.sum(sum_neighbor)
        return 1.0 - np.sum(sum_differ) / np.sum(sum_neighbor)

    def calculate_unsatisfied(self):
        sum_differ, sum_neighbor = self.calculate_similarity()
        # print('sum_differ: \n%s' % sum_differ)
        # print('sum_neighbor: \n%s' % sum_neighbor)
        differ_ratio = sum_differ / denominator(sum_neighbor)
        differ_ratio *= self.empty_matrix[1:-1,1:-1]
        # print(differ_ratio)
        self.unsatified = np.where(differ_ratio > 1-self.similarity_threshold)
        # print(len(self.unsatified))
        # print(self.unsatified)

    def update(self):
        for i in range(self.n_iterations):
            # print(self.calculate_homo())
            if i % 20 == 0:
                print('iterations: %d' % i)
                print('homo: %s' %self.calculate_homo())
                print('unsatisfied: %s' % len(self.unsatified[0]))
                # print('occupied: %s' % np.sum(self.empty_matrix))
            self.calculate_unsatisfied()
            self.search_empty()
            self.move()
            #if i % 20 == 0:
            #    print(len(self.unsatified[0]))
            # print(self.calculate_homo())
            # print(len(self.unsatified[0]))

    def search_empty(self):
        empty = self.empty_matrix[1:-1, 1:-1]
        self.empty = np.where(empty == 0.0)
        # print(self.empty)

    def move(self):
        unsatified = [(self.unsatified[0][i], self.unsatified[1][i]) 
                for i in range(0, len(self.unsatified[0]))]
        empty = [(self.empty[0][i], self.empty[1][i]) 
                for i in range(0, len(self.empty[0]))]
        random.shuffle(unsatified)
        if len(unsatified) > len(empty):
            unsatified = unsatified[:len(empty)]
        #print(unsatified)
        #print(empty)
        for i in range(0, len(unsatified)):
            x_from, y_from = unsatified[i][0]+1, unsatified[i][1]+1
            x_to, y_to = empty[i][0]+1, empty[i][1]+1
            self.empty_matrix[x_from][y_from] = 0.0
            self.empty_matrix[x_to][y_to] = 1.0
            self.feature_matrix[x_to][y_to] = self.feature_matrix[x_from][y_from]
            self.feature_matrix[x_from][y_from] = 0
            # print(self.feature_matrix)
        # print('occupied: %s' % (np.sum(self.empty_matrix)))


    def plot(self, title, file_name, figsize=None):
        #if figsize is None:
        #    figsize = (self.width, self.height)
        #fig, ax = plt.subplots(figsize=figsize)
        fig, ax = plt.subplots()
        #If you want to run the simulation with more than 7 colors, you should set agent_colors accordingly
        agent_colors = {1:'b', 2:'r', 3:'g', 4:'c', 5:'m', 6:'y', 7:'k'}
        #for agent in self.agents:
        #    ax.scatter(agent[0]+0.5, agent[1]+0.5, color=agent_colors[self.agents[agent]])
        #for x in range(1, self.width+1):
        #    for y in range(1, self.height+1):
        #        type_no = self.feature_matrix[x][y]
        #        if type_no > 0:
        #            ax.scatter(x-0.5, y-0.5, color = agent_colors[type_no])
        for key, value in agent_colors.items():
            location = np.where(self.feature_matrix == key)
            ax.scatter(location[0], location[1], color=value)

        ax.set_title(title, fontsize=10, fontweight='bold')
        ax.set_xlim([0, self.width])
        ax.set_ylim([0, self.height])
        ax.set_xticks([])
        ax.set_yticks([])
        print('save: %s' % file_name)
        plt.savefig(file_name)

class CultureSchelling(Schelling):


def main():

    ##First Simulation
    # the parameters of class Schelling:
    # width, height, empty_ratio, similarity_threshold, n_iterations, races = 2
    schelling_1 = Schelling(50, 50, 0.2, 0.4, 2000, 3)
    schelling_1.prepare()
    schelling_1.plot('Schelling Model with 3 colors: Initial State', 'schelling_3_initial.png')
    schelling_1.update()
    schelling_1.plot('Schelling Model with 3 colors: Final State with Happiness Threshold 20%', 'schelling_3_20_final.png')

	#schelling_2 = Schelling(50, 50, 0.3, 0.5, 500, 2)
	#schelling_2.prepare()

	#schelling_3 = Schelling(50, 50, 0.3, 0.8, 500, 2)
	#schelling_3.prepare()

	# schelling_1.plot('Schelling Model with 5 colors: Initial State', 'schelling_5_initial.png')

	# schelling_1.update()
	#schelling_2.update()
	#schelling_3.update()

	# schelling_1.plot('Schelling Model with 5 colors: Final State with Happiness Threshold 20%', 'schelling_5_20_final.png')
	#schelling_2.plot('Schelling Model with 2 colors: Final State with Happiness Threshold 50%', 'schelling_2_50_final.png')
	#schelling_3.plot('Schelling Model with 2 colors: Final State with Happiness Threshold 80%', 'schelling_2_80_final.png')


	##Second Simulation Measuring Seggregation
	#similarity_threshold_ratio = {}
	#for i in [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]:
	#	schelling = Schelling(50, 50, 0.3, i, 500, 2)
	#	schelling.prepare()
	#	schelling.update()
	#	similarity_threshold_ratio[i] = schelling.calculate_similarity()

        #print('---result of seggregation---')
        #for i in [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]:
        #    print('%s : %s' % (i, similarity_threshold_ratio[i])

	#fig, ax = plt.subplots()
	#plt.plot(similarity_threshold_ratio.keys(), similarity_threshold_ratio.values(), 'ro')
	#ax.set_title('Similarity Threshold vs. Mean Similarity Ratio', fontsize=15, fontweight='bold')
	#ax.set_xlim([0, 1])
	#ax.set_ylim([0, 1.1])
	#ax.set_xlabel("Similarity Threshold")
	#ax.set_ylabel("Mean Similarity Ratio")
	#plt.savefig('schelling_segregation.png')

def denominator(to_transfer):
    # n = np.array([to_transfer, [1.0 for num in to_transfer]])
    n = np.ones(to_transfer.shape)
    return np.max(np.array([n,to_transfer]), axis=0)
 
if __name__ == "__main__":
    main()
