from schelling import *
import matplotlib.pyplot as plt
import numpy as np

def sum_square(distribution):
    s = 0.0
    for value in distribution:
        s += value ** 2
    return s

if __name__ == '__main__':
    #Second Simulation Measuring Seggregation
    similarity_threshold_ratio = {}
    experiment_ratios = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
    experiment_times = 3
    experiment_races = {}
    base_similarity_ratio = {}
    for i in experiment_ratios:
        similarity_threshold_ratio[i] = 0.0
        experiment_races[i] = []
        base_similarity_ratio[i] = 0.0
    for j in range(0, experiment_times):
        for i in experiment_ratios: 
            height, weight, empty_ratio = 50, 50, 0.3
            schelling = CultureSchelling(height, weight, empty_ratio, i, 500, [0.01, 0.05, 0.10, 0.20], 4)
            schelling.prepare()
            schelling.update()
            similarity_threshold_ratio[i] += schelling.calculate_homo()
            # print(height*weight*(1-empty_ratio))
            # print(schelling.race_distribution())
            # experiment_races[i].append(np.array(schelling.race_distribution())/(height*weight*(1-empty_ratio)))
            # experiment_races[i].append(
            distribution = np.array(schelling.race_distribution())/(height*weight*(1-empty_ratio))
            experiment_races[i].append(distribution)
            base_similarity_ratio[i] += sum_square(distribution)

    for value in experiment_ratios:
        similarity_threshold_ratio[value] /= experiment_times
        base_similarity_ratio[value] /= experiment_times
 
    print('---result of seggregation---')
    for i in experiment_ratios:
        print('%s : %s' % (i, similarity_threshold_ratio[i]))
        print(experiment_races[i])
        print(base_similarity_ratio[i])
 
    fig, ax = plt.subplots()
    plt.plot(similarity_threshold_ratio.keys(), similarity_threshold_ratio.values(), 'ro')
    ax.set_title('Similarity Threshold vs. Mean Similarity Ratio', fontsize=15, fontweight='bold')
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.1])
    ax.set_xlabel("Similarity Threshold")
    ax.set_ylabel("Mean Similarity Ratio")
    plt.savefig('2.2_similarity_seggregation.png')
