from schelling import *
import matplotlib.pyplot as plt

if __name__ == '__main__':
    #Second Simulation Measuring Seggregation
    similarity_threshold_ratio = {}
    experiment_ratios = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
    experiment_times = 3
    for i in experiment_ratios:
        similarity_threshold_ratio[i] = 0.0
    for j in range(0, experiment_times):
        for i in experiment_ratios: 
            schelling = Schelling(50, 50, 0.1, i, 500, 4)
            schelling.prepare()
            schelling.update()
            similarity_threshold_ratio[i] += schelling.calculate_homo()
    for value in experiment_ratios:
        similarity_threshold_ratio[value] /= experiment_times
 
    print('---result of seggregation---')
    for i in experiment_ratios:
        print('%s : %s' % (i, similarity_threshold_ratio[i]))
 
    fig, ax = plt.subplots()
    plt.plot(similarity_threshold_ratio.keys(), similarity_threshold_ratio.values(), 'ro')
    ax.set_title('Similarity Threshold vs. Mean Similarity Ratio', fontsize=15, fontweight='bold')
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.1])
    ax.set_xlabel("Similarity Threshold")
    ax.set_ylabel("Mean Similarity Ratio")
    plt.savefig('0.7_similarity_seggregation.png')
