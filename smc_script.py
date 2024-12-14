import os
import sys
import dill
import time
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from bingo.symbolic_regression.agraph.agraph import AGraph
from bingo.local_optimizers.local_opt_fitness import LocalOptFitnessFunction
from bingo.symbolic_regression.explicit_regression import ExplicitTrainingData, ExplicitRegression
from bingo.local_optimizers.scipy_optimizer import ScipyOptimizer
from bingo.local_optimizers.normalized_marginal_likelihood import NormalizedMarginalLikelihood
from bingo.evolutionary_algorithms.generalized_crowding import \
                                                GeneralizedCrowdingEA
from bingo.selection.deterministic_crowding import DeterministicCrowding
from bingo.evaluation.evaluation import Evaluation
from bingo.evolutionary_optimizers.island import Island
from bingo.stats.pareto_front import ParetoFront
from bingo.symbolic_regression import ComponentGenerator, \
                                      AGraphGenerator, \
                                      AGraphCrossover, \
                                      AGraphMutation
from bingo.util import log


POP_SIZE = 100 
STACK_SIZE = 32
MAX_GEN = 100
FIT_THRESH = -np.inf
CHECK_FREQ = 50
MIN_GEN = 1

PARTICLES = 10
MCMC_STEPS = 1
ESS_THRESHOLD = 0.75

def run_SMC_bingo():
    log.configure_logging(verbosity=log.DETAILED_INFO,
                      module=False, timestamp=False,
                      stats_file='trainingstats', logfile='traininglog')
    # data = np.load("../data/"+str(sys.argv[-2])+".npy")
    current_directory = os.getcwd()
    df = pd.read_csv(os.path.join(current_directory, 'data', 'train_data.csv'))
    df = df[df['ID'] == 1]
    df = df.sample(frac=1, random_state=42)
    df.reset_index(drop=True, inplace=True)
    features = ['a', 'c/b', 'a^2']
    target = 'fw'
    # scaler = StandardScaler()
    # df[df.columns] = scaler.fit_transform(df[df.columns])
    X = df[features].values
    y = df[target].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,  random_state=42)

    explicit_data = ExplicitTrainingData(X_train,y_train)
    fitness = ExplicitRegression(explicit_data)
    scipy_opt = ScipyOptimizer(fitness, method="lm")
    nmll = NormalizedMarginalLikelihood(
        fitness,
        scipy_opt,
        num_particles=PARTICLES,
        mcmc_steps=MCMC_STEPS
    )

    component_generator = ComponentGenerator(explicit_data.x.shape[1])
    component_generator.add_operator("+")
    component_generator.add_operator("-")
    component_generator.add_operator("*")
    component_generator.add_operator("/")

    crossover = AGraphCrossover()
    mutation = AGraphMutation(component_generator)

    agraph_generator = AGraphGenerator(STACK_SIZE, component_generator,
                                       use_simplification=True)

    pareto_front = ParetoFront(secondary_key = lambda ag: ag.get_complexity(), 
                            similarity_function=agraph_similarity)

    evaluator = Evaluation(fitness, redundant=True, multiprocess=20)

    ea = GeneralizedCrowdingEA(evaluator, crossover,
                      mutation, 0.4, 0.4)
    
    island = Island(ea, agraph_generator, POP_SIZE, hall_of_fame=pareto_front)
    
    ERROR_TOLERANCE = 1e-5

    best_indv_values = []
    best_indv_values.append(island.get_best_individual())
    best_indv_fitness = []
    best_indv_fitness.append(island.get_best_individual().fitness)
    best_indv_gen = []
    best_indv_gen.append(island.generational_age)

    da = pd.DataFrame(columns=["Generational Age", "individual","Fitness"])
    da.to_csv("best_individuals.csv", index=False, mode='w')  

    for _ in range(MAX_GEN):
        island.evolve(1)
        best_indv = island.get_best_individual()
        # if best_indv.fitness < best_indv_values[-1].fitness:
        best_indv_values.append(best_indv)
        best_indv_gen.append(island.generational_age)
        best_indv_fitness.append(best_indv.fitness)

        print(island.generational_age)
        print(best_indv.fitness)

        new_row = {"Generational Age": island.generational_age, "individual": best_indv, "Fitness": best_indv.fitness}
        pd.DataFrame([new_row]).to_csv("best_individuals.csv", index=False, mode='a', header=False)



    print("Generation: ", island.generational_age)
    print("Success!")
    print("Best individual\n f(X_0) =", island.get_best_individual())

    # plt.figure(figsize=(8, 6))
    plt.plot(best_indv_gen, best_indv_fitness, marker='o', linestyle='-', color='b')
    plt.xlabel("Generational Age")
    plt.ylabel("Fitness")
    plt.title("Generational Age vs. Fitness")
    plt.grid(True)
    plt.show()

    # plt.plot(best_indv_gen, best_indv_fitness, marker='o', linestyle='-', color='b')
    # plt.xlabel("Generational Age")
    # plt.ylabel("Fitness")
    # plt.title("Generational Age vs. Fitness")
    # plt.grid(True)
    # plt.show()
        
    # da2 = pd.DataFrame(columns=["Fitness", "Complexity"])
    # da2 = da2.append({"Fitness": member.fitness, "Complexity": member.get_complexity()}, ignore_index=True)
    # da2.to_csv("pareto_front.csv", index=False)

    # try:
    #     gens = []
    #     for fname in os.listdir():
    #         if fname.endswith('.pkl'):
    #             gens.append(int(fname.split('_')[1].split('.')[0]))
    #     maxgen = np.max(gens)
    #     with open('checkpoint_'+str(maxgen)+'.pkl', "rb") as load_file:
    #         island = dill.load(load_file)
    # except:
    #     island = Island(ea, agraph_generator, POP_SIZE, hall_of_fame=pareto_front)
    # opt_result = island.evolve_until_convergence(max_generations=MAX_GEN,
    #                                               fitness_threshold=FIT_THRESH,
    #                                     convergence_check_frequency=CHECK_FREQ,
    #                                           checkpoint_base_name='checkpoint')

def agraph_similarity(ag_1, ag_2):
    return ag_1.fitness == ag_2.fitness and \
                            ag_1.get_complexity() == ag_2.get_complexity()

if __name__ == '__main__':
    # dirname = sys.argv[-1]
    # if not os.path.exists(dirname):
    #     os.makedirs(dirname)
    # os.chdir(dirname)
    start_time = time.time()
    run_SMC_bingo()
    logging.info("--- %s seconds elapsed ---" % (time.time() - start_time))
