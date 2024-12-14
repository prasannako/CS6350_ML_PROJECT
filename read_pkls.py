import os
import sys
import dill
import numpy as np
from bingo.evolutionary_optimizers.parallel_archipelago \
    import load_parallel_archipelago_from_file
from bingo.symbolic_regression.agraph.agraph import AGraph
from bingo.local_optimizers.scipy_optimizer import ScipyOptimizer
from bingo.local_optimizers.smcpy_optimizer import SmcpyOptimizer
from bingo.evolutionary_optimizers.island import Island
from bingo.evolutionary_algorithms.generalized_crowding import \
                                                GeneralizedCrowdingEA
from bingo.evaluation.evaluation import Evaluation
from bingo.local_optimizers.local_opt_fitness import LocalOptFitnessFunction
from bingo.local_optimizers.normalized_marginal_likelihood import NormalizedMarginalLikelihood
from bingo.symbolic_regression.explicit_regression import ExplicitRegression, ExplicitTrainingData
from bingo.symbolic_regression.agraph.generator import AGraphGenerator
from bingo.stats.pareto_front import ParetoFront
from bingo.symbolic_regression import ComponentGenerator, \
                                      AGraphGenerator, \
                                      AGraphCrossover, \
                                      AGraphMutation 

with open(filename, "rb") as load_file:
        pkl = dill.load(load_file)
        data = pkl._ea.evaluation.fitness_function.training_data
        eqn_hof = pkl.hall_of_fame
        print('Equation', 'Fitness')
        for eqn in eqn_hof:
                print(eqn, eqn.fitness)