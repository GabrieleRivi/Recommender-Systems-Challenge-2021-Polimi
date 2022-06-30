from src.Recommenders.SLIM.SLIMElasticNetRecommender import MultiThreadSLIM_SLIMElasticNetRecommender
from src.Recommenders.GraphBased.RP3betaRecommender import RP3betaRecommender
from src.Recommenders.MatrixFactorization.IALSRecommenderLinear import IALSRecommender
from src.Recommenders.KNN.UserKNNCFRecommender import UserKNNCFRecommender
from Utils.utils import create_URM, create_ICM
from src.Hybrids.MergingModelsByScores import MergeThreeModelsByScores
from src.Evaluation.Evaluator import EvaluatorHoldout
from src.Data_manager.split_functions.split_train_validation_random_holdout import \
    split_train_in_two_percentage_global_sample
from skopt.space import Real, Integer, Categorical
from src.HyperparameterTuning.SearchAbstractClass import SearchInputRecommenderArgs

URM = create_URM()
ICM = create_ICM()

URM_train, URM_validation = split_train_in_two_percentage_global_sample(URM, train_percentage=0.85)
evaluator_validation = EvaluatorHoldout(URM_validation, cutoff_list=[10])

### RP3Beta Recommender
RP3Beta_recommender = RP3betaRecommender(URM_train=URM_train)
RP3Beta_recommender.fit(topK=67, alpha=1.0, beta=0.6676517342477193, implicit=True, normalize_similarity=True)

## IALS Recommender
IALS_recommender = IALSRecommender(URM_train=URM_train)
IALS_recommender.fit(epochs=10, num_factors=49, alpha=1.0, reg=0.01)

##USERCF Recommender
UserCF_recommender = UserKNNCFRecommender(URM_train=URM_train)
UserCF_recommender.fit(topK=200, shrink=10, similarity="jaccard")

tuning_params = {
    "alpha": (0.0, 1.0),
    "beta": (0.0, 1.0),
    "gamma": (0.0, 1.0)
}

from bayes_opt import BayesianOptimization, JSONLogger, Events

recommenders = [RP3betaRecommender, IALS_recommender, UserCF_recommender]


def BO_func(alpha, beta, gamma):
    recommender = MergeThreeModelsByScores(URM_train=URM_train,
                                           recommenders=recommenders)

    recommender.fit(alpha, beta, gamma)

    MAP = evaluator_validation.evaluateRecommender(recommender)[10]["MAP"]
    print("MAP: " + str(MAP) + "-> alpha: " + str(alpha) + " beta: " + str(beta) + " gamma: " + str(gamma) + "\n")
    return MAP


# ----------------------------------------------------------------------------------------------------------
# Defining optimizers attributes

optimizer = BayesianOptimization(
    f=BO_func,
    pbounds=tuning_params,
    verbose=5
)

# Defining a logger to save the tuning
logger = JSONLogger(path="./hybrid2.json")
optimizer.subscribe(Events.OPTIMIZATION_STEP, logger)

optimizer.maximize(
    init_points=60,
    n_iter=70,
)

# ----------------------------------------------------------------------------------------------------------
# printing the final best result using optimizer.max
print("\n\nRESULT\n\n")
print(optimizer.max)
