##################################################################################################
#    Merging a RP3Beta GraphBased Recommender, an Implicit-ALS MatrixFactorization Recommender,  #
#    and a SLIMElasticnet Multithread Recommender                                            #
##################################################################################################
# BEST CONFIG IALS: {'epochs': 50, 'num_factors': 40, 'alpha': 0.7917156202136513, 'reg': 1.0}
# BEST CONFIG RP3: {'topK': 181, 'alpha': 0.5709402717259106, 'beta': 0.3566066452521787, 'implicit': True, 'normalize_similarity': True}
# BEST CONFIG SLIMELASTIC: {topK=420, l1_ratio = 0.06061216785568925, alpha= 0.00554982938879437, workers = 10}
from src.Recommenders.SLIM.SLIMElasticNetRecommender import MultiThreadSLIM_SLIMElasticNetRecommender
from src.Recommenders.MatrixFactorization.IALSRecommenderLinear import IALSRecommender
from src.Recommenders.GraphBased.RP3betaRecommender import RP3betaRecommender
from Utils.utils import create_URM, create_ICM, combine_matrices
from src.Hybrids.MergingModelsByScores import MergeThreeModelsByScores
from src.Evaluation.Evaluator import EvaluatorHoldout
from src.Data_manager.split_functions.split_train_validation_random_holdout import split_train_in_two_percentage_global_sample
from skopt.space import Real
from src.HyperparameterTuning.SearchAbstractClass import SearchInputRecommenderArgs

##CREATE URM,ICM, ICM STACKED COMPLETELY TO TRAIN THE SINGLE RECOMMENDERS, AND STACKED PARTIALLY TO TUNE THE HYBRID
URM = create_URM()
ICM = create_ICM()

URM_train_validation, URM_test = split_train_in_two_percentage_global_sample(URM, train_percentage=0.85)
URM_train, URM_validation = split_train_in_two_percentage_global_sample(URM_train_validation, train_percentage=0.85)

evaluator_test = EvaluatorHoldout(URM_test, cutoff_list=[10])
evaluator_validation = EvaluatorHoldout(URM_validation, cutoff_list=[10])
combined_matrices = combine_matrices(ICM=ICM, URM=URM_train)
ICM_combined = combine_matrices(ICM=ICM,URM=URM)

### RP3Beta Recommender
RP3Beta_recommender = RP3betaRecommender(URM_train=ICM_combined)
RP3Beta_recommender.fit(topK=181, alpha=0.5709402717259106, beta=0.3566066452521787, implicit=True, normalize_similarity=True)

## IALS Recommender
IALS_recommender = IALSRecommender(URM_train=ICM_combined)
IALS_recommender.fit(epochs=50, num_factors=40, alpha=0.7917156202136513, reg=1.0)

##SLIMELASTICNET Recommender
SLIME_recommender = MultiThreadSLIM_SLIMElasticNetRecommender(URM_train=ICM_combined)
SLIME_recommender.fit(topK=420, l1_ratio=0.06061216785568925, alpha=0.00554982938879437, workers = 10)

from src.HyperparameterTuning.SearchBayesianSkopt import SearchBayesianSkopt

hyperparameters_range_dictionary = {
    "alpha": Real(0.0, 1.0),
    "beta": Real(0.0, 1.0),
    "gamma": Real(0.0, 1.0)
}
recommenders = [RP3Beta_recommender, IALS_recommender, SLIME_recommender]
recommender_input_args = SearchInputRecommenderArgs(
    CONSTRUCTOR_POSITIONAL_ARGS=[combined_matrices, recommenders],
    CONSTRUCTOR_KEYWORD_ARGS={},
    FIT_POSITIONAL_ARGS=[],
    FIT_KEYWORD_ARGS={}
)
recommender_input_args_last_test = SearchInputRecommenderArgs(
    CONSTRUCTOR_POSITIONAL_ARGS=[URM_train_validation],
    CONSTRUCTOR_KEYWORD_ARGS={},
    FIT_POSITIONAL_ARGS=[],
    FIT_KEYWORD_ARGS={}
)

tuning_class = SearchBayesianSkopt(recommender_class=MergeThreeModelsByScores,
                                   evaluator_validation=evaluator_validation,
                                   evaluator_test=evaluator_test)

n_cases = 70
n_random_starts = n_cases * 0.3
output_folder_path = "logs/"

tuning_class.search(recommender_input_args=recommender_input_args,
                    hyperparameter_search_space=hyperparameters_range_dictionary,
                    metric_to_optimize="MAP",
                    cutoff_to_optimize=10,
                    n_cases=n_cases,
                    n_random_starts=n_random_starts,
                    output_folder_path=output_folder_path,
                    output_file_name_root=MergeThreeModelsByScores.RECOMMENDER_NAME,
                    save_model="best"
                    )
