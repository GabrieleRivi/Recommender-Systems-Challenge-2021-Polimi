from src.HyperparameterTuning.SearchAbstractClass import SearchInputRecommenderArgs
from src.Recommenders.SLIM.SLIMElasticNetRecommender import MultiThreadSLIM_SLIMElasticNetRecommender
from Utils.utils import create_URM,create_ICM
from src.Evaluation.Evaluator import EvaluatorHoldout
from src.Data_manager.split_functions.split_train_validation_random_holdout import split_train_in_two_percentage_global_sample
from skopt.space import Real, Integer, Categorical


URM = create_URM()
ICM = create_ICM()

URM_train_validation,URM_test = split_train_in_two_percentage_global_sample(URM,train_percentage=0.85)
URM_train, URM_validation = split_train_in_two_percentage_global_sample(URM_train_validation,train_percentage=0.85)

evaluator_test = EvaluatorHoldout(URM_test, cutoff_list=[10])
evaluator_validation = EvaluatorHoldout(URM_validation, cutoff_list=[10])


from src.HyperparameterTuning.SearchBayesianSkopt import SearchBayesianSkopt
hyperparameters_range_dictionary = {
    "topK": Integer(10,500),
    "l1_ratio": Real(0,1),
    "alpha": Categorical([0.0001]),
    "workers": Categorical([10])
}
recommender_input_args = SearchInputRecommenderArgs(
    CONSTRUCTOR_POSITIONAL_ARGS = [URM_train],     # For a CBF model simply put [URM_train, ICM_train]
    CONSTRUCTOR_KEYWORD_ARGS = {},
    FIT_POSITIONAL_ARGS = [],
    FIT_KEYWORD_ARGS = {}
)
recommender_input_args_last_test = SearchInputRecommenderArgs(
    CONSTRUCTOR_POSITIONAL_ARGS = [URM_train_validation],     # For a CBF model simply put [URM_train_validation, ICM_train]
    CONSTRUCTOR_KEYWORD_ARGS = {},
    FIT_POSITIONAL_ARGS = [],
    FIT_KEYWORD_ARGS = {}
)

tuning_class = SearchBayesianSkopt(recommender_class=MultiThreadSLIM_SLIMElasticNetRecommender,
                                   evaluator_validation=evaluator_validation,
                                   evaluator_test=evaluator_test)

n_cases = 25
n_random_starts = n_cases*0.3
output_folder_path = "logs/"

tuning_class.search(recommender_input_args=recommender_input_args,
                    hyperparameter_search_space=hyperparameters_range_dictionary,
                    metric_to_optimize="MAP",
                    cutoff_to_optimize= 10,
                    n_cases= n_cases,
                    n_random_starts=n_random_starts,
                    output_folder_path=output_folder_path,
                    output_file_name_root=MultiThreadSLIM_SLIMElasticNetRecommender.RECOMMENDER_NAME,
                    save_model="best"
                    )
