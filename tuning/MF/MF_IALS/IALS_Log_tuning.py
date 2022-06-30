from src.HyperparameterTuning.SearchAbstractClass import SearchInputRecommenderArgs
from src.Recommenders.MatrixFactorization.IALSRecommenderLinear import IALSRecommender
from Utils.utils import create_URM,create_ICM,combine_matrices
from src.Evaluation.Evaluator import EvaluatorHoldout
from src.Data_manager.split_functions.split_train_validation_random_holdout import split_train_in_two_percentage_global_sample
from skopt.space import Real, Integer, Categorical

URM = create_URM()
ICM = create_ICM()

URM_train_validation,URM_test = split_train_in_two_percentage_global_sample(URM,train_percentage=0.85)
URM_train, URM_validation = split_train_in_two_percentage_global_sample(URM_train_validation,train_percentage=0.85)


CombinedMatrix = combine_matrices(ICM=ICM, URM=URM_train)

evaluator_test = EvaluatorHoldout(URM_test, cutoff_list=[10])
evaluator_validation = EvaluatorHoldout(URM_validation, cutoff_list=[10])


from src.HyperparameterTuning.SearchBayesianSkopt import SearchBayesianSkopt

hyperparameters_range_dictionary = {
    "epochs": Integer(50,500),
    "num_factors": Integer(10,200),
    "alpha": Real(1.0,10.0),
    "epsilon": Real(1.0,10.0),
    "reg": Real(0.01,3.0)
}
earlystopping_keywargs = {"validation_every_n": 5,
                          "stop_on_validation": True,
                          "evaluator_object": evaluator_validation,
                          "lower_validations_allowed": 4,
                          "validation_metric": "MAP",
                          }

recommender_input_args = SearchInputRecommenderArgs(
    CONSTRUCTOR_POSITIONAL_ARGS = [CombinedMatrix],
    CONSTRUCTOR_KEYWORD_ARGS = {},
    FIT_POSITIONAL_ARGS = [],
    FIT_KEYWORD_ARGS = earlystopping_keywargs
)
recommender_input_args_last_test = SearchInputRecommenderArgs(
    CONSTRUCTOR_POSITIONAL_ARGS = [URM_train_validation],
    CONSTRUCTOR_KEYWORD_ARGS = {},
    FIT_POSITIONAL_ARGS = [],
    FIT_KEYWORD_ARGS = earlystopping_keywargs
)

tuning_class = SearchBayesianSkopt(recommender_class=IALSRecommender,
                                   evaluator_validation=evaluator_validation,
                                   evaluator_test=evaluator_test)

n_cases = 50
n_random_starts = n_cases*0.3
output_folder_path = "logs/"

tuning_class.search(recommender_input_args=recommender_input_args,
                    hyperparameter_search_space=hyperparameters_range_dictionary,
                    metric_to_optimize="MAP",
                    cutoff_to_optimize= 10,
                    n_cases= n_cases,
                    n_random_starts=n_random_starts,
                    output_folder_path=output_folder_path,
                    output_file_name_root=IALSRecommender.RECOMMENDER_NAME,
                    save_model="best"
                    )