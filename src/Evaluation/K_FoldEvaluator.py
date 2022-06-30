from src.Evaluation.Evaluator import Evaluator
from src.Evaluation.Evaluator import EvaluatorHoldout

class K_FoldEvaluator(Evaluator):

    def __init__(self, URM_test_list: list, cutoff_list, min_ratings_per_user=1, exclude_seen=True,
                 diversity_object=None, ignore_items=None, ignore_users_list=None, verbose=True):
        self.evaluator_list = []

        if ignore_users_list is None:
            ignore_users_list = [None] * len(URM_test_list)

        for i in range(len(URM_test_list)):
            self.evaluator_list.append(
                EvaluatorHoldout(URM_test_list=URM_test_list[i],
                                 cutoff_list=cutoff_list,
                                 min_ratings_per_user=min_ratings_per_user,
                                 exclude_seen=exclude_seen,
                                 diversity_object=diversity_object,
                                 ignore_items=ignore_items,
                                 ignore_users=ignore_users_list[i],
                                 verbose=verbose
                                 )
            )

    def evaluateRecommender(self, recommender_list: list):

        results=[]
        for i in range(len(recommender_list)):
            result_df, _ = self.evaluator_list[i].evaluateRecommender(recommender_list[i])

        results.append(result_df.loc[10]["MAP"])

        return results
