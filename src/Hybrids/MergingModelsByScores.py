from src.Recommenders.Base.BaseSimilarityMatrixRecommender import BaseItemSimilarityMatrixRecommender
from src.Recommenders.Recommender_utils import check_matrix


class MergeThreeModelsByScores(BaseItemSimilarityMatrixRecommender):
    """ ItemKNNScoresHybridThreeRecommender
    Hybrid of three prediction scores R = R1*alpha + R2*beta + R3*gamma
    """

    RECOMMENDER_NAME = "MergeThreeModelsByScore"

    def __init__(self, URM_train, recommenders: list):
        super(MergeThreeModelsByScores, self).__init__(URM_train)

        self.URM_train = check_matrix(URM_train.copy(), 'csr')
        self.Recommender_1 = recommenders[0]
        self.Recommender_2 = recommenders[1]
        self.Recommender_3 = recommenders[2]
        self.W_sparse = None

    def fit(self, alpha=0.5, beta=0.5, gamma=0.5):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def _compute_item_score(self, user_id_array, items_to_compute=None):
        item_weights_1 = self.Recommender_1._compute_item_score(user_id_array, items_to_compute)
        item_weights_2 = self.Recommender_2._compute_item_score(user_id_array, items_to_compute)
        item_weights_3 = self.Recommender_3._compute_item_score(user_id_array, items_to_compute)

        item_weights = item_weights_1 * self.alpha + item_weights_2 * self.beta + item_weights_3 * self.gamma

        return item_weights


class MergeTwoModelsByScores(BaseItemSimilarityMatrixRecommender):
    """ ItemKNNScoresHybridThreeRecommender
    Hybrid of two prediction scores R = R1*alpha + R2*beta
    """

    RECOMMENDER_NAME = "MergeTwoModelsByScore"

    def __init__(self, URM_train, recommenders: list):
        super(MergeTwoModelsByScores, self).__init__(URM_train)

        self.URM_train = check_matrix(URM_train.copy(), 'csr')
        self.Recommender_1 = recommenders[0]
        self.Recommender_2 = recommenders[1]
        self.W_sparse = None

    def fit(self, alpha=0.5, beta=0.5):
        self.alpha = alpha
        self.beta = beta

    def _compute_item_score(self, user_id_array, items_to_compute=None):
        item_weights_1 = self.Recommender_1._compute_item_score(user_id_array, items_to_compute)
        item_weights_2 = self.Recommender_2._compute_item_score(user_id_array, items_to_compute)

        item_weights = item_weights_1 * self.alpha + item_weights_2 * self.beta

        return item_weights


