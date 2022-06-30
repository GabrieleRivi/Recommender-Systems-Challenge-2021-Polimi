"""
@author: Gabriele Rivi
purpose: RecSys Challenge 21/22 @Polimi
"""

from src.Recommenders.GraphBased.P3alphaRecommender import P3alphaRecommender
from src.Recommenders.Base.BaseSimilarityMatrixRecommender import BaseItemSimilarityMatrixRecommender
from Utils.utils import combine_matrices

class P3alphaRecommenderMatricesCombined (BaseItemSimilarityMatrixRecommender):
    """
    Experimenting in feeding a model a different version of matrices as input to compute recommendetations,
    see if improves the quality of it.
    """
    RECOMMENDER_NAME = "P3alphaRecommenderMatricesCombined"

    def __init__(self, URM_train, ICM, verbose = True):
        super(P3alphaRecommenderMatricesCombined, self).__init__(URM_train, verbose = verbose)
        self.ICM = ICM

    def fit(self, topK=100, alpha=1., min_rating=0, implicit=False, normalize_similarity=False):

        ICM_combined = combine_matrices(self.ICM,self.URM_train)
        recommender = P3alphaRecommender(ICM_combined, verbose=self.verbose)
        recommender.fit(topK=topK, alpha=alpha, min_rating=min_rating, implicit=implicit, normalize_similarity=normalize_similarity)
        self.W_sparse = recommender.W_sparse