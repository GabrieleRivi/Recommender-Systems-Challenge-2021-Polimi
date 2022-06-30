from src.Recommenders.GraphBased.P3alphaRecommender import P3alphaRecommender
from Utils.utils import create_URM,create_ICM, create_submission, write_submission


URM = create_URM()
ICM = create_ICM()

recommender = P3alphaRecommender(ICM.T)
recommender.fit(topK=400, alpha=0.1, implicit=True, normalize_similarity=False)

submission = create_submission(recommender)
write_submission(submission,"P3ALPHAItemTransposedSubmisison")