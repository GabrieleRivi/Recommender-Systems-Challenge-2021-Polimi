from src.Recommenders.GraphBased.P3alphaRecommender import P3alphaRecommender
from Utils.utils import create_URM,create_ICM, create_submission, write_submission


URM = create_URM()
ICM = create_ICM()

recommender = P3alphaRecommender(URM)
recommender.fit(topK=46,alpha=0.7723235971713814, implicit=True, normalize_similarity=False)

submission = create_submission(recommender)
write_submission(submission,"P3AlphaSubmission")