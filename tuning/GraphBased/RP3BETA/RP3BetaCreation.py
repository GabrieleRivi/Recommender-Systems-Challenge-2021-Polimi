from src.Recommenders.GraphBased.RP3betaRecommender import RP3betaRecommender
from Utils.utils import create_URM,create_ICM, create_submission, write_submission


URM = create_URM()
ICM = create_ICM()

recommender = RP3betaRecommender(URM)
recommender.fit(topK=67,alpha=1.0, beta=0.6676517342477193, implicit=True, normalize_similarity=True)

submission = create_submission(recommender)
write_submission(submission,"RP3BetaSubmission")