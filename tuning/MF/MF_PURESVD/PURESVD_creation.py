from src.Recommenders.MatrixFactorization.PureSVDRecommender import PureSVDItemRecommender
from Utils.utils import create_URM,create_ICM, create_submission, write_submission,combine_matrices


URM = create_URM()
ICM = create_ICM()
ICM_combined = combine_matrices(ICM=ICM, URM=URM)

recommender = PureSVDItemRecommender(URM)
recommender.fit(num_factors=23,topK=500)

submission = create_submission(recommender)
write_submission(submission,"PURESVDItemSubmission")