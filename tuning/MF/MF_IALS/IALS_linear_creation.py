from src.Recommenders.MatrixFactorization.IALSRecommenderLinear import IALSRecommender
from Utils.utils import create_URM,create_ICM, create_submission, write_submission,combine_matrices


URM = create_URM()
ICM = create_ICM()
ICM_combined = combine_matrices(ICM=ICM, URM=URM)

recommender = IALSRecommender(URM)
recommender.fit(epochs= 10, num_factors= 42, alpha=0.7617528864750021, reg=8.926401306541349)

submission = create_submission(recommender)
write_submission(submission,"IALS_retuned")