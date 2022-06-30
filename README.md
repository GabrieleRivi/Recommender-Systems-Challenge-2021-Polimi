# Recommender-Systems-Challenge-2021-Polimi
 
Provide an accurate Recommender System in the domain of TV programs, with data provided about interactions between users and TV shows, and shows' features.
The main goal of the competition is to discover which items (TV shows) a user will interact with, recommending a list of 10 potentially relevant items for each user. 
MAP@10 was used for evaluation.
Each TV show can be composed of several episodes (for instance, episode 5, season 3). The goal of the recommender system is not to recommend a specific episode, 
but to recommend the TV show.
The datasets include around 6.2M interactions, 13k users, 18k items (TV shows) and four feature categories: 8 genres, 
213 channels, 113 subgenres and 358k events (episode ids).

Leaderboard final position - 25/72

## Final Model

The final model used for the submission, scoring 0.48259 against 0.50966 of the first position, was an hybrid obtained merging two models, RPR Beta and Slim-ElasticNet, then making a linear combination of the scores of this hybrid model with an IALS matrix factorization Recommender.
The final model can be found here ([RPR_hybrid_SLIMElasticnet_Ials](/tuning/Hybrids/RP3HYBRIDSLIME_IALS.ipynb))
## Methodology

Most of the models come from the Course-Repository ([src](/src)), and our aim was to find the best scoring model by tuning and combining them with the
techniques seen during the course. For the tuning, along with the Bayesian Search for the hyperparameters, a K-Fold tuner was implemented.
