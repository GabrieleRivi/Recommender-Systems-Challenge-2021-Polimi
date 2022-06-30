import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import scipy as sp
import scipy.sparse as sps

def create_URM():
    users = load_data_interactions()

    URM = sps.coo_matrix((users["data"].values,
                          (users["user"].values, users["item"].values)))
    return URM.tocsr()

def create_ICM():
    channels = load_data_channels()
    genres = load_data_genres()
    subgenres = load_data_subgenres()

    channels = sps.coo_matrix((channels["data_ch"].values,
                               (channels["items"].values, channels["channel"].values)))
    genres = sps.coo_matrix((genres["data_gr"].values,
                             (genres["items"].values, genres["genre"].values)))
    subgenres = sps.coo_matrix((subgenres["data_sgr"].values,
                                (subgenres["items"].values, subgenres["subgenre"].values)))

    ICM = sps.hstack([channels, genres, subgenres])
    return ICM.tocsr()

def create_ICM_with_events():
    channels = load_data_channels()
    genres = load_data_genres()
    subgenres = load_data_subgenres()
    events = load_data_events()

    channels = sps.coo_matrix((channels["data_ch"].values,
                               (channels["items"].values, channels["channel"].values)))
    genres = sps.coo_matrix((genres["data_gr"].values,
                             (genres["items"].values, genres["genre"].values)))
    subgenres = sps.coo_matrix((subgenres["data_sgr"].values,
                                (subgenres["items"].values, subgenres["subgenre"].values)))
    events = sps.coo_matrix((events["data_ev"].values,
                                (events["items"].values, events["episode"].values)))

    ICM = sps.hstack([channels, genres, subgenres,events])
    return ICM.tocsr()

def combine_matrices(ICM: sps.csr_matrix ,URM: sps.csr_matrix):
    return sps.vstack([URM,ICM.T], format='csr')


def load_data_interactions():
    return pd.read_csv("/Users/gabriele/PycharmProjects/RecSys/data/urm-data/data_train.csv",
                       sep=",",
                       names=["user","item","data"],
                       header=0)

def load_data_channels():
    return pd.read_csv("/Users/gabriele/PycharmProjects/RecSys/data/icm-data/data_ICM_channel.csv",
                       sep=",",
                       names=["items","channel","data_ch"],
                       header=0)

def load_data_genres():
    return pd.read_csv("/Users/gabriele/PycharmProjects/RecSys/data/icm-data/data_ICM_genre.csv",
                       sep=",",
                       names=["items","genre","data_gr"],
                       header=0)

def load_data_subgenres():
    return pd.read_csv("/Users/gabriele/PycharmProjects/RecSys/data/icm-data/data_ICM_subgenre.csv",
                       sep=",",
                       names=["items","subgenre","data_sgr"],
                       header=0)
def load_data_events():
    return pd.read_csv("/Users/gabriele/PycharmProjects/RecSys/data/icm-data/data_ICM_event.csv",
                       sep=",",
                       names=["items","episode","data_ev"],
                       header=0)

def load_users_for_submission():
    return pd.read_csv("/Users/gabriele/PycharmProjects/RecSys/data/utils-csv/data_target_users_test.csv",
                        names = ['userID'],
                        header = 0)

def create_submission(recommender):
    users_to_recommend = load_users_for_submission()
    submission =[]
    for user in users_to_recommend["userID"].values:
        submission.append((user,recommender.recommend(user,10)))

    return submission

def write_submission(submission,name):
    with open("./"+name+".csv", "w") as f:
        f.write("user_id,item_list\n")
        for user_id, items in submission:
            f.write(f"{user_id},{' '.join([str(item) for item in items])}\n")



