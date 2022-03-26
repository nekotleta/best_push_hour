import pandas as pd
import numpy as np
from surprise import SVD
from surprise import Dataset, Reader
import pickle
reader = Reader(rating_scale=(0, 1))

import warnings
warnings.filterwarnings('ignore')

def load_data(filename='data/raw_push_stat_2022_01_13_.gz'):
    """
    Load data from file
    :param filename: str - path to the file with input data in gz format
    :return: pandas.DataFrame
    """
    df = pd.read_csv(filename, compression='gzip', header=0, sep=';',
                     dtype={
                         'user_id': str,
                         'content_id': str,
                         'push_opened': np.int8,
                         'content_type': str
                     },
                     parse_dates=['push_time', 'push_opened_time', 'create_at'])
    df = df.drop_duplicates()
    #generate some data
    df["push_day"] = df["push_time"].dt.date
    df['hour'] = np.where(df.push_opened == 1, df["push_opened_time"].dt.hour, df["push_time"].dt.hour)
    return df

def calc_metric(y_true, y_pred):
    """
    Calculate roc_auc metric for checking a quality of model
    :param y_true: list - true values
    :param y_pred: list of Prediction, for example
           Prediction(uid='0001NZ0', iid=2, r_ui=0, est=0.05276711328890782, details={'was_impossible': False}
    :return: float - roc auc score
    """
    from sklearn.metrics import roc_auc_score
    from collections import defaultdict
    # First map the predictions to each user.
    top_n = defaultdict(list)
    for uid, iid, true_r, est, _ in y_pred:
        top_n[(uid, iid)] = est
    y_pred = list(top_n.values())
    auc = roc_auc_score(y_true, y_pred)
    return auc

def train_svd(df):
    """
    Train surprise.SVD
    :param df: pd.DataFrame - train data
    :return: surprise.SVD - training model
    """
    reader = Reader(rating_scale=(0, 1)) 
    #Drop not unique values - leaving only the maximum
    train = df.groupby(['user_id', 'hour'], as_index=False).push_opened.max()
    train_load = Dataset.load_from_df(train, reader).build_full_trainset()
    algo = SVD(random_state = 23)
    algo.fit(train_load)
    return algo

def train(df, check_quality=False):
    """
    Main method of training model
    :param df: pd.DataFrame - input data
    :param check_quality: bool - if True the train will be with validation and then train for all data,
                                    if Else the train will be on all data
    :return: model surprise.SVD
    """
    if check_quality:
        #train with validation
        #validation - last allowed data
        max_allowed_data = df.push_day.max()
        train = df[df.push_day < max_allowed_data]\
                    [['user_id', 'hour', 'push_opened']].drop_duplicates()
        train_algo = train_svd(train)

        valid = df[df.push_day == max_allowed_data]\
                    [['user_id', 'hour', 'push_opened']].drop_duplicates()
        valid = valid.groupby(['user_id', 'hour'], as_index=False).push_opened.max()

        test_load = Dataset.load_from_df(valid, reader)
        testset = [test_load.df.loc[i].to_list() for i in range(len(test_load.df))]
        pred = train_algo.test(testset)
        auc = calc_metric(valid.push_opened, pred)

        if auc <= 0.5:
            print(f"The model is of dubious quality {auc}")
            return 0
           
    #train without validation
    algo = train_svd(df)
    return algo

def get_top_k_recs(svd, user):
    """
    Predict best hour for one user
    :param svd: surprise.SVD - model
    :param user: str - user_id
    :return: int - best hour
    """
    list_hours = np.arange(0, 24)
    res = []
    for h in list_hours:
        res.append(svd.predict(user, h).est)
    return np.array(res).argmax()

def get_predicts(algo, df):
    """
    Main method for get predicts
    :param algo: surprise.SVD - model
    :param df: pandsd.DataFrame - input data, must have column 'user_id'
    :return: dict - user: best-hour
    """
    from collections import defaultdict
    from tqdm.notebook import tqdm
    topn = defaultdict(list)
    for u in tqdm(df.user_id.unique()):
        topn[u] = get_top_k_recs(algo, u)
    return topn

def get_predicts_parallel(svd, df):
    """
    Method for parallel getting predicts
    :param svd: surprise.SVD - model
    :param df: pandsd.DataFrame - input data, must have column 'user_id'
    :return: dict - user: best-hour
    """
    from joblib import Parallel, delayed
    #n_jobs == -1 is too dangerous
    result = Parallel(n_jobs=8)(delayed(get_top_k_recs)(svd, u) for u in tqdm(df.user_id.unique()))
    return dict(zip(df.user_id.unique(), result))
    
if __name__ == "__main__":
    filename = "data/raw_push_stat_2022_01_13_.gz"
    print("Loading data...")
    df = load_data(filename)
    print("Start train model...")
    model = train(df, check_quality=True)
    print("Predict data...")
    result = get_predicts(model, df)
    res = pd.DataFrame.from_dict(result, orient='index').reset_index()
    res.columns = ["user_id", "best_push_hour"]
    res = res[(res.user_id != 0) & (res.best_push_hour >= 0)]
    #check for correctness in users
    assert res.shape[0] == df.user_id.nunique(), "Input user_ids != predicted user_ids"
    
    #save result and model
    res.to_csv("data/output_file.csv", index=False)
    with open(f"data/svd_{df.push_day.max()}.pkl", "wb") as f:
        pickle.dump(model, f)