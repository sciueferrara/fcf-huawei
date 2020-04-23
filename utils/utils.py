import pickle
import numpy as np
from typing import Tuple


def save_obj(obj, name):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(name):
    with open(name, 'rb') as f:
        return pickle.load(f)


def process_results(results):
    new_results = {}
    for k, v in results.items():
        clients_precs = []
        clients_recs = []
        for cid, c in v.items():
            clients_precs.append(c[0])
            clients_recs.append(c[1])
        prnp = np.array(clients_precs)
        prec = np.mean(prnp, axis=0)
        recnp = np.array(clients_recs)
        rec = np.mean(recnp, axis=0)
        new_results[k] = prec
    return new_results


def convert_unique_idx(df, column_name):
    column_dict = {x: i for i, x in enumerate(df[column_name].unique())}
    df[column_name] = df[column_name].apply(column_dict.get)
    df[column_name] = df[column_name].astype('int')
    assert df[column_name].min() == 0
    assert df[column_name].max() == len(column_dict) - 1
    return df


def create_user_lists(df, user_size):
    user_list = [dict() for _ in range(user_size)]
    for row in df.itertuples():
        user_list[row.user_id][row.item_id] = row.utc
    return user_list


def split_train_test(user_list, test_size=0.2, validation_size=0) -> Tuple[list, list, list]:
    train_user_list = [None] * len(user_list)
    validation_user_list = [None] * len(user_list)
    test_user_list = [None] * len(user_list)
    for user, item_dict in enumerate(user_list):
        item = sorted(item_dict.items(), key=lambda x: x[1], reverse=True)

        latest_item = item[:int(len(item)*test_size)]
        assert max(item_dict.values()) == latest_item[0][1]
        test_item = set(map(lambda x: x[0], latest_item))
        assert len(test_item) > 0, "No test item for user %d" % user
        test_user_list[user] = test_item

        latest_item = item[int(len(item)*test_size):int((len(item)-len(test_item))*validation_size)+int(len(item)*test_size)]
        validation_item = set(map(lambda x: x[0], latest_item))
        if validation_size > 0:
            assert len(validation_item) > 0, "No validation item for user %d" % user
        validation_user_list[user] = validation_item

        latest_item = item[int((len(item)-len(test_item))*validation_size)+int(len(item)*test_size):]
        train_item = set(map(lambda x: x[0], latest_item))
        train_user_list[user] = train_item

    return train_user_list, validation_user_list, test_user_list


def create_file_prefix(positive_fraction, with_delta, fraction, sampler_size):
    if positive_fraction:
        string = 'P3RecPlus' + str(positive_fraction)
    else:
        if with_delta:
            string = 'P3RecMinusDelta'
        else:
            string = 'P3RecMinus'
    string += '-Frac' + str(fraction) + '-Samp' + sampler_size
    return string
