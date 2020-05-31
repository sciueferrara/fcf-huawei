import numpy as np
import pandas as pd
import sys
import random
import os
import argparse
from modules import Server, ServerModel, Client, ClientModel, TripletSampler, ProcessingStrategy, SendStrategy
import utils.utils as utils
from progress.bar import IncrementalBar
import scipy as sp
import scipy.sparse

np.random.seed(43)
random.seed(43)
np.set_printoptions(threshold=sys.maxsize)


def main(args):

    if not os.path.exists('results'):
        os.makedirs('results')

    #exp_type = utils.create_file_prefix(args.positive_fraction, args.with_delta, args.fraction, args.sampler_size)

    #processing_strategy = ProcessingStrategy.MultiProcessing() if args.mp else ProcessingStrategy.SingleProcessing()
    send_strategy = SendStrategy.SendDelta() if args.with_delta else SendStrategy.SendVector()

    for dataset in args.datasets:
        print("Working on", dataset, "dataset")

        if not os.path.exists('results/{}/recs'.format(dataset)):
            os.makedirs('results/{}/recs'.format(dataset))

        df = pd.read_csv('datasets/{}_trainingset.tsv'.format(dataset), sep='\t', names=['user_id', 'item_id', 'rating'])
        df, reverse_dict = utils.convert_unique_idx(df, 'item_id')
        user_size = len(df['user_id'].unique())
        item_size = len(df['item_id'].unique())
        print('Found {} users and {} items'.format(user_size, item_size))
        train_user_lists = utils.create_user_lists(df, user_size, 3)
        train_interactions_size = sum([len(user_list) for user_list in train_user_lists])
        print('{} interactions considered for training'.format(train_interactions_size))

        train_sets = []
        for u in range(user_size):
            s = set(train_user_lists[u])
            train_sets.append([])
            for i in range(item_size):
                train_sets[u].append(1 if i in s else 0)
            train_sets[u] = sp.sparse.csr_matrix(train_sets[u])


        #train_sets_tmp = [{k: 1 for k in train_user_lists[u]} for u in range(user_size)]
        #train_sets = []

        for n_factors in args.n_factors:
            exp_setting_1 = "_F" + str(n_factors)
            for lr in args.lr:
                exp_setting_2 = exp_setting_1 + "_LR" + str(lr)

                # Create server and clients
                server_model = ServerModel(item_size, n_factors)
                server = Server(server_model, lr, args.fraction, args.mp, send_strategy)
                print(train_user_lists[0])
                clients = [Client(u, ClientModel(n_factors), train_sets[u], train_user_lists[u]) for u in range(user_size)]

                print('\n\n')
                # Start training
                bar = IncrementalBar('Training', max=args.n_epochs)
                for i in range(args.n_epochs):
                    bar.next()
                    server.train_model(clients)

                    print(clients[0].model.user_vec)
                    # Evaluation
                    if ((i + 1) % (args.eval_every)) == 0:
                        exp_setting_3 = exp_setting_2 + "_I" + str((i + 1))
                        results = server.predict(clients, max_k=10)
                        with open('results/{}/recs/{}.tsv'.format(dataset, exp_setting_3), 'w') as out:
                            for u in range(len(results)):
                                for e, p in results[u].items():
                                    out.write(str(u) + '\t' + str(reverse_dict[e]) + '\t' + str(p) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets', nargs='+', help='Set the datasets you want to use', required=True)
    parser.add_argument('-F', '--n_factors', nargs='+', help='Set the latent factors you want', type=int, required=True)
    parser.add_argument('-U', '--fraction', help='Set the fraction of clients per round (0 for just one client)', type=float, default=0, required=True)
    parser.add_argument('-lr', '--lr', nargs='+', help='Set the learning rates', type=float, required=True)
    parser.add_argument('-E', '--n_epochs', help='Set the number of epochs', type=int, required=True)
    parser.add_argument('--with_delta', action='store_true', help='Use if you want server to send deltas instead of overwriting item information')
    parser.add_argument('--validation_size', help='Set a validation size, if needed', type=float, default=0)
    parser.add_argument('--eval_every', type=int, default=10)
    parser.add_argument('--mp', action='store_true', help='Use if you want to use multiprocessing (if fraction > 0)')
    parsed_args = parser.parse_args()
    main(parsed_args)
