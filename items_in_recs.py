import os
import argparse

import matplotlib.pyplot as plt




def count_elements(seq) -> dict:
    hist = {}
    for i in seq:
        hist[i] = hist.get(i, 0) + 1
    return hist



def main(args):

    results = {}

    for dataset in args.datasets:
        results[dataset] = {}
        results[dataset]['single'] = {}
        results[dataset]['uniform'] = {}
        print("Working on", dataset, "dataset")

        #exp_type = utils.create_file_prefix(args.positive_fraction, args.with_delta, args.fraction, args.sampler_size)

        for filename in os.listdir('results/{}/recs'.format(dataset)):
            if filename.startswith('P3Rec') and filename.endswith('I20.0.tsv'):
                with open('results/{}/recs/{}'.format(dataset, filename)) as f:
                    users = {}
                    for r in f:
                        info = r.split("\t")
                        try:
                            users[int(info[0])].append(int(info[1]))
                        except:
                            users[int(info[0])] = []
                            users[int(info[0])].append(int(info[1]))

                    items = []
                    for k, v in users.items():
                        items.extend(v[:10])

                    how_many = sorted([v for _, v in count_elements(items).items()], reverse=True)

                    #how_many = [int(r.split("\t")[1].replace('\n', '')) for r in f]
                    #how_many = sorted(how_many, reverse=True)
                    #if filename.startswith('P3RecPlus'):
                    #    factor = float(filename.split("-")[0].split("P3RecPlus")[1])
                    #    how_many = [x/(factor*10) for x in how_many]

                    if filename.split("-")[2].startswith('Sampsingle'):
                        results[dataset]['single'][filename.split("-")[0]] = how_many
                    elif filename.split("-")[2].startswith('Sampuniform'):
                        results[dataset]['uniform'][filename.split("-")[0]] = how_many

    j = 0
    fig1, f1_axes = plt.subplots(ncols=3, nrows=2, constrained_layout=True)
    for d, d_v in results.items(): #distinguo dataset
        i = 0
        for t, t_v in d_v.items(): #distinguo single e uniform
            f1_axes[i, j].set_title(d+t)
            for f, f_v in t_v.items(): #distinguo i pi
                #lists = sorted(f_v.items())  # sorted by key, return a list of tuples
                #x, y = zip(*lists)  # unpack a list of pairs into two tuples
                f1_axes[i, j].plot(f_v[:1000], label=f)
            f1_axes[i, j].legend()
            i += 1
        j += 1

    plt.savefig('prova.png')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets', nargs='+', help='Set the datasets you want to use', required=True)
    parsed_args = parser.parse_args()
    main(parsed_args)
