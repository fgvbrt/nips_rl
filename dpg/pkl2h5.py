import os
import h5py
try:
    import cPickle as pickle
except:
    import pickle
import argparse


def read_pkl(fname):
    with open(fname, 'rb') as f:
        return pickle.load(f)


def write_weights(f, weights, name=''):
    for i, w in enumerate(weights):
        name_ = '{}/weight_{}'.format(name, i)
        f.create_dataset(name_, data=w)


def write_h5(fname, weights):
    with h5py.File(fname, 'w') as f:
        write_weights(f, weights[0], 'actor')
        write_weights(f, weights[1], 'critic')


def read_h5(fname):
    with h5py.File(fname, 'r') as f:
        all_weights = []
        for name in ('actor', 'critic'):
            weights = []
            for i in range(len(f[name])):
                name_ = '{}/weight_{}'.format(name, i)
                w = f[name_][:]
                weights.append(w)
            all_weights.append(weights)

    return all_weights


def convert_pkl2h5(fin, fout):
    weights = read_pkl(fin)
    write_h5(fout, weights)


def _parse_args():
    parser = argparse.ArgumentParser(description="Run commands")
    parser.add_argument('src', type=str, help='source file with weights')
    parser.add_argument('--dst', type=str, default=None, help='destination file with weights')
    return parser.parse_args()


def create_dir(dirname):
    try:
        os.makedirs(dirname)
    except:
        pass


if __name__ == '__main__':
    args = _parse_args()

    # get destination path
    dst = args.dst
    if dst is None:
        dst = args.src[:-3] + 'h5'

    # create directory if needed
    save_dir = os.path.dirname(dst)
    create_dir(save_dir)

    # convert file
    convert_pkl2h5(args.src, dst)
