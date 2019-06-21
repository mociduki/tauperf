import os
import numpy as np
from . import log; log = log.getChild(__name__)
import tables
from sklearn import model_selection


def prepare_samples(filenames, labels):
    """
    printing information about the samples
    (training, validation and testing size)
    """
    log.info('Samples:')

    if len(filenames) != len(labels):
        raise ValueError('filenames and labels must have the same length')

    tables_size = []
    n_tables = []
    for f in filenames:
        file = tables.open_file(f)
        sizes = []
        n_table_file = 0
        for obj in file.root.data:
            if isinstance(obj, tables.Table):
                n_table_file += 1
                sizes.append(len(obj))
        tables_size.append(sizes)
        n_tables.append(n_table_file)
        file.close()

    if not all(v == n_tables[0] for v in n_tables):
        raise ValueError('samples have different number of tables')

    log.info('Number of tables for each sample: {}'.format(n_tables))
    # take 20% of the sample for validation and testing
    train_ind, test_ind = model_selection.train_test_split(
        range(3, n_tables[0]), test_size=0.1, random_state=42)
    val_ind, test_ind = np.split(test_ind, [len(test_ind) / 2])

    

    headers = ["Sample", "Training", "Validation", "Testing", "Training tables"]
    sample_size_table = []
    for l, sizes in zip(labels, tables_size):
        train = sum([sizes[i] for i in train_ind])
        test  = sum([sizes[i] for i in test_ind])
        val   = sum([sizes[i] for i in val_ind])
        sample_size_table.append([l, train, val, test, len(train_ind)])

    print 
    from tabulate import tabulate
    print tabulate(sample_size_table, headers=headers, tablefmt='simple')
    print
    
    return train_ind, test_ind, val_ind

def get_X_y(h5_files, data_type, equal_size=False, debug=False):

    data = []
    for h5_file in h5_files:
        t = getattr(h5_file.root.data, data_type)
        data.append(t)
    
    if equal_size:
        log.info('Train and validate with equal size for each mode')
        min_size = min([len(t) for t in data])

    if debug:
        # log.info('Train with very small stat for debugging')
        min_size = min([len(t) for t in data] + [1000])
        
    if equal_size or debug:
        data = [t[0:min_size] for t in data]
    else:
        data = [t.read() for t in data]

    X_data = np.concatenate([d for d in data])
    y_data = np.concatenate([d['truthmode'] for d in data])

    return X_data, y_data


def load_test_data(filenames, test_indices, val_indices, debug=False):

    h5files = [tables.open_file(filename) for filename in filenames]

    X_test = []
    y_test = []
    for index in test_indices:
        X, y = get_X_y(h5files, 'table_{}'.format(index), debug=debug)
        X_test.append(X)
        y_test.append(y)

    X_val = []
    y_val = []
    for index in val_indices:
        X, y = get_X_y(h5files, 'table_{}'.format(index), debug=debug)
        X_val.append(X)
        y_val.append(y)

    X_test = np.concatenate(X_test)
    y_test = np.concatenate(y_test)
    X_val = np.concatenate(X_val)
    y_val = np.concatenate(y_val)

    for f in h5files:
        f.close()
    return X_test, X_val, y_test, y_val
