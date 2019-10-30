import os
import tables
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.utils import Sequence
from keras.utils.np_utils import to_categorical

from load import get_X_y
from . import log; log = log.getChild(__name__)

import matplotlib.pyplot as plt

class TrainSequence(Sequence):
    """
    See https://keras.io/utils/
    """
    def __init__(
            self, 
            filenames, 
            train_indices,
            n_chunks, 
            features, 
            reg_features=None,
            equal_size=False, 
            debug=False):
        self._files = filenames
        self._features = features
        self._reg_features = reg_features
        self.n_chunks = n_chunks
        self._equal_size = equal_size
        self._debug = debug
        self._train_indices = train_indices

    def __len__(self):
        return self.n_chunks

    def __getitem__(self, idx):
        h5files = [tables.open_file(f) for f in self._files]
        index = self._train_indices[idx]
        X, y = get_X_y(
            h5files, 
            'table_{0}'.format(index), 
            equal_size=self._equal_size, 
            debug=self._debug)
        X_inputs = [X[feat] for feat in self._features]
        for f in h5files:
            f.close()
        if self._reg_features is None:
            return X_inputs, to_categorical(y, len(self._files))
        else:
            # outputs = [to_categorical(y, len(self._files))]
            # if not isinstance(self._reg_features, (list, tuple)):
            #     self._reg_features = [self._reg_features]
            
            # for feat in self._reg_features:
            #     outputs.append(X[feat])
            return X_inputs, X[self._reg_features[0]]

def fit_model_gen(
        model,
        h5files, 
        features,
        train_indices,
        X_test, y_test, 
        reg_features=None,
        n_chunks=100,
        epochs=10,
        patience=5,
        use_multiprocessing=False,
        workers=1,
        filename='cache/test.h5',
        metrics='categorical_accuracy',
        losses='categorical_crossentropy',
        model_optimizer='rmsprop',
        loss_weights=1,
        overwrite=False,
        no_train=False,
        equal_size=False,
        dev=False,
        debug=False):
    
 
    if not overwrite and os.path.exists(filename):
        log.error('weight file {0} exists, aborting!'.format(filename))
        raise ValueError('overwrite needs to be set to true')

    try:
        log.info('Compile model')
        if not isinstance(losses, (tuple, list)):
            losses = [losses]
        
        if not isinstance(metrics, (tuple, list)):
            metrics = [metrics]

        if not isinstance(loss_weights, (tuple, list)):
            loss_weights = [loss_weights]
        
        if reg_features != None:
            if not isinstance(reg_features, (tuple, list)):
                reg_features = [reg_features]
            

        model.compile(
            optimizer=model_optimizer,
            loss=losses,
            loss_weights=loss_weights,
            metrics=metrics)

        log.info('Create the sequence')
        train_sequence = TrainSequence(
            h5files, 
            train_indices,
            n_chunks, 
            features, 
            reg_features=reg_features, 
            debug=debug)
            
        log.info('Start training ...')

        # make a list of callbacks
        callbacks = []
        if dev:
            callbacks = [
                ModelCheckpoint(filename, monitor='val_loss', verbose=True)
                ]
        else:
            callbacks = [
                EarlyStopping(verbose=True, patience=patience, monitor='val_loss'),
                ModelCheckpoint(filename, monitor='val_loss', verbose=True, save_best_only=True)
                ]

        validation_data = (
            [X_test[feat] for feat in features],
            y_test if reg_features is None else X_test[reg_features[0]])#[X_test[feat] for feat in reg_features])


        logs= model.fit_generator(
            train_sequence,
            len(train_sequence),
            epochs=epochs,
            validation_data=validation_data,
            use_multiprocessing=use_multiprocessing,
            workers=workers,
            callbacks=callbacks)

        #plot accuracy vs epoch
        plt.plot(logs.history['categorical_accuracy'], label='train')
        plt.plot(logs.history['val_categorical_accuracy'], label='valid')
        plt.legend()
        plt.title('')
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.savefig('./plots/imaging/acc_vs_epochs.png')
        plt.clf()

        #plot loss vs epoch
        plt.plot(logs.history['loss'], label='train')
        plt.plot(logs.history['val_loss'], label='valid')
        plt.legend()
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title('')
        plt.savefig('./plots/imaging/loss_vs_epochs.png')
        plt.clf()

#         model.save(filename)
    

    except KeyboardInterrupt:
        print 'Ended early..'

def fit_model(
    model,
    X_train, y_train, 
    X_test, y_test, 
    filename='cache/crackpot.h5',
    loss='binary_crossentropy',
    overwrite=False,
    no_train=False):

    if not overwrite and os.path.exists(filename):
        log.error('weight file {0} exists, aborting!'.format(filename))
        raise ValueError('overwrite needs to be set to true')

    try:
        log.info('Compile model')
        model.compile(
            optimizer='rmsprop',
            loss=loss,
            metrics=['accuracy'])

        log.info('Start training ...')
        model.fit(
            [
                # X_train['tracks'], 
                X_train['s1'], 
                X_train['s2'], 
                X_train['s3'], 
                X_train['s4'], 
                X_train['s5']
                ],
            y_train,
            epochs=100,
            batch_size=128,
            validation_data=(
                [
                    # X_test['tracks'], 
                    X_test['s1'], 
                    X_test['s2'], 
                    X_test['s3'], 
                    X_test['s4'], 
                    X_test['s5']
                    ], y_test),
            callbacks=[
                EarlyStopping(verbose=True, patience=20, monitor='val_loss'),
                ModelCheckpoint(
                    filename, monitor='val_loss', 
                    verbose=True, save_best_only=True)
                ])

        model.save(filename)
    

    except KeyboardInterrupt:
        print 'Ended early..'


def fit_model_single_layer(
    model,
    X_train, y_train, 
    X_test, y_test, 
    filename='cache/crackpot.h5',
    loss='binary_crossentropy',
    overwrite=False,
    no_train=False):

 
    if not overwrite and os.path.exists(filename):
        log.error('weight file {0} exists, aborting!'.format(filename))
        raise ValueError('overwrite needs to be set to false')

    try:
        log.info('Compile model')
        model.compile(
            optimizer='rmsprop',
            loss=loss,
            metrics=['accuracy'])

        log.info('Start training ...')
        log.info(X_train.shape)
        model.fit(
            X_train,
            y_train,
            epochs=100,
            batch_size=128,
            validation_data=(
                X_test, y_test),
            callbacks=[
                EarlyStopping(verbose=True, patience=20, monitor='val_loss'),
                ModelCheckpoint(
                    filename, monitor='val_loss', 
                    verbose=True, save_best_only=True)
                ])

        model.save(filename)
    

    except KeyboardInterrupt:
        log.info('Ended early..')

