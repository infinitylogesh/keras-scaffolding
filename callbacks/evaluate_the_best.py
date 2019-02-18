import numpy as np
import warnings
import keras

class RunForEveryBest(keras.callbacks.Callback):
    """Run a callback for every best model based on the parameters shared.
    # Arguments
        evaluation_callback: Callback to run on every best model.
        monitor: quantity to monitor.
        verbose: verbosity mode, 0 or 1.
        execute_on_best_only: if `execute_on_best_only=True`,
            the latest best model according to
            the quantity monitored will not be overwritten.
        mode: one of {auto, min, max}.
            If `execute_on_best_only=True`, the decision
            to overwrite the current save file is made
            based on either the maximization or the
            minimization of the monitored quantity. For `val_acc`,
            this should be `max`, for `val_loss` this should
            be `min`, etc. In `auto` mode, the direction is
            automatically inferred from the name of the monitored quantity.
        save_weights_only: if True, then only the model's weights will be
            saved (`model.save_weights(filepath)`), else the full model
            is saved (`model.save(filepath)`).
        period: Interval (number of epochs) between checkpoints.
    """

    def __init__(self, evaluation_callback, monitor='val_loss', verbose=0,
                 execute_on_best_only=False,
                 mode='auto', period=1,log_folder=None):
        super(RunForEveryBest, self).__init__()
        self.monitor = monitor
        self.verbose = verbose
        self.evaluation_callback = evaluation_callback
        self.execute_on_best_only = execute_on_best_only
        self.period = period
        self.epochs_since_last_save = 0
        self.log_folder = log_folder

        if mode not in ['auto', 'min', 'max']:
            warnings.warn('EvaluateTheBest mode %s is unknown, '
                          'fallback to auto mode.' % (mode),
                          RuntimeWarning)
            mode = 'auto'

        if mode == 'min':
            self.monitor_op = np.less
            self.best = np.Inf
        elif mode == 'max':
            self.monitor_op = np.greater
            self.best = -np.Inf
        else:
            if 'acc' in self.monitor or self.monitor.startswith('fmeasure'):
                self.monitor_op = np.greater
                self.best = -np.Inf
            else:
                self.monitor_op = np.less
                self.best = np.Inf

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.epochs_since_last_save += 1
        if self.epochs_since_last_save >= self.period:
            self.epochs_since_last_save = 0
            if self.execute_on_best_only:
                current = logs.get(self.monitor)
                if current is None:
                    warnings.warn('Can save best model only with %s available, '
                                  'skipping.' % (self.monitor), RuntimeWarning)
                else:
                    if self.monitor_op(current, self.best):
                        if self.verbose > 0:
                            print('\nEpoch %05d: %s improved from %0.5f to %0.5f,'
                                  ' Executing function'
                                  % (epoch + 1, self.monitor, self.best,
                                     current))
                        self.best = current
                        self.evaluation_callback(self.model,self.log_folder,epoch,self.best)
                    else:
                        if self.verbose > 0:
                            print('\nEpoch %05d: %s did not improve from %0.5f' %
                                  (epoch + 1, self.monitor, self.best))
            else:
                if self.verbose > 0:
                    print('\nEpoch %05d: Executing callback' % (epoch + 1))
                self.evaluation_callback(self.model,self.log_folder,epoch,self.best)
