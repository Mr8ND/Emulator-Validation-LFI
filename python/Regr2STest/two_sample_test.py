import numpy as np
from classifiers import classifier_dict
from sklearn.base import clone
from tqdm.auto import tqdm


class TwoSampleRegressionTest:

    def __init__(self, classifier_name, model=None, verbose=True):

        '''
        Initializes the TwoSampleRegressionTest class
        :param classifier_name: string with the classifier name. It fetches the respective classifier from the
        available classifiers. If "custom", then a model argument needs to be specified.
        :param model: only used if classifier_name is "custom". Needs to be a sklearn compatible classifier: needs
        to be compatible with the sklean.base.copy function, and have a `fit` and `predict_proba` argument.
        :param verbose: if True, it shows a progress bar in calculating the test statistic.
        '''

        self.classifier_name = classifier_name
        self.verbose = verbose

        if model == 'custom' and model is not None:
            self.model = model
        elif model == 'custom' and model is None:
            raise ValueError('Custom is passed as classifier name, model argument needs to be a sklearn '
                             'compatible classifier.')
        elif classifier_name not in classifier_dict:
            raise ValueError('Classifier name not found. The following are available: '
                             '%s' % ','.join(classifier_dict.keys()))
        else:
            self.model = classifier_dict[classifier_name]

    def _compute_test_statistic(self, x_val, y_val):

        '''
        Fits model and computes regression test statistics
        :param x_val: training covariates
        :param y_val: boolean variables, indicating whether the covariate belong to the first or second sample
        of the two sample tests.
        :return: regression test statistic
        '''

        curr_model = clone(self.model)
        curr_model.fit(X=x_val, y=y_val)
        probabilities = curr_model.predict_proba(x_val)[:, 1]
        return np.average((probabilities - np.average(y_val)) ** 2)

    def ts_test(self, first_sample, second_sample, sample_dimension, n_bootstrap=100):

        '''
        Perform the two sample regression test.
        :param first_sample: First sample for the two sample test
        :param second_sample: Second sample for the two sample test
        :param sample_dimension: Dimensionality of the covariates in both samples
        :param n_bootstrap: Number of bootstrap repetitions in calculating the p-value for the two sample regression
        test
        :return: p-value for the two sample regression test
        '''

        y_val = np.concatenate((np.zeros((first_sample.shape[0],)), np.ones((second_sample.shape[0],))))
        x_val = np.vstack((first_sample, second_sample)).reshape(-1, sample_dimension)

        stat = self._compute_test_statistic(x_val, y_val)
        if self.verbose:
            pbar = tqdm(total=n_bootstrap, desc='Boostrap Calculations')
            null = []
            for _ in range(n_bootstrap):
                null.append(self._compute_test_statistic(x_val=x_val, y_val=np.random.choice(y_val, size=y_val.shape)))
                pbar.update(1)
        else:
            null = [self._compute_test_statistic(x_val=x_val, y_val=np.random.choice(y_val, size=y_val.shape))
                    for _ in range(n_bootstrap)]

        return (np.sum(null >= stat) + 1) / float(n_bootstrap + 1)
