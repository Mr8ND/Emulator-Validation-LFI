import sys
sys.path.append("..")
from Regr2STest.two_sample_test import TwoSampleRegressionTest
import numpy as np


def test__p_value_two_different_samples():

    model = TwoSampleRegressionTest(classifier_name='RF100')

    np.random.seed(7)
    first_sample = np.random.normal(0, 1, 500)
    second_sample = np.random.normal(0, 1, 500)

    p_val = model.ts_test(first_sample=first_sample, second_sample=second_sample, sample_dimension=1, n_bootstrap=100)
    assert p_val > 0.01

    np.random.seed(7)
    first_sample = np.random.normal(0, 1, 500)
    second_sample = np.random.uniform(-1, 0, 500)

    p_val = model.ts_test(first_sample=first_sample, second_sample=second_sample, sample_dimension=1, n_bootstrap=100)
    assert p_val <= 0.05
