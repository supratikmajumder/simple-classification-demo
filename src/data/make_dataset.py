# -*- coding: utf-8 -*-
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

import numpy as np
import os

def main(output_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')

    features, labels = make_classification(n_samples=1000000, random_state=13)
    features_train, features_test, labels_train, labels_test = train_test_split(
        features, labels, test_size=0.15, train_size=0.85, random_state=7)

    np.savetxt(os.path.join(output_filepath, "train_features.csv"), features_train)
    np.savetxt(os.path.join(output_filepath, "train_labels.csv"), labels_train)
    np.savetxt(os.path.join(output_filepath, "test_features.csv"), features_test)
    np.savetxt(os.path.join(output_filepath, "test_labels.csv"), labels_test)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    output_filepath = "data/processed"
    main(output_filepath)
