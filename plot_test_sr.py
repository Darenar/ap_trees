import pandas as pd
import pickle
import logging
import argparse
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from src.constants import DataPaths, Parameters, Columns


def main(args: argparse.Namespace):
    paths = DataPaths()
    tree_portfolio = pd.read_pickle(paths.processed_data / f"{args.feature_combination}.pkl")

    logging.info("Selecting only returns")
    ret_indexes = tree_portfolio.index.get_level_values(Columns.features_col) == Columns.w_returns_col
    tree_portfolio = tree_portfolio[ret_indexes]

    logging.info('Splitting data')
    _, test_portfolios = train_test_split(
        tree_portfolio, test_size=Parameters.test_size, shuffle=False)

    logging.info('Loading model dump')
    model_output_name = f"{args.feature_combination}{paths.sep}{paths.model_suffix}"
    with open(paths.model_dumps / model_output_name, 'rb') as f:
        ap_tree_model = pickle.load(f)

    sdf = ap_tree_model.predict(test_portfolios)
    sharpe_values = np.mean(sdf, axis=0) / (np.std(sdf, axis=0) + 1e-20)

    sns.lineplot(np.sum(ap_tree_model.betas != 0, 1), sharpe_values)
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plotting test SR')
    parser.add_argument('--feature_combination', default='lme_op_investment')
    arguments = parser.parse_args()
    main(arguments)
