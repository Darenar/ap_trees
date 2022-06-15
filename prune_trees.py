import pandas as pd
from tqdm import tqdm
import pickle
import logging
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV, train_test_split

from src.constants import Chars, DataPaths, Parameters, Columns
from src.model import TreeElastic


if __name__ == '__main__':
    chars = Chars()
    paths = DataPaths()
    for tree_file_path in tqdm(paths.processed_data.iterdir()):
        if tree_file_path.suffix != '.pkl':
            continue
        logging.info(f"Loading {str(tree_file_path)}")
        tree_portfolio = pd.read_pickle(tree_file_path)

        logging.info("Selecting only returns")
        ret_indexes = tree_portfolio.index.get_level_values(Columns.features_col) == Columns.w_returns_col
        tree_portfolio = tree_portfolio[ret_indexes]

        logging.info('Splitting data')
        train_val_portfolios, test_portfolios = train_test_split(
            tree_portfolio, test_size=Parameters.test_size, shuffle=False)

        param_grid = {
            'mean_shrinkage': Parameters.mean_shrinkage,
            'ridge_lambda': Parameters.ridge_lambda
        }
        tscv = TimeSeriesSplit(n_splits=Parameters.n_splits)
        tree_model = TreeElastic(k_min=Parameters.k_min, k_max=Parameters.k_max)
        cv_search = GridSearchCV(estimator=tree_model, param_grid=param_grid, verbose=3, cv=tscv)
        cv_search.fit(train_val_portfolios)

        logging.info('Train overall model with tuned parameters')
        overall_model = TreeElastic(k_min=Parameters.k_min, k_max=Parameters.k_max,
                                    ridge_lambda=cv_search.best_params_['ridge_lambda'],
                                    mean_shrinkage=cv_search.best_params_['mean_shrinkage']
                                    )
        overall_model.fit(train_val_portfolios)

        model_output_name = f"{tree_file_path.with_suffix('').name}{paths.sep}{paths.model_suffix}"
        with open(paths.model_dumps / model_output_name, 'wb') as f:
            pickle.dump(overall_model, f)



        train_val_portolios, test_portfolios = train_test_split(
            tree_portolio, test_size=Parameters.test_size, shuffle=False)
        cv_tree_model = GridSearchCV(estimator=tree_model, param_grid=param_grid, verbose=1, cv=tscv)
        cv_tree_model.fit(train_val_portolios)
        model_output_name = f"{tree_portolio.with_suffix('').name}{paths.sep}{paths.model_suffix}"
        with open(paths.processed_data / model_output_name, 'wb') as f:
            pickle.dump(cv_tree_model, f)
