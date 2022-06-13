import pandas as pd
from tqdm import tqdm
import pickle
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV, train_test_split

from src.constants import Chars, DataPaths, Parameters
from src.model import TreeElastic


if __name__ == '__main__':
    chars = Chars()
    paths = DataPaths()
    for tree_file_path in tqdm(paths.processed_data.iterdir()):
        if paths.model_suffix in tree_file_path:
            continue
        tree_portolio = pd.read_pickle(tree_file_path)
        param_grid = {
            'mean_shrinkage': Parameters.mean_shrinkage,
            'ridge_lambda': Parameters.ridge_lambda
        }
        tscv = TimeSeriesSplit(n_splits=Parameters.n_splits)
        tree_model = TreeElastic(k_min=Parameters.k_min, k_max=Parameters.k_max)

        # Select only Returns from the data frame and multiply by the adjusting feature weights
        ret_indexes = tree_portolio.index.get_level_values(Columns.features_col) == Columns.w_returns_col
        tree_portolio = tree_portolio[ret_indexes]

        train_val_portolios, test_portfolios = train_test_split(
            tree_portolio, test_size=Parameters.test_size, shuffle=False)
        cv_tree_model = GridSearchCV(estimator=tree_model, param_grid=param_grid, verbose=1, cv=tscv)
        cv_tree_model.fit(train_val_portolios)
        model_output_name = f"{tree_portolio.with_suffix('').name}{paths.sep}{paths.model_suffix}"
        with open(paths.processed_data / model_output_name, 'wb') as f:
            pickle.dump(cv_tree_model, f)
