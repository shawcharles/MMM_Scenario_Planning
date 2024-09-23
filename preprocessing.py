import pandas as pd

def split_data(input_data_processed: pd.DataFrame, config: dict):
    """
    Splits the data into training and testing datasets based on a configuration.
    """
    target_column = config.get('target_col', 'subscribers')
    split_ratio = config.get("train_test_ratio", 0.8)
    
    if split_ratio < 0 or split_ratio > 1:
        raise ValueError("split_ratio must be between 0 and 1")
    
    split_point = int(split_ratio * len(input_data_processed))
    
    X = input_data_processed.drop(columns=[target_column], errors='ignore')
    y = input_data_processed[target_column]
    
    if split_ratio == 1.0:
        X_train = X
        y_train = y
        X_test = pd.DataFrame(columns=X.columns)
        y_test = pd.Series(dtype=y.dtype)
    else:
        X_train = input_data_processed.iloc[:split_point].drop(columns=[target_column], errors='ignore')
        y_train = input_data_processed.iloc[:split_point][target_column]
        X_test = input_data_processed.iloc[split_point:].drop(columns=[target_column], errors='ignore')
        y_test = input_data_processed.iloc[split_point:][target_column]
    
    return X, y, X_train, y_train, X_test, y_test
