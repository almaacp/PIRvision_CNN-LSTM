def load_and_segment_data(paths):
    import pandas as pd
    df_list = [pd.read_csv(p) for p in paths]
    df = pd.concat(df_list, ignore_index=True)
    return df

def stratified_split(X, y, test_size, val_size, seed):
    from sklearn.model_selection import train_test_split
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=test_size, stratify=y, random_state=seed)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=val_size / (1 - test_size), stratify=y_temp, random_state=seed)
    return X_train, X_val, X_test, y_train, y_val, y_test
