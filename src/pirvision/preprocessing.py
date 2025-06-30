def handle_missing_values(df):
    return df.dropna()

def detect_outliers_iqr(df, col):
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    return df[(df[col] >= lower) & (df[col] <= upper)]

def normalize_zscore(df):
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    df[df.columns] = scaler.fit_transform(df[df.columns])
    return df