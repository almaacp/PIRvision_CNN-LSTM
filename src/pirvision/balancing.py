def apply_smote(X, y):
    from imblearn.over_sampling import SMOTE
    return SMOTE(random_state=42).fit_resample(X, y)

def apply_rus(X, y):
    from imblearn.under_sampling import RandomUnderSampler
    return RandomUnderSampler(random_state=42).fit_resample(X, y)