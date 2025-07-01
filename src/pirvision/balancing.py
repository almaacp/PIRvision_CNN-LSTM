from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

def apply_smote(X, y):
    return SMOTE(random_state=42).fit_resample(X, y)

def apply_rus(X, y):
    return RandomUnderSampler(random_state=42).fit_resample(X, y)