# %%
import numpy as np
from statsmodels.multivariate.pca import PCA
from sklearn.preprocessing import StandardScaler
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LinearRegression
from scipy.stats import pearsonr
from sklearn.metrics import r2_score, make_scorer
from argparse import ArgumentParser
from factor_analyzer import Rotator
from junifer.storage import HDF5FeatureStorage
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import GridSearchCV, KFold, cross_val_score

# %%
parser = ArgumentParser()
parser.add_argument("--pca_component", type=str, required=True, help="pca component to be selected")
args = parser.parse_args()
pca_component = args.pca_component
# %%
# 80 specific subjects same as the paper
train_subs = pd.read_csv(
    "MMP_HCP_80_subs_componentscoreestimation.txt", header=None
).values.flatten()  # 80 subjects

# 753 specific subject for main analysis same as the paper
test_subs = pd.read_csv(
    "MMP_HCP_753_subs.txt", header=None
).values.flatten()  # 753 subjects

columns= ['PicSeq_Unadj','CardSort_Unadj','Flanker_Unadj','PMAT24_A_CR','ReadEng_Unadj','PicVocab_Unadj','ProcSpeed_Unadj','VSPLOT_TC','SCPT_SEN','SCPT_SPEC','IWRD_TOT','ListSort_Unadj','MMSE_Score',
                     'PSQI_Score','Endurance_Unadj','Dexterity_Unadj','Strength_Unadj','Odor_Unadj','PainInterf_Tscore','Taste_Unadj','Mars_Final','Emotion_Task_Face_Acc','Language_Task_Math_Avg_Difficulty_Level',
                     'Language_Task_Story_Avg_Difficulty_Level','Relational_Task_Acc','Social_Task_Perc_Random','Social_Task_Perc_TOM','WM_Task_Acc','NEOFAC_A','NEOFAC_O','NEOFAC_C','NEOFAC_N','NEOFAC_E','ER40_CR','ER40ANG','ER40FEAR',
                     'ER40HAP','ER40NOE','ER40SAD','AngAffect_Unadj','AngHostil_Unadj','AngAggr_Unadj','FearAffect_Unadj','FearSomat_Unadj','Sadness_Unadj','LifeSatisf_Unadj','MeanPurp_Unadj','PosAffect_Unadj','Friendship_Unadj',
                     'Loneliness_Unadj','PercHostil_Unadj','PercReject_Unadj','EmotSupp_Unadj','InstruSupp_Unadj','PercStress_Unadj','SelfEff_Unadj','DDisc_AUC_40K','GaitSpeed_Comp']
# Load the dataset
full_df = pd.read_csv("Behavioral_Data", index_col="Subject")[columns]

test_df = full_df.loc[test_subs]
train_df = full_df.loc[train_subs]
# %% Z-score and PCA
def compute_factors(df_train, df_test):
    scaler = StandardScaler()
    df_trans = scaler.fit_transform(df_train)
    pca = PCA(df_trans, ncomp=3, missing="fill-em")
    loadings = pca.loadings

    rotator = Rotator(method="varimax")
    rotated_loadings = rotator.fit_transform(loadings)

    scaled = scaler.transform(df_test)
    factors = np.dot(scaled, rotated_loadings)
    factors_df = pd.DataFrame(
        factors,
        index=df_test.index,
        columns=["varimax_satisf", "varimax_cog", "varimax_er"],
    )
    return factors_df

factors_correct = compute_factors(train_df, test_df)

# %% Feature generation
storage = HDF5FeatureStorage("features.hdf5")
df_features = storage.read_df('BOLD_Schaefer400x17_functional_connectivity')
df_features_reset = df_features.reset_index() # Reset the index to convert MultiIndex to columns

# Ignore ROI1==ROI2
columns_with_tilde = [col for col in df_features_reset.columns if '~' in col]
columns_to_keep = ['phase_encoding'] + ['subject'] + [col for col in columns_with_tilde if col.split('~')[0] != col.split('~')[1]]
df_filtered_features = df_features_reset[columns_to_keep]

# Select only numeric columns for the mean calculation
numeric_columns = df_filtered_features.select_dtypes(include='number').columns

# Group by 'subject' and calculate the mean for numeric columns across REST1/LR, REST1/RL, REST2/LR, REST2/RL 
averaged_features = df_filtered_features.groupby('subject', as_index=False)[numeric_columns].mean()

# Convert 'subject' column to int in averaged_features
averaged_features['subject'] = averaged_features['subject'].astype(int)

# merge the dataframe
factors_correct.reset_index(inplace=True)
merged_df = pd.merge(averaged_features,factors_correct,left_on='subject',right_on='Subject')
merged_df.drop('Subject', axis=1, inplace=True)
# %%
# Regression model training
# Step 1: Seperate features (X) and targets (y)

X = merged_df.iloc[:,1:79801]
y = merged_df[f"{pca_component}"]

# Define a parameter grid for hyperparameter tuning
param_grid = {
    'alpha': [0.001, 0.01, 0.1, 1, 10],  # Regularization parameter
    'kernel': ['linear', 'rbf'],  # Kernel type
    'gamma': np.logspace(-2, 2, 5)  # Gamma for RBF kernel
}
krr = KernelRidge()

n_repetitions = 60
n_splits = 10

# Function to calculate Pearson correlation
def r_corr_scorer(y_true, y_pred):
    print(y_pred)
    r_corr, _ = pearsonr(y_true, y_pred)
    return r_corr

# Create custom scorers
r2_scorer = make_scorer(r2_score)
r_corr_scorer = make_scorer(r_corr_scorer)

# For storing results
all_folds = []
all_r_corr = []
all_r_squared = []

for replication in range(n_repetitions):
    print(f"In replication:{replication}")
    outer_cv = KFold(n_splits=n_splits, shuffle=True, random_state=replication)
    inner_cv = KFold(n_splits=n_splits, shuffle=True, random_state=replication)

    grid_search = GridSearchCV(estimator=krr, param_grid=param_grid, cv=inner_cv, scoring=r2_scorer)

    # Nested cross-validation for R2
    scores_r2 = cross_val_score(grid_search, X, y, cv=outer_cv, scoring=r2_scorer)
    
    # Nested cross-validation for r_corr
    scores_r_corr = cross_val_score(grid_search, X, y, cv=outer_cv, scoring=r_corr_scorer)

    for fold in range(n_splits):
        all_folds.append(fold + 1)  # Folds go from 1 to 10
        all_r_corr.append(scores_r_corr[fold])
        all_r_squared.append(scores_r2[fold])

result_df = pd.DataFrame(
    {
        "Folds" : all_folds,
        "r_corr": all_r_corr,
        "r2": all_r_squared
    }
)
result_df.to_csv(f"/home/haotsung/HCP_behavioral_prediction/results/replicate/scores_ridge_{pca_component}.csv")