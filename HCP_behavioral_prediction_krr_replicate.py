# %%
import numpy as np
from statsmodels.multivariate.pca import PCA
from sklearn.preprocessing import StandardScaler
import pandas as pd
from argparse import ArgumentParser
from factor_analyzer import Rotator
from junifer.storage import HDF5FeatureStorage
from sklearn.kernel_ridge import KernelRidge
from julearn import run_cross_validation
from julearn.pipeline import PipelineCreator
from julearn.utils import configure_logging
from pprint import pprint
from sklearn.model_selection import RepeatedKFold
from julearn.config import set_config

try:
    configure_logging(level="DEBUG")
except Exception as e:
    print(e)
configure_logging(level="DEBUG")

set_config("disable_x_check", True)
set_config("disable_xtypes_check", True)

# %%
parser = ArgumentParser()
parser.add_argument("--pca_component", type=str, required=True, help="pca component to be selected")
args = parser.parse_args()
pca_component = args.pca_component
# %%
# 80 specific subjects same as the paper
train_subs = pd.read_csv(
    "MMP_HCP_80_subs_componentscoreestimation.txt", header=None
).values.flatten().astype(str)  # 80 subjects

# 753 specific subject for main analysis same as the paper
test_subs = pd.read_csv(
    "MMP_HCP_753_subs.txt", header=None
).values.flatten().astype(str)  # 753 subjects

columns= ['PicSeq_Unadj','CardSort_Unadj','Flanker_Unadj','PMAT24_A_CR','ReadEng_Unadj','PicVocab_Unadj','ProcSpeed_Unadj','VSPLOT_TC','SCPT_SEN','SCPT_SPEC','IWRD_TOT','ListSort_Unadj','MMSE_Score',
                     'PSQI_Score','Endurance_Unadj','Dexterity_Unadj','Strength_Unadj','Odor_Unadj','PainInterf_Tscore','Taste_Unadj','Mars_Final','Emotion_Task_Face_Acc','Language_Task_Math_Avg_Difficulty_Level',
                     'Language_Task_Story_Avg_Difficulty_Level','Relational_Task_Acc','Social_Task_Perc_Random','Social_Task_Perc_TOM','WM_Task_Acc','NEOFAC_A','NEOFAC_O','NEOFAC_C','NEOFAC_N','NEOFAC_E','ER40_CR','ER40ANG','ER40FEAR',
                     'ER40HAP','ER40NOE','ER40SAD','AngAffect_Unadj','AngHostil_Unadj','AngAggr_Unadj','FearAffect_Unadj','FearSomat_Unadj','Sadness_Unadj','LifeSatisf_Unadj','MeanPurp_Unadj','PosAffect_Unadj','Friendship_Unadj',
                     'Loneliness_Unadj','PercHostil_Unadj','PercReject_Unadj','EmotSupp_Unadj','InstruSupp_Unadj','PercStress_Unadj','SelfEff_Unadj','DDisc_AUC_40K','GaitSpeed_Comp']
# Load the dataset
full_df = pd.read_csv("Behavioral_Data", index_col="Subject")[columns]
full_df.index = full_df.index.astype(str)
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

# Ignore ROI1==ROI2
# columns_with_tilde = [col for col in df_features.columns if '~' in col]
# columns_to_keep = ['phase_encoding'] + ['subject'] + [col for col in columns_with_tilde if col.split('~')[0] != col.split('~')[1]]
# df_filtered_features = df_features[columns_to_keep]


# Group by 'subject' and calculate the mean across REST1/LR, REST1/RL, REST2/LR, REST2/RL 
df_features = df_features.groupby('subject').mean()

# merge the dataframe
df_features.index.name= "Subject"
df_data = df_features.join(factors_correct, how="inner")

# %%
# Regression model training
# Step 1: Seperate features (X) and targets (y)
X = [x for x in df_data.columns if "~" in x]
X = [x for x in X if x.split("~")[0] != x.split("~")[1]]
y = f"{pca_component}"
X_types = {
    "features":[".*~.*"],
} 
# %%
# Create the pipeline that will be used to predict the target.
kernel_ridge_model = KernelRidge()
creator = PipelineCreator(problem_type="regression",apply_to="features")
creator.add("zscore")
creator.add(
    kernel_ridge_model, 
    alpha=[        
        0,
        0.00001,
        0.0001,
        0.001,
        0.004,
        0.007,
        0.01,
        0.04,
        0.07,
        0.1,
        0.4,
        0.7,
        1,
        1.5,
        2,
        2.5,
        3,
        3.5,
        4,
        5,
        10,
        15,
        20,], 
    kernel='linear', 
    )
# %%
# Evaluate the model within the cross validation.
rkf = RepeatedKFold(n_splits=10,n_repeats=5,random_state=42)
scores_tuned, model_tuned = run_cross_validation(
    X=X,
    y=y,
    X_types=X_types,
    data=df_data,
    model=creator, 
    return_estimator="all",
    cv=rkf,
    scoring= ['r2_corr','r_corr'],
    n_jobs = -1,
)
print("r2_corr:\n",scores_tuned["test_r2_corr"])
print("r_corr:\n",scores_tuned["test_r_corr"])
print(f"Scores with best hyperparameter: {scores_tuned['test_r_corr'].mean()}")
pprint(model_tuned.best_params_) 

# %%
scores_tuned.to_csv(f"/home/haotsung/HCP_behavioral_prediction/results/replicate/scores_ridge_{pca_component}.csv")