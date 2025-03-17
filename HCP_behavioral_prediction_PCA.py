# %%
import pandas as pd
import numpy as np
from julearn import run_cross_validation
from julearn.pipeline import PipelineCreator
from junifer.storage import HDF5FeatureStorage
from argparse import ArgumentParser
from sklearn.kernel_ridge import KernelRidge
from pprint import pprint
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import GroupKFold
from julearn.model_selection import RepeatedContinuousStratifiedGroupKFold 
from julearn.utils import configure_logging

configure_logging(level="INFO")
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer


from julearn.config import set_config
set_config("disable_x_check", True)
set_config("disable_xtypes_check", True)

# %%
parser = ArgumentParser()
parser.add_argument("--pca_component", type=str, required=True, help="pca component to be selected")
args = parser.parse_args()
pca_component = args.pca_component

# %%
# load family group
groups_df = pd.read_csv("family_id.csv",index_col="Subject")
groups_df.index = groups_df.index.astype(str)

# Feature generation
storage = HDF5FeatureStorage("features.hdf5")
df_features = storage.read_df('BOLD_Schaefer400x17_functional_connectivity')
df_features = df_features.groupby("subject").mean()
columns = [x for x in df_features.columns if "~" in x]
columns = [x for x in columns if x.split("~")[0] != x.split("~")[1]]
df_features = df_features[columns]

# merge the dataframe
df_behavioral = pd.read_csv("Behavioral_Data",index_col="Subject")
df_behavioral.index = df_behavioral.index.astype(str)
behavioral_columns = ['PicSeq_Unadj','CardSort_Unadj','Flanker_Unadj','PMAT24_A_CR','ReadEng_Unadj','PicVocab_Unadj','ProcSpeed_Unadj','VSPLOT_TC','SCPT_SEN','SCPT_SPEC','IWRD_TOT','ListSort_Unadj','MMSE_Score',
                     'PSQI_Score','Endurance_Unadj','Dexterity_Unadj','Strength_Unadj','Odor_Unadj','PainInterf_Tscore','Taste_Unadj','Mars_Final','Emotion_Task_Face_Acc','Language_Task_Math_Avg_Difficulty_Level',
                     'Language_Task_Story_Avg_Difficulty_Level','Relational_Task_Acc','Social_Task_Perc_Random','Social_Task_Perc_TOM','WM_Task_Acc','NEOFAC_A','NEOFAC_O','NEOFAC_C','NEOFAC_N','NEOFAC_E','ER40_CR','ER40ANG','ER40FEAR',
                     'ER40HAP','ER40NOE','ER40SAD','AngAffect_Unadj','AngHostil_Unadj','AngAggr_Unadj','FearAffect_Unadj','FearSomat_Unadj','Sadness_Unadj','LifeSatisf_Unadj','MeanPurp_Unadj','PosAffect_Unadj','Friendship_Unadj',
                     'Loneliness_Unadj','PercHostil_Unadj','PercReject_Unadj','EmotSupp_Unadj','InstruSupp_Unadj','PercStress_Unadj','SelfEff_Unadj','DDisc_AUC_40K','GaitSpeed_Comp']
df_behavioral = df_behavioral[behavioral_columns]
df_features.index.name = "Subject"
df_combined = df_features.join(df_behavioral,how="inner")
df_data = df_combined.join(groups_df,how='inner')
df_data = df_data.reset_index()


# %%
X = df_data.columns.tolist()
y = "__generated__"

# Define feature types
X_types = {
    "features": [".*~.*"],
    "behavioral": ['PicSeq_Unadj','CardSort_Unadj','Flanker_Unadj','PMAT24_A_CR','ReadEng_Unadj','PicVocab_Unadj','ProcSpeed_Unadj','VSPLOT_TC','SCPT_SEN','SCPT_SPEC','IWRD_TOT','ListSort_Unadj','MMSE_Score',
                     'PSQI_Score','Endurance_Unadj','Dexterity_Unadj','Strength_Unadj','Odor_Unadj','PainInterf_Tscore','Taste_Unadj','Mars_Final','Emotion_Task_Face_Acc','Language_Task_Math_Avg_Difficulty_Level',
                     'Language_Task_Story_Avg_Difficulty_Level','Relational_Task_Acc','Social_Task_Perc_Random','Social_Task_Perc_TOM','WM_Task_Acc','NEOFAC_A','NEOFAC_O','NEOFAC_C','NEOFAC_N','NEOFAC_E','ER40_CR','ER40ANG','ER40FEAR',
                     'ER40HAP','ER40NOE','ER40SAD','AngAffect_Unadj','AngHostil_Unadj','AngAggr_Unadj','FearAffect_Unadj','FearSomat_Unadj','Sadness_Unadj','LifeSatisf_Unadj','MeanPurp_Unadj','PosAffect_Unadj','Friendship_Unadj',
                     'Loneliness_Unadj','PercHostil_Unadj','PercReject_Unadj','EmotSupp_Unadj','InstruSupp_Unadj','PercStress_Unadj','SelfEff_Unadj','DDisc_AUC_40K','GaitSpeed_Comp'],
}


# %%
target_creator = PipelineCreator(problem_type="transformer", apply_to="behavioral")
imputer = IterativeImputer(max_iter=20, random_state=0)
target_creator.add(imputer)
target_creator.add("zscore")
target_creator.add("pca",n_components=3)
# Select a pca component
target_creator.add("pick_columns", keep=pca_component)

# %%
# Create the pipeline that will be used to predict the target.
#from sklearn.kernel_ridge import KernelRidge
kernel_ridge_model = KernelRidge()
creator = PipelineCreator(problem_type="regression",apply_to="features")
creator.add("zscore")
creator.add("generate_target", apply_to="behavioral", transformer=target_creator)
creator.add(
    kernel_ridge_model, 
    alpha=[        
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
        20,
        50,],
    kernel='linear', 
    apply_to= "features"
    )

# %% Define a repeated group k-fold generator.
#rkf = RepeatedKFold(n_splits=10,n_repeats=1,random_state=42)
#cv = RepeatedContinuousStratifiedGroupKFold(n_bins=2, method='quantile', n_splits=3, n_repeats=1, random_state=42)
gkf = GroupKFold(n_splits=3)
# %%
# Evaluate the model within the cross validation.
scores_tuned, model_tuned = run_cross_validation(
    X=X,
    y=y,
    X_types=X_types,
    data=df_data,
    model=creator, 
    return_estimator="all",
    groups="Family_ID",
    cv=gkf,
    scoring= ['r2_corr','r_corr'],
    n_jobs = -1,
)
print("r2_corr:\n",scores_tuned["test_r2_corr"])
print("r_corr:\n",scores_tuned["test_r_corr"])
print(f"Scores with best hyperparameter: {scores_tuned['test_r_corr'].mean()}")
pprint(model_tuned.best_params_)

# %%
scores_tuned.to_csv(f"/home/haotsung/HCP_behavioral_prediction/results/pca_results/scores_krr_{pca_component}_0315.csv")