# %%
import pandas as pd
from julearn import run_cross_validation
from julearn.pipeline import PipelineCreator
from julearn.utils import configure_logging
from argparse import ArgumentParser
# Set the logging level to info to see extra information.
configure_logging(level="DEBUG")

# %%
# parser = ArgumentParser()
# parser.add_argument("--pca_component", type=str, required=True, help="pca component to be selected")
# args = parser.parse_args()
# pca_component = args.pca_component

# %%
df = pd.read_csv("HCP_data.csv")
X = df.columns[1:79860].tolist()
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
# Use a Pipeline Creator to create the pipeline that will generate the targets.

target_creator = PipelineCreator(problem_type="transformer", apply_to="behavioral")
target_creator.add("pca", n_components=3)

# Select a pca component
target_creator.add("pick_columns", keep="pca__pca0")

# %%
# Create the pipeline that will be used to predict the target.

creator = PipelineCreator(problem_type="regression")
creator.add("zscore", apply_to="*")
creator.add("generate_target", apply_to="behavioral", transformer=target_creator)
creator.add("linreg", apply_to="features")

# %%
# Evaluate the model within the cross validation.
scores, model = run_cross_validation(
    X=X,
    y=y,
    X_types=X_types,
    data=df,
    model=creator,
    return_estimator="final",
    cv=5,
)

print(scores["test_score"])  # type: ignore