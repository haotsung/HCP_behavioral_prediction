# %%
import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np
from sklearn.preprocessing import StandardScaler
import pandas as pd
from argparse import ArgumentParser
from junifer.storage import HDF5FeatureStorage
from sklearn.kernel_ridge import KernelRidge
from julearn import run_cross_validation
from julearn.pipeline import PipelineCreator
from julearn.utils import configure_logging
from pprint import pprint
from sklearn.model_selection import RepeatedKFold
from julearn.config import set_config
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

configure_logging(level="INFO")
set_config("disable_x_check", True)
set_config("disable_xtypes_check", True)

# %%
parser = ArgumentParser()
parser.add_argument("--component", type=str, required=True, help="component to be selected")
args = parser.parse_args()
component = args.component
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

scaler = StandardScaler()
imputer = IterativeImputer(max_iter=20, random_state=0)
train_df_imputed = imputer.fit_transform(train_df)
train_df_scaled = scaler.fit_transform(train_df_imputed)
test_df_imputed = imputer.transform(test_df)
test_df_scaled = scaler.transform(test_df_imputed)

# %%
# Define Autoencoder
class Autoencoder(nn.Module):
    def __init__(self, input_dim, encoding_dim):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024,256),
            nn.ReLU(),
            nn.Linear(256,64),
            nn.ReLU(),
            nn.Linear(64, encoding_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, 64),
            nn.ReLU(),
            nn.Linear(64,256),
            nn.ReLU(),
            nn.Linear(256,1024),
            nn.ReLU(),
            nn.Linear(1024, input_dim)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded

# Model parameters
input_dim = full_df.shape[1]
encoding_dim = 3  #Reduced dimensionality
model = Autoencoder(input_dim, encoding_dim)

# Loss and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Training
epochs = 50
batch_size = 16
X_train_tensor = torch.tensor(train_df_scaled, dtype=torch.float32)

for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    _, reconstructed = model(X_train_tensor)
    loss = criterion(reconstructed, X_train_tensor)
    loss.backward()
    optimizer.step()
    #print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

# Encode the behavioral data to lower dimensions
with torch.no_grad():
    X_train_reduced = model.encoder(X_train_tensor).numpy()
    X_test_tensor = torch.tensor(test_df_scaled, dtype=torch.float32)
    X_test_reduced = model.encoder(X_test_tensor).numpy()

test_df_reduced = pd.DataFrame(
        X_test_reduced,
        index=test_df.index,
        columns=["component_1", "component_2", "component_3"],
    )

# %% Feature generation
storage = HDF5FeatureStorage("features.hdf5")
df_features = storage.read_df('BOLD_Schaefer400x17_functional_connectivity')

# Group by 'subject' and calculate the mean across REST1/LR, REST1/RL, REST2/LR, REST2/RL 
df_features = df_features.groupby('subject').mean()

# merge the dataframe
df_features.index.name= "Subject"
df_data = df_features.join(test_df_reduced, how="inner")

# %%
# Regression model training
# Step 1: Seperate features (X) and targets (y)
X = [x for x in df_data.columns if "~" in x]
X = [x for x in X if x.split("~")[0] != x.split("~")[1]]
y = f"{component}"
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
        20,
        50,
        100,
        200,
        500,
        1000], 
    kernel='linear', 
    apply_to= "features"
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
scores_tuned.to_csv(f"/home/haotsung/HCP_behavioral_prediction/results/scores_krr_autoencoder_{component}.csv")