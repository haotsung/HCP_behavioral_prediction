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
from torch.utils.data import DataLoader, TensorDataset

from sklearn.base import BaseEstimator, TransformerMixin
from julearn import transformers
configure_logging(level="INFO")
set_config("disable_x_check", True)
set_config("disable_xtypes_check", True)


# %%
# Define Autoencoder
class AutoencoderTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, input_dim, encoding_dim,epochs=100, batch_size=10, learning_rate=0.01, weight_decay=1e-4):
        self.input_dim = input_dim
        self.encoding_dim = encoding_dim
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.model = None
        self.feature_names_out_ = [f"component_{i+1}" for i in range(self.encoding_dim)] # Define column names

    class Autoencoder(nn.Module):
        def __init__(self, input_dim, encoding_dim):
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, 32),
                nn.ReLU(),
                nn.Linear(32,16),
                nn.ReLU(),
                nn.Linear(16,8),
                nn.ReLU(),
                nn.Linear(8, encoding_dim)
            )
            self.decoder = nn.Sequential(
                nn.Linear(encoding_dim, 8),
                nn.ReLU(),
                nn.Linear(8, 16),
                nn.ReLU(),
                nn.Linear(16, 32),
                nn.ReLU(),
                nn.Linear(32, input_dim)
            )

        def forward(self, x):
            encoded = self.encoder(x)
            decoded = self.decoder(encoded)
            return encoded, decoded

    def fit(self, X , y=None): 
        # Train the autoencoder
        X_train_tensor = torch.tensor(X.to_numpy(), dtype=torch.float32)
        train_loader = DataLoader(TensorDataset(X_train_tensor), batch_size=self.batch_size, shuffle=True)

        self.model = self.Autoencoder(self.input_dim, self.encoding_dim)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(),lr=self.learning_rate,weight_decay=self.weight_decay)

        # Training Loop


        for epoch in range(self.epochs):
            self.model.train()
            epoch_loss = 0 

            for batch in train_loader:
                batch_data = batch[0]
                optimizer.zero_grad()
                _, reconstructed = self.model(batch_data)
                loss = criterion(reconstructed, batch_data)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            #print(f"Epoch {epoch+1}/{self.epochs}, Loss: {loss.item():.4f}")
        
        return self
    def transform(self, X):
        # Encode the behavioral data to lower dimensions

        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.tensor(X.to_numpy(), dtype=torch.float32)
            encoded = self.model.encoder(X_tensor).numpy()
        df_encoded = pd.DataFrame(
            encoded,
            index = X.index,
            columns = self.feature_names_out_
        )

        return df_encoded

    def get_feature_names_out(self, input_features=None):
        return self.feature_names_out_
    
transformers.register_transformer("autoencoder", AutoencoderTransformer)
# %%
parser = ArgumentParser()
parser.add_argument("--component", type=str, required=True, help="component to be selected")
args = parser.parse_args()
component = args.component

# %%
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
df_data = df_features.join(df_behavioral,how="inner")

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
#autoencoder_transformer = AutoencoderTransformer(input_dim=len(behavioral_columns), encoding_dim=3)

target_creator = PipelineCreator(problem_type="transformer", apply_to="behavioral")
imputer = IterativeImputer(max_iter=20, random_state=0)
target_creator.add(imputer)
target_creator.add("zscore")
target_creator.add("autoencoder",input_dim=len(behavioral_columns), encoding_dim=3,epochs=100, batch_size=10, learning_rate=0.01, weight_decay=1e-4)
# Select a pca component
target_creator.add("pick_columns", keep=component)

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
        1e-4,
        1e-3,
        1e-2,
        1e-1,
        1e0,       
        1e1,
        1e2,
        1e3,
        1e4,
        1e5,
        1e6,
        1e7,
        1e8,
        1e9,
        1e10,
        ],
    kernel='linear', 
    apply_to= "features"
    )

# %%
# Evaluate the model within the cross validation.
rkf = RepeatedKFold(n_splits=3,n_repeats=10,random_state=42)
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
scores_tuned.to_csv(f"/home/haotsung/HCP_behavioral_prediction/results/scores_krr_autoencoder_{component}_CV.csv")