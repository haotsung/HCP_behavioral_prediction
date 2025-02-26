# %%
import torch
import torch.nn as nn
import torch.optim as optim

from junifer.storage import HDF5FeatureStorage
import numpy as np
import pandas as pd
from argparse import ArgumentParser
from sklearn.kernel_ridge import KernelRidge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RepeatedKFold, cross_validate ##
from sklearn.model_selection import KFold
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from torch.utils.data import DataLoader, TensorDataset
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import make_scorer
from sklearn.compose import ColumnTransformer

# %%
parser = ArgumentParser()
parser.add_argument("--component", type=str, required=True, help="component to be selected")
args = parser.parse_args()
component = args.component
# %%
# Define Autoencoder Transformer 
class AutoencoderTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, input_dim, encoding_dim, epochs=100, batch_size=32, learning_rate=0.001, weight_decay=1e-4):
        self.input_dim = input_dim
        self.encoding_dim = encoding_dim
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.model = None
        self.feature_names_out_ = [f"component_{i+1}" for i in range(self.encoding_dim)]
        self.loadings_ = None

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

        X_train_tensor = torch.tensor(X, dtype=torch.float32)
        train_loader = DataLoader(TensorDataset(X_train_tensor), batch_size=self.batch_size, shuffle=True)

        self.model = self.Autoencoder(self.input_dim, self.encoding_dim)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)

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
                
            print(f"Epoch {epoch+1}/{self.epochs}, Loss: {loss.item():.4f}")

        self.extract_loadings(X)
        return self
    
    def transform(self, X):
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.tensor(X, dtype=torch.float32)
            encoded = self.model.encoder(X_tensor).numpy()
        df_encoded = pd.DataFrame(encoded, columns=self.feature_names_out_)
        return df_encoded

    def extract_loadings(self, X):
        first_layer_weights = self.model.encoder[0].weight.detach().cpu().numpy()
        second_layer_weights = self.model.encoder[2].weight.detach().cpu().numpy()
        third_layer_weights = self.model.encoder[4].weight.detach().cpu().numpy()
        fourth_layer_weights = self.model.encoder[6].weight.detach().cpu().numpy()   

        combined_weights = fourth_layer_weights @ third_layer_weights @ second_layer_weights @ first_layer_weights

        components = [f"component_{i+1}" for i in range(self.encoding_dim)]
        feature_names = [f"feature_{i+1}" for i in range(self.input_dim)]
        self.loadings_ = pd.DataFrame(combined_weights.T, columns=components, index=feature_names)

    def get_loadings(self):
        if self.loadings_ is not None:
            return self.loadings_
        else: 
            raise ValueError("Model has not been trained yet. Please call fit() first.")

# %%
# Load Data
storage = HDF5FeatureStorage("features.hdf5")
df_features = storage.read_df('BOLD_Schaefer400x17_functional_connectivity')
df_features = df_features.groupby("subject").mean()
columns = [x for x in df_features.columns if "~" in x]
columns = [x for x in columns if x.split("~")[0] != x.split("~")[1]]
df_features = df_features[columns]

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

X = df_data[df_features.columns]
y = df_data[behavioral_columns]

# %%
# Preprocessing
behavioral_imputer = IterativeImputer(max_iter=20, random_state=0)
behavioral_scaler = StandardScaler()
fmri_scaler = StandardScaler()

# Create ColumnTransformer for preprocessing
behavioral_preprocessor = ColumnTransformer([
    ("impute_scale", Pipeline([
        ("impute", behavioral_imputer),
        ("scale", behavioral_scaler)
    ]), behavioral_columns)
], remainder="drop")
fmri_preprocessor = ColumnTransformer([
    ("scale",fmri_scaler,X.columns)
])

# Define Autoencoder 
autoencoder = AutoencoderTransformer(input_dim=len(behavioral_columns), encoding_dim=3, epochs=100, batch_size=32, learning_rate=0.001, weight_decay=1e-4)

# Create Pipeline for Encoding Behavioral Data
behavioral_pipeline = Pipeline([
    ("preproessor",behavioral_preprocessor),
    ("autoencoder", autoencoder)
])
y_encoded = behavioral_pipeline.fit_transform(y)

fmri_pipeline = Pipeline([
    ("preprocessor",fmri_preprocessor)
])
X_scaled = fmri_pipeline.fit_transform(X)

# Define Kernel Ridge Regression Model
krr = KernelRidge(alpha= 1,kernel='linear')


# %%
# Define cross-validation
kf = KFold(n_splits=10, shuffle=True, random_state=42)

# Storage for loadings
all_fold_loadings = []

for fold_idx, (train_idx, test_idx) in enumerate(kf.split(X_scaled)):
    print(f"Training fold {fold_idx+1}")

    # Split data
    X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    # Train autoencoder
    y_train_encoded = behavioral_pipeline.fit_transform(y_train)
    
    # Extract trained autoencoder
    autoencoder_step = behavioral_pipeline.named_steps["autoencoder"]
    fold_loadings = autoencoder_step.get_loadings()
    
    # Store loadings
    all_fold_loadings.append(fold_loadings)
    fold_loadings.to_csv(f"/home/haotsung/HCP_behavioral_prediction/results/autoencoder_loadings/autoencoder_loadings_fold{fold_idx+1}_{component}.csv")

    # Transform test data
    y_test_encoded = behavioral_pipeline.transform(y_test)

    # Train regression model
    krr.fit(X_train, y_train_encoded[component])

    # Predict
    predictions = krr.predict(X_test)

    # Compute scores
    r2_score = krr.score(X_test, y_test_encoded[component])
    print(f"Fold {fold_idx+1}: R² Score = {r2_score:.4f}")

