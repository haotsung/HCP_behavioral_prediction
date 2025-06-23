# %%
import pandas as pd
import numpy as np
import umap
from sklearn.decomposition import PCA
from junifer.storage import HDF5FeatureStorage
from argparse import ArgumentParser
from sklearn.preprocessing import StandardScaler
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import GroupKFold
from sklearn.metrics import make_scorer
from scipy.stats import pearsonr
from sklearn.metrics import r2_score
from scipy.optimize import linear_sum_assignment

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

# %%
from sklearn.model_selection._split import BaseCrossValidator
class RepeatedGroupKFold(BaseCrossValidator):
    def __init__(self, n_splits=5, n_repeats=5, random_state=None):
        self.n_splits = n_splits
        self.n_repeats = n_repeats
        self.random_state = random_state

    def split(self, X, y=None, groups=None):
        if groups is None:
            raise ValueError("Groups must be provided for GroupKFold.")

        unique_groups = np.unique(groups)
        
        rng = np.random.default_rng(self.random_state)
        for repeat in range(self.n_repeats):
            # Shuffle groups before each repeat
            shuffled_groups = rng.permutation(unique_groups)

            folds = np.array_split(shuffled_groups, self.n_splits)

            for fold_groups in folds:
                test_idx = np.isin(groups, fold_groups)
                train_idx = ~test_idx

                yield np.where(train_idx)[0], np.where(test_idx)[0]

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits * self.n_repeats

# %%
def pearson_corr(y_true, y_pred):
    return pearsonr(y_true, y_pred)[0] 
pearson_scorer = make_scorer(pearson_corr, greater_is_better=True)

# %%
parser = ArgumentParser()
parser.add_argument("--component", type=str, required=True, help="component to be selected")
args = parser.parse_args()
component = args.component

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
brain_columns = df_features.columns.tolist()

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

# %%
X = df_data[columns]
y = df_data[behavioral_columns]
groups = df_data["Family_ID"].values
rgkf = RepeatedGroupKFold(n_repeats=5,n_splits = 10, random_state=42)
results = [] 
for fold_idx, (train_idx,test_idx) in enumerate(rgkf.split(X,y,groups=groups)): 
    X_train, X_test = X.iloc[train_idx,:], X.iloc[test_idx,:] 
    y_train, y_test = y.iloc[train_idx,:], y.iloc[test_idx,:] 

    # Impute missing value
    imputer = IterativeImputer(max_iter=20,random_state=42)
    y_train_imputed = imputer.fit_transform(y_train)
    y_test_imputed = imputer.transform(y_test)

    # Z-scoring
    X_scaler = StandardScaler()
    X_train_scaled = X_scaler.fit_transform(X_train)
    X_test_scaled = X_scaler.transform(X_test)
    y_scaler = StandardScaler()
    y_train_scaled = y_scaler.fit_transform(y_train_imputed)
    y_test_scaled = y_scaler.transform(y_test_imputed)

    # Compute reference PCA embedding
    pca = PCA(n_components=3,random_state=42)
    y_train_pca = pca.fit_transform(y_train_scaled)
    y_test_pca = pca.transform(y_test_scaled)
    y_test_pca = pd.DataFrame(
        y_test_pca,
        columns=["component_1", "component_2", "component_3"],
    )

    # Dimensionality reduction of behavioral data 
    reducer = umap.UMAP(n_components=3,random_state=42) 
    y_train_reduced = reducer.fit_transform(y_train_scaled)
    y_test_reduced = reducer.transform(y_test_scaled)

    # Reorder the components
    # train set
    corr_train = np.corrcoef(y_train_reduced.T,y_train_pca.T)[:3,3:]
    cost_train = -np.abs(corr_train)
    row_ind, col_ind = linear_sum_assignment(cost_train)
    y_train_aligned = np.empty_like(y_train_reduced)
    for ui, pj in zip(row_ind, col_ind):
        sign = np.sign(corr_train[ui, pj])
        y_train_aligned[:, pj] = y_train_reduced[:, ui] * sign

    # test set
    corr_test = np.corrcoef(y_test_reduced.T,y_test_pca.T)[:3,3:]
    cost_test = -np.abs(corr_test)
    row_ind, col_ind = linear_sum_assignment(cost_test)
    y_test_aligned = np.empty_like(y_test_reduced)
    for ui, pj in zip(row_ind, col_ind):
        sign = np.sign(corr_test[ui, pj])
        y_test_aligned[:, pj] = y_test_reduced[:, ui] * sign

    y_train_aligned = pd.DataFrame(
        y_train_aligned,
        columns=["component_1", "component_2", "component_3"],
    )
    y_test_aligned = pd.DataFrame(
        y_test_aligned,
        columns=["component_1", "component_2", "component_3"],
    )


    # Regression
    krr = KernelRidge(kernel='linear')
    param_grid = {
    'alpha': [0.000001,
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
              50]
              }
    scoring = {
        'r2': "r2",
        'r_corr': pearson_scorer
    }

    grid_search = GridSearchCV(
        estimator=krr, 
        param_grid=param_grid, 
        scoring=scoring, 
        refit='r_corr',
        cv=10,
        return_train_score=True
        )

    grid_search.fit(X_train, y_train_aligned[component])

    # Evaluate best model
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    print("Best alpha:", grid_search.best_params_['alpha'])
    print("Test r score:", pearson_corr(y_test_aligned[component], y_pred))
    # store whatever you want
    results.append({
        "fold":     fold_idx,
        "r_corr":   pearson_corr(y_test_aligned[component], y_pred),
        "r2_corr":  r2_score(y_test_aligned[component], y_pred),
    })

#%%
# turn into DataFrame and save
df_results = pd.DataFrame(results)
df_results.to_csv(f"/home/haotsung/HCP_behavioral_prediction/results/umap_results/scores_krr_{component}_align.csv")