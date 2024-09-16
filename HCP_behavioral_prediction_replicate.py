# %%
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from scipy.stats import pearsonr
from sklearn.metrics import r2_score
from argparse import ArgumentParser

# %%
parser = ArgumentParser()
parser.add_argument("--pca_component", type=str, required=True, help="pca component to be selected")
args = parser.parse_args()
pca_component = args.pca_component
# %%
# Load the dataset
df = pd.read_csv("HCP_data.csv")

# 80 specific subjects same as the paper
_80subject_number = [109325,114116,115219,119126,122418,123824,130518,141119,143426,146634,147030,149741,151728,153126,153833,154330,155635,156637,160830,165032,172635,180129,182032,183337,186949,198249,199958,200917,204218,207426,208428,
211720,212318,248238,250427,286347,293748,298051,298455,303119,311320,314225,342129,368753,376247,378756,385046,392447,419239,433839,453542,462139,465852,518746,555348,555651,562345,569965,571548,578057,578158,579665,609143,
615441,656253,689470,701535,727553,748662,757764,788876,825654,827052,877168,878877,888678,905147,922854,929464,995174]

# 753 specific subject for main analysis same as the paper
main_anlysis_number = [
100206,100307,100408,101006,101107,101309,101410,101915,102311,102513,102614,102715,102816,103111,103212,103414,103818,104012,104416,104820,105014,105115,105216,105620,105923,106016,106319,106521,106824,107018,107321,107422,
107725,108020,108222,108525,108828,109123,110007,110411,110613,111009,111211,111312,111413,111514,111716,112112,112314,112516,112920,113215,113316,113619,113922,114217,114419,114621,114823,115017,115320,115724,115825,116221,
117021,117324,117930,118124,118225,118528,118730,118831,118932,119025,119732,119833,120111,120212,120414,120515,120717,121416,121618,121921,122317,122620,122822,123925,124422,124826,125222,125424,125525,126325,126426,126628,
127226,127630,127832,127933,128026,128127,128632,128935,129028,129129,129331,129634,129937,130013,130114,130316,130417,130619,130720,130821,131217,131419,131722,131823,132017,133019,133625,133827,133928,134021,134223,134425,
134728,134829,135124,135225,135528,135629,135730,135932,136227,136631,136833,137229,137532,137633,138130,138231,138332,138534,138837,139233,139637,139839,140117,140925,141422,141826,143224,143325,143830,144125,144428,144832,
144933,145127,145632,146129,146331,146432,146533,146735,146836,146937,147636,147737,148032,148133,148335,148840,148941,149236,149337,149539,149842,150019,150625,150726,150928,151021,151223,151425,151627,151829,151930,152225,
152427,152831,153025,153227,153429,154229,154431,154532,154734,154835,154936,155231,155938,156031,156233,156334,156435,156536,157336,157437,157942,158035,158136,158338,158540,158843,159138,159239,159340,159441,159744,159946,
160123,160931,161731,162026,162228,162329,162733,163129,164030,164131,164636,164939,165436,165638,166438,167036,167238,167440,168341,168745,168947,169444,169545,169949,170631,170934,171330,172029,172130,172332,172433,172534,
172938,173334,173435,173536,173637,173839,173940,174437,174841,175136,175237,175338,175742,176037,176441,176542,176845,177140,177241,177645,178243,178849,178950,179245,179346,180230,180432,180735,180937,181232,182739,183034,
183741,185038,185139,185341,185442,185947,186141,186444,186545,187345,187547,187850,188145,188347,188448,188549,189349,189450,189652,190031,191033,191235,191336,191841,191942,192035,192136,192540,192641,192843,193239,193845,
194140,194443,194645,195041,195445,196144,196346,196750,198350,198451,198653,199352,199453,199655,200008,200109,200210,200311,200513,200614,201111,201414,201515,201818,202113,202719,202820,203418,203923,204319,204420,204521,
204622,205119,205725,205826,206525,206828,208226,208327,209127,209228,209329,209935,210011,210112,210617,211114,211215,211316,211417,211619,211821,211922,212015,212419,213017,213421,213522,214019,214221,214423,214726,217126,
217429,219231,220721,221319,227432,227533,228434,231928,233326,237334,238033,239944,245333,246133,248339,250932,251833,255639,255740,257542,257946,263436,268749,268850,270332,274542,275645,281135,285345,285446,286650,287248,
297655,299154,300618,300719,303624,304727,307127,308129,308331,309636,310621,316633,316835,317332,318637,320826,321323,322224,325129,329844,330324,333330,334635,336841,339847,341834,346945,349244,350330,351938,352132,353740,
358144,360030,361234,365343,366042,371843,377451,379657,380036,381038,381543,385450,386250,387959,389357,391748,393550,394956,395251,395756,395958,397154,397760,397861,401422,406836,412528,413934,414229,415837,421226,422632,
424939,429040,432332,436239,436845,441939,445543,448347,453441,459453,461743,463040,467351,469961,473952,479762,481042,481951,485757,486759,492754,495255,497865,499566,500222,506234,510326,513130,513736,516742,517239,519950,
520228,522434,523032,524135,525541,529549,529953,530635,531536,536647,540436,541640,541943,545345,548250,550439,552241,553344,555954,557857,558960,559053,559457,561242,561444,561949,562446,565452,566454,567052,567759,567961,
568963,570243,571144,572045,573249,573451,579867,580044,580751,581349,581450,583858,585256,586460,587664,588565,589567,594156,597869,598568,599065,599671,604537,611938,613538,615744,616645,617748,618952,622236,623844,626648,
627549,627852,628248,634748,635245,638049,644246,645450,645551,647858,654350,654552,654754,656657,657659,660951,663755,664757,665254,667056,671855,672756,673455,675661,677968,679568,679770,680452,680957,683256,687163,688569,
690152,692964,693764,694362,698168,700634,702133,704238,707749,709551,715950,720337,724446,725751,728454,729557,731140,732243,734045,742549,744553,748258,749058,751348,751550,753150,756055,759869,761957,765056,765864,767464,
769064,770352,771354,773257,774663,782561,783462,784565,786569,788674,789373,792564,792867,800941,802844,803240,810843,812746,814548,815247,816653,818455,818859,820745,825048,825553,826353,826454,828862,832651,833148,835657,
837560,841349,843151,844961,845458,849264,849971,852455,856463,856766,856968,857263,859671,861456,865363,869472,870861,871762,871964,872562,873968,877269,880157,882161,885975,886674,887373,889579,891667,894067,894673,894774,
896778,896879,898176,899885,901139,901442,902242,904044,907656,908860,910241,910443,911849,912447,917255,917558,923755,927359,930449,932554,933253,942658,943862,952863,955465,957974,958976,959574,962058,965367,965771,966975,
969476,970764,971160,973770,978578,983773,984472,987074,987983,989987,990366,991267,992673,992774,993675,994273,996782]

behavioral_measure_columns= ['PicSeq_Unadj','CardSort_Unadj','Flanker_Unadj','PMAT24_A_CR','ReadEng_Unadj','PicVocab_Unadj','ProcSpeed_Unadj','VSPLOT_TC','SCPT_SEN','SCPT_SPEC','IWRD_TOT','ListSort_Unadj','MMSE_Score',
                     'PSQI_Score','Endurance_Unadj','Dexterity_Unadj','Strength_Unadj','Odor_Unadj','PainInterf_Tscore','Taste_Unadj','Mars_Final','Emotion_Task_Face_Acc','Language_Task_Math_Avg_Difficulty_Level',
                     'Language_Task_Story_Avg_Difficulty_Level','Relational_Task_Acc','Social_Task_Perc_Random','Social_Task_Perc_TOM','WM_Task_Acc','NEOFAC_A','NEOFAC_O','NEOFAC_C','NEOFAC_N','NEOFAC_E','ER40_CR','ER40ANG','ER40FEAR',
                     'ER40HAP','ER40NOE','ER40SAD','AngAffect_Unadj','AngHostil_Unadj','AngAggr_Unadj','FearAffect_Unadj','FearSomat_Unadj','Sadness_Unadj','LifeSatisf_Unadj','MeanPurp_Unadj','PosAffect_Unadj','Friendship_Unadj',
                     'Loneliness_Unadj','PercHostil_Unadj','PercReject_Unadj','EmotSupp_Unadj','InstruSupp_Unadj','PercStress_Unadj','SelfEff_Unadj','DDisc_AUC_40K','GaitSpeed_Comp']

# %%
# Select the 80 samples for building the PCA model 
_80selected_df = df[df['subject'].isin(_80subject_number)]  # Select 80 subjects
_80selected_to_scale_df = _80selected_df.drop(columns=['subject']) # Separate the 'subject' column
# Standardize the data
scaler = StandardScaler()
_80selected_scaled_df = scaler.fit_transform(_80selected_to_scale_df)
_80selected_scaled_df = pd.DataFrame(_80selected_scaled_df, columns=_80selected_to_scale_df.columns) # Convert the scaled NumPy array back to a DataFrame, keeping the original column names
final_80selected_scaled_df = pd.concat([_80selected_df['subject'].reset_index(drop=True), _80selected_scaled_df], axis=1) # Combine the subject column with the scaled columns
# Select the behavioral measure columns
_80selected_behavioral_measures_df = final_80selected_scaled_df[behavioral_measure_columns]

# For the main analysis, repeat the same process
main_analysis_df = df[df['subject'].isin(main_anlysis_number)]  # Select subjects for main analysis
main_analysis_to_scale_df = main_analysis_df.drop(columns=['subject']) # Separate the 'subject' column
# Apply StandardScaler to the main analysis data
main_analysis_scaled_df = scaler.transform(main_analysis_to_scale_df) 
main_analysis_scaled_df  = pd.DataFrame(main_analysis_scaled_df , columns=main_analysis_to_scale_df.columns) # Convert the scaled NumPy array back to a DataFrame, keeping the original column names
final_main_analysis_scaled_df = pd.concat([main_analysis_df['subject'].reset_index(drop=True), main_analysis_scaled_df], axis=1) # Combine the subject column with the scaled columns
# Select the behavioral measure columns
main_analysis_behavioral_measures_df = final_main_analysis_scaled_df[behavioral_measure_columns]


# Fit PCA to get the component matrix 
pca = PCA(n_components=3)
pca.fit(_80selected_behavioral_measures_df)
main_analysis_transformed_behavioral_measures= pca.transform(main_analysis_behavioral_measures_df)

main_analysis_transformed_behavioral_measures = pd.DataFrame(main_analysis_transformed_behavioral_measures)
main_analysis_transformed_behavioral_measures.insert(0, 'subject', main_analysis_df['subject'].reset_index(drop=True)) # Add the subject column to the data
# Rename specific columns
main_analysis_transformed_behavioral_measures = main_analysis_transformed_behavioral_measures.rename(columns={0: 'PC1', 1: 'PC2', 2: 'PC3'})
#print(main_analysis_transformed_behavioral_measures.columns)
# %%
# Regression model training
# Step 1: Seperate features (X) and targets (y)

X = final_main_analysis_scaled_df.iloc[:,1:79801]
y = main_analysis_transformed_behavioral_measures[f"{pca_component}"]

# Step 2: train/test split by 10-fold cross validation
k = 10
kf = KFold(n_splits = k)
r_squared_scores = []
r_corr_scores = []

for train_index, test_index in kf.split(X):

    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

# Step 3: Train a Linear Regression Model for Data
    model = LinearRegression()
    model.fit(X_train, y_train)
        

# Step 4: Prediction for data
    y_pred = model.predict(X_test)

# Step 5: Scoring for data 
    r_squared = r2_score(y_test, y_pred)
    r_corr, _ = pearsonr(y_test, y_pred)
    r_squared_scores.append(r_squared)
    r_corr_scores.append(r_corr)

result_df = pd.DataFrame(
    {
        "Folds" : [i for i in range(k)],
        "component" : [pca_component for i in range(k)],
        "r_corr": r_corr_scores,
        "r2": r_squared_scores
    }
)
result_df.to_csv(f"/home/haotsung/HCP_behavioral_prediction/results/replicate/scores_component{pca_component}.csv")