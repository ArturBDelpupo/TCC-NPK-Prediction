# ==========================================================
# DATASET A - PREDIÇÃO E CLASS DE NPK
# ==========================================================

import os
import re
import warnings
import numpy as np
import pandas as pd
import shap
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns 
from sklearn.metrics import confusion_matrix
from matplotlib.colors import LinearSegmentedColormap

warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import logging
for _logger in ["joblib", "sklearn", "lightgbm", "xgboost", "shap", "catboost"]:
    logging.getLogger(_logger).setLevel(logging.ERROR)

print("=" * 30)
print("DATASET A - PREDIÇÃO E CLASS DE NPK")
print("=" * 30)

np.set_printoptions(suppress=True, formatter={'float_kind':'{:0.6f}'.format})
pd.set_option('display.float_format', '{:.6f}'.format)

# ==========================================================
# 1. CARREGAMENTO E PRÉ-PROCESSAMENTO
# ==========================================================
ARQUIVO_DADOS = r"LUCAS_2015_COMPLETO_CODIFICADO.csv"
VALOR_INVALIDO = -9999

print("\n[1] Carregando dataset...")
df = pd.read_csv(ARQUIVO_DADOS, sep=",", engine="python", on_bad_lines="skip")
df = df.loc[:, ~df.columns.duplicated()]

df = df.rename(columns={'pH(H2O)': 'pH_H2O', 'pH(CaCl2)': 'pH_CaCl2', 'Elevation': 'Elev'})
print(f"    Dataset: {len(df)} linhas × {len(df.columns)} colunas")

cols_numericas = df.select_dtypes(include=[np.number]).columns
for col in cols_numericas:
    mask = (df[col] == VALOR_INVALIDO) | (df[col] == float(VALOR_INVALIDO))
    if mask.sum() > 0:
        df.loc[mask, col] = np.nan

cols_numericas_base = ["N", "P", "K", "TH_LAT", "TH_LONG"]

for col in cols_numericas_base:
    if col in df.columns:
        df[col] = df[col].astype(str).str.replace(",", ".")
        df[col] = pd.to_numeric(df[col], errors="coerce") 

from scipy.stats.mstats import winsorize
for nut in ["N", "P", "K"]:
    if nut in df.columns:
        df[nut] = winsorize(df[nut], limits=(0.01, 0.01))

# ==========================================================
# 2. CONFIGURAÇÃO DE FILTROS
# ==========================================================
print("\n" + "=" * 30)
print("CONFIGURAÇÃO DE FILTROS")
print("=" * 30)

def perguntar(msg):
    return input(f"{msg} (Y/N): ").strip().upper() in ["Y", "YES", "S", "SIM"]

usar_filtro = perguntar("Gostaria de usar filtro?")
if usar_filtro:
    usar_lat = perguntar("Filtro geográfico (Portugal)?")
    usar_outliers = perguntar("Remover outliers extremos (NPK)?")
    usar_pH = perguntar("Filtrar pH agrícola (6.0–7.0)?")
    usar_ec = perguntar("Filtrar EC (salinidade extrema)?")
    usar_declive = perguntar("Filtrar elevação extrema?")
    usar_ndvi = perguntar("Filtrar vegetação (NDVI > 0.1)?")
    usar_clima = perguntar("Filtrar clima extremo?")
    usar_solo = perguntar("Filtrar solos agrícolas apenas?")
    usar_solo_natural = perguntar("Filtrar solos NATURAIS (Florestas/Matagais)?")
    usar_textura = perguntar("Filtrar por Tipo de Solo específico (Textura OLM)?")
    usar_producao = perguntar("Filtrar por Tipo de produção específica (LC1)?")


    print("\n" + "=" * 30)
    print("Filtros selecionados:")
    print("=" * 30)

    if usar_lat:
        df = df[(df["TH_LAT"] >= 36.95) & (df["TH_LAT"] <= 42.15)]
        print("Filtro Portugal aplicado")

    if usar_pH and "pH_H2O" in df.columns:
        df = df[(df["pH_H2O"] >= 6) & (df["pH_H2O"] <= 7)]
        print("Filtro de pH aplicado (6.0–7.0)")

    if usar_ec and "EC" in df.columns:
        limite_ec = df["EC"].quantile(0.99)
        df = df[df["EC"] < limite_ec]
        print(f"Filtro EC aplicado (limite: {limite_ec:.1f})")

    if usar_declive and "Elev" in df.columns:
        limite_elev = df["Elev"].quantile(0.99)
        df = df[df["Elev"] < limite_elev]
        print("Filtro de elevação aplicado")

    ndvi_cols = [c for c in df.columns if "NDVI" in c and re.match(r"^L_NDVI_t\d+$", c)]
    if usar_ndvi and len(ndvi_cols) > 0:
        for col in ndvi_cols: df[col] = pd.to_numeric(df[col], errors="coerce")
        df["NDVI_mean_temp"] = df[ndvi_cols].mean(axis=1, skipna=True)
        df = df[df["NDVI_mean_temp"].notna() & (df["NDVI_mean_temp"] > 0.1)]
        print("Filtro NDVI aplicado (vegetação real)")

    if usar_clima:
        temp_cols = [c for c in df.columns if re.match(r"^ERA5_temp_t\d+$", c)]
        chuva_cols = [c for c in df.columns if re.match(r"^ERA5_precip_t\d+$", c)]
        if len(temp_cols) > 0:
            for col in temp_cols: df[col] = pd.to_numeric(df[col], errors="coerce")
            df["Temp_mean_temp"] = df[temp_cols].mean(axis=1, skipna=True)
            if df["Temp_mean_temp"].mean() > 100: df["Temp_mean_temp"] = df["Temp_mean_temp"] - 273.15
            df = df[df["Temp_mean_temp"].notna() & (df["Temp_mean_temp"] > -20) & (df["Temp_mean_temp"] < 50)]
        if len(chuva_cols) > 0:
            for col in chuva_cols: df[col] = pd.to_numeric(df[col], errors="coerce")
            df["Chuva_mean_temp"] = df[chuva_cols].mean(axis=1, skipna=True)
            limite_chuva = df["Chuva_mean_temp"].quantile(0.99)
            df = df[df["Chuva_mean_temp"].notna() & (df["Chuva_mean_temp"] < limite_chuva)]
        print("Filtro climático aplicado")

    if usar_solo and "LU1_Desc" in df.columns:
        classes_agricolas = ["Agriculture (excluding fallow land and kitchen gardens)", "Fallow land", "Kitchen gardens"]
        df = df[df["LU1_Desc"].isin(classes_agricolas)]
        print("Filtro de uso do solo aplicado (Agricultura)")

    if usar_solo_natural and "LU1_Desc" in df.columns:
        classes_naturais = ["FORESTRY", "Semi-natural and natural areas not in use"]
        df = df[df["LU1_Desc"].isin(classes_naturais)]
        print("Filtro de solo natural aplicado (Florestas)")

    if usar_textura and "OLM_TextureClass" in df.columns:
        df = df.dropna(subset=["OLM_TextureClass"])
        alvo_textura = input("\nDigite as Texturas para manter (ex: 4.0, 7.0, 8.0, 9.0): ").strip()
        if alvo_textura:
            try:
                lista_alvos = [float(x.strip()) for x in alvo_textura.split(",")]
                df = df[df["OLM_TextureClass"].isin(lista_alvos)]
                print(f"Filtro aplicado")
            except ValueError:
                pass
else:
    print("Nenhum filtro usado")

print(f"\nDataset após filtros: {len(df)} amostras")
df = df.dropna(subset=["N", "P", "K"], how="all").drop_duplicates().reset_index(drop=True)
print(f"Dataset final sem duplicadas: {len(df)} amostras válidas")

# ==========================================================
# 3. EXTRAÇÃO DE FEATURES TEMPORAIS
# ==========================================================
print("\n" + "=" * 30)
print("[3] Extraindo features temporais...")
print("=" * 30)

todas_cols_brutas = []

l_ndvi_cols = [c for c in df.columns if re.match(r"^L_NDVI_t\d+$", c)]
todas_cols_brutas.extend(l_ndvi_cols)
if len(l_ndvi_cols) > 0:
    ndvi = df[l_ndvi_cols]
    df["L_NDVI_Max"], df["L_NDVI_Min"] = ndvi.max(axis=1).fillna(0), ndvi.min(axis=1).fillna(0)
    df["L_NDVI_Mean"], df["L_NDVI_Std"] = ndvi.mean(axis=1).fillna(0), ndvi.std(axis=1).fillna(0)
    df["L_NDVI_Amp"] = df["L_NDVI_Max"] - df["L_NDVI_Min"]
    df["L_NDVI_Skew"], df["L_NDVI_Kurt"] = ndvi.skew(axis=1).fillna(0), ndvi.kurt(axis=1).fillna(0)

l_ndmi_cols = [c for c in df.columns if re.match(r"^L_NDMI_t\d+$", c)]
todas_cols_brutas.extend(l_ndmi_cols)
if len(l_ndmi_cols) > 0:
    ndmi = df[l_ndmi_cols]
    df["L_NDMI_Max"], df["L_NDMI_Min"] = ndmi.max(axis=1).fillna(0), ndmi.min(axis=1).fillna(0)
    df["L_NDMI_Mean"], df["L_NDMI_Std"] = ndmi.mean(axis=1).fillna(0), ndmi.std(axis=1).fillna(0)
    df["L_NDMI_Amp"] = df["L_NDMI_Max"] - df["L_NDMI_Min"]
    df["L_NDMI_Skew"] = ndmi.skew(axis=1).fillna(0)

l_bsi_cols = [c for c in df.columns if re.match(r"^L_BSI_t\d+$", c)]
todas_cols_brutas.extend(l_bsi_cols)
if len(l_bsi_cols) > 0:
    bsi = df[l_bsi_cols]
    df["L_BSI_Max"], df["L_BSI_Min"] = bsi.max(axis=1).fillna(0), bsi.min(axis=1).fillna(0)
    df["L_BSI_Mean"], df["L_BSI_Std"] = bsi.mean(axis=1).fillna(0), bsi.std(axis=1).fillna(0)
    df["L_BSI_Amp"] = df["L_BSI_Max"] - df["L_BSI_Min"]
    df["L_BSI_Skew"] = bsi.skew(axis=1).fillna(0)

modis_ndvi_cols = [c for c in df.columns if re.match(r"^MODIS_NDVI_t\d+$", c)]
modis_evi_cols = [c for c in df.columns if re.match(r"^MODIS_EVI_t\d+$", c)]
todas_cols_brutas.extend(modis_ndvi_cols + modis_evi_cols)
for col_group, nome in [(modis_ndvi_cols, "MODIS_NDVI"), (modis_evi_cols, "MODIS_EVI")]:
    if len(col_group) > 0:
        tmp = df[col_group]
        df[f"{nome}_Mean"], df[f"{nome}_Std"] = tmp.mean(axis=1, skipna=True), tmp.std(axis=1, skipna=True).fillna(0)
        df[f"{nome}_Amp"] = tmp.max(axis=1, skipna=True) - tmp.min(axis=1, skipna=True)

modis_lst_day_cols = [c for c in df.columns if re.match(r"^MODIS_LST_day_t\d+$", c)]
modis_lst_night_cols = [c for c in df.columns if re.match(r"^MODIS_LST_night_t\d+$", c)]
todas_cols_brutas.extend(modis_lst_day_cols + modis_lst_night_cols)
for col_group, nome in [(modis_lst_day_cols, "MODIS_LST_day"), (modis_lst_night_cols, "MODIS_LST_night")]:
    if len(col_group) > 0:
        tmp = df[col_group]
        df[f"{nome}_Mean"], df[f"{nome}_Std"] = tmp.mean(axis=1, skipna=True), tmp.std(axis=1, skipna=True).fillna(0)
        df[f"{nome}_Amp"] = tmp.max(axis=1, skipna=True) - tmp.min(axis=1, skipna=True)

prefixos_outros = ["L_LST_t", "ERA5_temp_t", "ERA5_precip_t", "ERA5_soil_moist_t"]
for pref in prefixos_outros:
    cols = [c for c in df.columns if c.startswith(pref) and not c.endswith("Mean")]
    todas_cols_brutas.extend(cols)
    if len(cols) > 0:
        tmp = df[cols]
        df[f"{pref}_Mean"], df[f"{pref}_Std"] = tmp.mean(axis=1).fillna(0), tmp.std(axis=1).fillna(0)
        df[f"{pref}_Max"], df[f"{pref}_Min"] = tmp.max(axis=1).fillna(0), tmp.min(axis=1).fillna(0)
        df[f"{pref}_Amp"] = df[f"{pref}_Max"] - df[f"{pref}_Min"]

# ==========================================================
# 4. INTERAÇÕES AGRONÔMICAS
# ==========================================================
print("\n[4] Criando interações agronômicas...")
if "EC" in df.columns and "pH_H2O" in df.columns:
    df["RelacaoECpH"] = np.where(df["pH_H2O"] > 0, df["EC"] / df["pH_H2O"], 0)

if "L_NDVI_Mean" in df.columns:
    if "ERA5_temp_t_Mean" in df.columns: df["VigorClima"] = df["L_NDVI_Mean"] * df["ERA5_temp_t_Mean"]
    if "L_NDMI_Mean" in df.columns: df["Umidade_x_Vigor"] = df["L_NDMI_Mean"] * df["L_NDVI_Mean"]
    if "EC" in df.columns: df["EC_x_NDVI"] = df["EC"] * df["L_NDVI_Mean"]

if "L_NDMI_Mean" in df.columns:
    if "pH_H2O" in df.columns: df["pH_x_NDMI"] = df["pH_H2O"] * df["L_NDMI_Mean"]

if "MODIS_LST_day_Mean" in df.columns and "MODIS_NDVI_Mean" in df.columns: df["Estresse_Termico_MODIS"] = df["MODIS_LST_day_Mean"] / (df["MODIS_NDVI_Mean"] + 0.001)

# ==========================================================
# 5. PREPARAÇÃO DE FEATURES FINAIS
# ==========================================================
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, KFold, cross_val_predict
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

print("\n[5] Preparando matriz (80/20)...")

colunas_excluir = set(["N","P","K","POINTID","id_original","system:index",".geo","pH_CaCl2","OC","CaCO3",
                       "Coarse", "Clay", "Sand", "Silt", "Revisited_point", "Soil_Stones", "LC1", "LU1",
                       "LC1_Desc", "LU1_Desc", "NUTS_0", "NUTS_1", "NUTS_2", "NUTS_3",
                       "NDVI_mean_temp", "Temp_mean_temp", "Chuva_mean_temp", "SG_N_profundo",] + todas_cols_brutas)

df_numerico = df.select_dtypes(include=[np.number])
features_base = [c for c in df_numerico.columns if c not in colunas_excluir and not df[c].isna().all()]

# Remover features correlacionadas
print("   [5.1] Removendo features redundantes (r > 0.95)...")
tmp = df[features_base].dropna()
corr_matrix = tmp.corr().abs()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

fórmulas_agronomicas = ["RelacaoECpH", "VigorClima", "Umidade_x_Vigor", "EC_x_NDVI", "pH_x_NDMI", "Estresse_Termico_MODIS"]
to_drop = [column for column in upper.columns if any(upper[column] > 0.95) and column not in fórmulas_agronomicas]

features_base = [f for f in features_base if f not in to_drop]
print(f"Removidas {len(to_drop)} features. Restam {len(features_base)}.")

y_reg_original = df[["N", "P", "K"]].values
y_reg_log = np.log1p(y_reg_original)

idx_all = np.arange(len(df))
idx_train, idx_test = train_test_split(idx_all, test_size=0.20, random_state=42)

if len(l_ndvi_cols) > 0:
    imputer_pca_ndvi = SimpleImputer(strategy="mean")
    ndvi_train_imputed = imputer_pca_ndvi.fit_transform(df[l_ndvi_cols].iloc[idx_train])
    ndvi_all_imputed = imputer_pca_ndvi.transform(df[l_ndvi_cols])
    pca = PCA(n_components=8, random_state=42).fit(ndvi_train_imputed)
    emb = pca.transform(ndvi_all_imputed)
    for i in range(8): df[f"NDVI_embed_{i}"] = emb[:, i]
    features_base += [f"NDVI_embed_{i}" for i in range(8)]

if len(l_ndmi_cols) > 0:
    imputer_pca_ndmi = SimpleImputer(strategy="mean")
    ndmi_train_imputed = imputer_pca_ndmi.fit_transform(df[l_ndmi_cols].iloc[idx_train])
    ndmi_all_imputed = imputer_pca_ndmi.transform(df[l_ndmi_cols])
    pca_ndmi = PCA(n_components=5, random_state=42).fit(ndmi_train_imputed)
    emb_ndmi = pca_ndmi.transform(ndmi_all_imputed)
    for i in range(5): df[f"NDMI_embed_{i}"] = emb_ndmi[:, i]
    features_base += [f"NDMI_embed_{i}" for i in range(5)]

# ==========================================================
# 6. CRIAÇÃO DE CLASSES TERCIS
# ==========================================================
print("\n[6] Criando classes (tercis)...")
y_train_clf, y_test_clf = np.full((len(idx_train), 3), -1, dtype=int), np.full((len(idx_test), 3), -1, dtype=int)

for i, nut in enumerate(["N", "P", "K"]):
    st_train = df[nut].iloc[idx_train].dropna()
    bins = np.percentile(st_train, [0, 33.33, 66.67, 100])
    bins[0], bins[-1] = -np.inf, np.inf
    y_train_clf[:, i] = pd.cut(df[nut].iloc[idx_train], bins=bins, labels=[0,1,2]).astype(float).fillna(-1).astype(int).values
    y_test_clf[:, i] = pd.cut(df[nut].iloc[idx_test], bins=bins, labels=[0,1,2]).astype(float).fillna(-1).astype(int).values

imputer = SimpleImputer(strategy="median")
X_train_full = imputer.fit_transform(df[features_base].iloc[idx_train])
X_test_full = imputer.transform(df[features_base].iloc[idx_test])

kf = KFold(n_splits=10, shuffle=True, random_state=42)
print(f"Features base: {len(features_base)} | Treino: {len(idx_train)} | Teste Cego: {len(idx_test)}")

# Padronização global (fundamental para MLP e KNN)
scaler_global = StandardScaler()
X_train_full = scaler_global.fit_transform(X_train_full)
X_test_full = scaler_global.transform(X_test_full)

# ==========================================================
# 7. OTIMIZAÇÃO DA REDE NEURAL (MLP)
# ==========================================================

mask_n = ~np.isnan(y_reg_original[idx_train, 0])
X_sample = X_train_full[mask_n][:3000]
y_sample = y_reg_log[idx_train][mask_n][:3000]

from sklearn.model_selection import RandomizedSearchCV
from sklearn.neural_network import MLPRegressor, MLPClassifier

param_dist_mlp = {
    'hidden_layer_sizes': [(50,), (100,), (50,25), (100,50), (100,100)],
    'alpha': [0.0001, 0.001, 0.01, 0.1],
    'learning_rate_init': [0.0005, 0.001, 0.005],
    'batch_size': [32, 64, 128],
    'activation': ['relu', 'tanh']
}

mlp_base = MLPRegressor(max_iter=300, early_stopping=True, validation_fraction=0.1, n_iter_no_change=15, random_state=42, verbose=False)

random_search = RandomizedSearchCV(mlp_base, param_dist_mlp, n_iter=20, cv=3, scoring='r2', n_jobs=-1, random_state=42, verbose=1)
random_search.fit(X_sample, y_sample)

best_params = random_search.best_params_

mlp_reg_otimizado = MLPRegressor(
    **best_params,
    max_iter=500, early_stopping=True, validation_fraction=0.1, 
    n_iter_no_change=20, random_state=42, verbose=False
)

mlp_clf_otimizado = MLPClassifier(
    **best_params,
    max_iter=500, early_stopping=True, validation_fraction=0.1, 
    n_iter_no_change=20, random_state=42, verbose=False
)

# ==========================================================
# 8. MODELOS E FUNÇÕES DE AVALIAÇÃO
# ==========================================================
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, accuracy_score, f1_score
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, RandomForestClassifier, ExtraTreesClassifier
from xgboost import XGBRegressor, XGBClassifier
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier

import lightgbm as lgb
from catboost import CatBoostRegressor, CatBoostClassifier
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.preprocessing import StandardScaler

class FlatCatBoostClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, iterations=500, learning_rate=0.03, depth=6, random_state=42, verbose=0):
        self.iterations, self.learning_rate, self.depth, self.random_state, self.verbose = iterations, learning_rate, depth, random_state, verbose
    def fit(self, X, y):
        self.model = CatBoostClassifier(iterations=self.iterations, learning_rate=self.learning_rate, depth=self.depth, random_state=self.random_state, verbose=self.verbose, thread_count=-1, allow_writing_files=False)
        self.model.fit(X, y)
        return self
    def predict(self, X): return self.model.predict(X).ravel()

# Função para treinar e avaliar (com todos os modelos)
def treinar_e_avaliar(dict_features, iteracao_nome, primeira_iteracao=False):
    print(f"\n>> {iteracao_nome}")
    
    X_tr_dict = {}
    X_te_dict = {}
    for nut in ["N", "P", "K"]:
        idx_cols = [features_base.index(f) for f in dict_features[nut]]
        X_tr_dict[nut] = X_train_full[:, idx_cols]
        X_te_dict[nut] = X_test_full[:, idx_cols] 
    
    resultados_iteracao = []
    dict_scatter_stacking = {}
    dict_clf_preds = {nut: {} for nut in ["N", "P", "K"]}
    cat_treinados_reg = {}
    
    # --- REGRESSÃO ---
    modelos_reg = {
        "RandomForest": RandomForestRegressor(n_estimators=500, max_depth=6, random_state=42, n_jobs=-1),
        "ExtraTrees": ExtraTreesRegressor(n_estimators=500, max_depth=6, random_state=42, n_jobs=-1),
        "XGBoost": XGBRegressor(n_estimators=500, learning_rate=0.03, max_depth=6, random_state=42, n_jobs=-1),
        "LightGBM": lgb.LGBMRegressor(n_estimators=500, learning_rate=0.03, max_depth=6, num_leaves=63, random_state=42, verbose=-1, n_jobs=-1),
        "CatBoost": CatBoostRegressor(iterations=500, learning_rate=0.03, depth=6, random_state=42, verbose=0, thread_count=-1),
        "KNN": KNeighborsRegressor(n_neighbors=20, weights='distance', n_jobs=-1),
        "MLP": mlp_reg_otimizado,
    }
    
    for nome, modelo_template in modelos_reg.items():
        for i, nut in enumerate(["N", "P", "K"]):
            mask_tr = ~np.isnan(y_reg_original[idx_train, i])
            mask_te = ~np.isnan(y_reg_original[idx_test, i]) 
            
            X_nut_tr, y_nut_log_tr = X_tr_dict[nut][mask_tr], y_reg_log[idx_train][mask_tr, i]
            X_nut_te, y_nut_orig_te = X_te_dict[nut][mask_te], y_reg_original[idx_test][mask_te, i]
            
            m_cloned = clone(modelo_template)
            m_cloned.fit(X_nut_tr, y_nut_log_tr)  
            preds_log_te = m_cloned.predict(X_nut_te)
            preds_orig_te = np.expm1(preds_log_te)
            
            r2 = r2_score(y_nut_orig_te, preds_orig_te)
            mae = mean_absolute_error(y_nut_orig_te, preds_orig_te)
            rmse = np.sqrt(mean_squared_error(y_nut_orig_te, preds_orig_te))
            
            resultados_iteracao.append({"Filtro": iteracao_nome, "Modelo": f"Reg_{nome}", "Nutriente": nut, "R2": round(r2,3), "MAE": round(mae,2), "RMSE": round(rmse,2), "Acc": np.nan, "F1": np.nan})
            
            if nome == "CatBoost":
                cat_treinados_reg[nut] = m_cloned

    # --- CLASSIFICAÇÃO ---
    modelos_clf = {
        "RandomForest": RandomForestClassifier(n_estimators=500, max_depth=6, random_state=42, n_jobs=-1, class_weight='balanced'),
        "ExtraTrees": ExtraTreesClassifier(n_estimators=500, max_depth=6, random_state=42, n_jobs=-1, class_weight='balanced'),
        "XGBoost": XGBClassifier(n_estimators=500, learning_rate=0.03, max_depth=6, random_state=42, n_jobs=-1, verbosity=0, eval_metric="mlogloss"),
        "LightGBM": lgb.LGBMClassifier(n_estimators=500, learning_rate=0.03, max_depth=6, num_leaves=63, random_state=42, verbose=-1, n_jobs=-1, class_weight='balanced'),
        "CatBoost": FlatCatBoostClassifier(iterations=500, learning_rate=0.03, depth=6, random_state=42, verbose=0),
        "KNN": KNeighborsClassifier(n_neighbors=20, weights='distance', n_jobs=-1),
        "MLP": mlp_clf_otimizado,
    }
    
    for nome, modelo_template in modelos_clf.items():
        for i, nut in enumerate(["N", "P", "K"]):
            mask_tr = y_train_clf[:, i] != -1
            mask_te = y_test_clf[:, i] != -1 
            
            X_nut_tr, y_nut_tr = X_tr_dict[nut][mask_tr], y_train_clf[mask_tr, i]
            X_nut_te, y_nut_te = X_te_dict[nut][mask_te], y_test_clf[mask_te, i]
            
            m_cloned = clone(modelo_template)
            if nome == "XGBoost":
                from sklearn.utils.class_weight import compute_class_weight
                classes = np.unique(y_nut_tr)
                weights = compute_class_weight('balanced', classes=classes, y=y_nut_tr)
                sample_weights = np.zeros_like(y_nut_tr, dtype=float)
                for cls, w in zip(classes, weights):
                    sample_weights[y_nut_tr == cls] = w
                m_cloned.fit(X_nut_tr, y_nut_tr, sample_weight=sample_weights)
            elif nome == "MLP":
                m_cloned.fit(X_nut_tr, y_nut_tr)
            else:
                m_cloned.fit(X_nut_tr, y_nut_tr)
            
            preds_te = m_cloned.predict(X_nut_te) 
            acc = accuracy_score(y_nut_te, preds_te)
            f1 = f1_score(y_nut_te, preds_te, average='weighted')
            resultados_iteracao.append({"Filtro": iteracao_nome, "Modelo": f"Clf_{nome}", "Nutriente": nut, "R2": np.nan, "MAE": np.nan, "RMSE": np.nan, "Acc": round(acc,3), "F1": round(f1,3)})
            
            if primeira_iteracao:
                dict_clf_preds[nut][nome] = (y_nut_te, preds_te, acc)

    # --- STACKING NÃO LINEAR ---
    from sklearn.model_selection import cross_val_predict
    
    # Modelos base para stacking
    modelos_stack = {
        "LGB": lgb.LGBMRegressor(n_estimators=500, learning_rate=0.03, max_depth=6, num_leaves=63, random_state=42, verbose=-1, n_jobs=-1),
        "KNN": KNeighborsRegressor(n_neighbors=20, weights='distance', n_jobs=-1),
       "MLP": MLPRegressor(
        hidden_layer_sizes=(50, 25), activation='relu', solver='adam', alpha=0.001,
        batch_size=64, learning_rate='adaptive', learning_rate_init=0.001,
        max_iter=500, early_stopping=True, validation_fraction=0.1,
        n_iter_no_change=20, random_state=42, verbose=False)}
    
    for nut_i, nut in enumerate(["N", "P", "K"]):
        mask_tr = ~np.isnan(y_reg_original[idx_train, nut_i])
        mask_te = ~np.isnan(y_reg_original[idx_test, nut_i])
        
        X_nut_tr = X_tr_dict[nut][mask_tr]
        y_nut_log_tr = y_reg_log[idx_train][mask_tr, nut_i]
        X_nut_te = X_te_dict[nut][mask_te]
        y_nut_orig_te = y_reg_original[idx_test][mask_te, nut_i]
        
        cv_g = list(kf.split(X_nut_tr))
        n_base = len(modelos_stack)
        oof_preds = np.zeros((len(X_nut_tr), n_base))
        preds_te = np.zeros((len(X_nut_te), n_base))
        
        for j, (nome_s, modelo_s) in enumerate(modelos_stack.items()):
            modelo = clone(modelo_s)
            if nome_s == "MLP":
                oof_preds[:, j] = cross_val_predict(modelo, X_nut_tr, y_nut_log_tr, cv=cv_g, n_jobs=-1)
                modelo.fit(X_nut_tr, y_nut_log_tr)
                preds_te[:, j] = modelo.predict(X_nut_te)
            
            else:
                oof_preds[:, j] = cross_val_predict(modelo, X_nut_tr, y_nut_log_tr, cv=cv_g, n_jobs=-1)
                modelo.fit(X_nut_tr, y_nut_log_tr)
                preds_te[:, j] = modelo.predict(X_nut_te)
        
        scaler = StandardScaler()
        oof_scaled = scaler.fit_transform(oof_preds)
        te_scaled = scaler.transform(preds_te)
        
        # Meta-modelo: XGBoost com poucas árvores
        meta = XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.05, random_state=42, n_jobs=-1)
        meta.fit(oof_scaled, y_nut_log_tr)
        
        pred_stack_log = meta.predict(te_scaled)
        pred_stack_orig = np.expm1(pred_stack_log)
        
        r2 = r2_score(y_nut_orig_te, pred_stack_orig)
        mae = mean_absolute_error(y_nut_orig_te, pred_stack_orig)
        rmse = np.sqrt(mean_squared_error(y_nut_orig_te, pred_stack_orig))
        resultados_iteracao.append({"Filtro": iteracao_nome, "Modelo": "Reg_Stacking_XGB", "Nutriente": nut, "R2": round(r2,3), "MAE": round(mae,2), "RMSE": round(rmse,2), "Acc": np.nan, "F1": np.nan})
        
        if primeira_iteracao:
            dict_scatter_stacking[nut] = (y_nut_orig_te, pred_stack_orig)

    # =========================================================================
    # GRÁFICOS
    # =========================================================================
    # if primeira_iteracao:
    #     print("\nExportando Gráficos")
    #     cmap_N = LinearSegmentedColormap.from_list('cmap_N', ['blue', 'red'])   
    #     cmap_P = LinearSegmentedColormap.from_list('cmap_P', ['red', 'green'])  
    #     cmap_K = LinearSegmentedColormap.from_list('cmap_K', ['green', 'blue']) 
    #     paletas_erro = {"N": cmap_N, "P": cmap_P, "K": cmap_K}
    #     for nut_i, nut in enumerate(["N", "P", "K"]):
    #         fig, ax = plt.subplots(figsize=(6, 6))
    #         y_true_te, y_pred_te = dict_scatter_stacking[nut]
    #         limite_95_true = np.percentile(y_true_te, 95)
    #         limite_95_pred = np.percentile(y_pred_te, 95)
    #         mask_plot = (y_true_te <= limite_95_true) & (y_pred_te <= limite_95_pred)
    #         y_true_plot = y_true_te[mask_plot]
    #         y_pred_plot = y_pred_te[mask_plot]
    #         erro_abs = np.abs(y_true_plot - y_pred_plot)
    #         sc = ax.scatter(y_true_plot, y_pred_plot, c=erro_abs, cmap=paletas_erro[nut], alpha=0.8, s=15)
    #         max_val = max(y_true_plot.max(), y_pred_plot.max())
    #         ax.plot([0, max_val], [0, max_val], 'k--', alpha=0.8) 
    #         ax.set_xlabel("Valor Observado (Laboratório)")
    #         ax.set_ylabel("Valor Previsto (Stacking)")
    #         ax.grid(True, linestyle='-', alpha=0.6)
    #         cbar = plt.colorbar(sc, ax=ax)
    #         label_unidade = "Erro Absoluto (%)" if nut == "N" else "Erro Absoluto (mg/kg)"
    #         cbar.set_label(label_unidade, fontweight='bold')
    #         plt.tight_layout()
    #         nome_fig = f"grafico_dispersao_teste_{nut}.png"
    #         plt.savefig(nome_fig, dpi=300, bbox_inches="tight")
    #         plt.close()
    #         print(f"Gráfico Salvo: {nome_fig}")
    #     cm_cmaps = ['Blues', 'Reds', 'Greens'] 
    #     for nut_i, nut in enumerate(["N", "P", "K"]):
    #         fig, ax = plt.subplots(figsize=(5, 5))
    #         melhor_nome = max(dict_clf_preds[nut], key=lambda k: dict_clf_preds[nut][k][2])
    #         y_true_te, y_pred_te, _ = dict_clf_preds[nut][melhor_nome]
    #         cm = confusion_matrix(y_true_te, y_pred_te)
    #         sns.heatmap(cm, annot=True, fmt='d', cmap=cm_cmaps[nut_i], cbar=False, ax=ax,
    #                     xticklabels=["Baixa", "Média", "Alta"], yticklabels=["Baixa", "Média", "Alta"],
    #                     annot_kws={"weight": "bold", "size": 14})
    #         ax.set_xlabel(f"Modelo {melhor_nome}")
    #         ax.set_ylabel("LUCAS Dataset")
    #         plt.tight_layout()
    #         nome_cm = f"matriz_confusao_teste_{nut}.png"
    #         plt.savefig(nome_cm, dpi=300, bbox_inches="tight")
    #         plt.close()
    #         print(f"Matriz Salva: {nome_cm}")
    return pd.DataFrame(resultados_iteracao), cat_treinados_reg



# ==========================================================
# 9. RANKING SHAP
# ==========================================================
print("\n" + "=" * 30)
print("[9] Calculando importância das features (SHAP)...")
print("=" * 30)

# SHAP 
def calcular_shap_individual(cat_dict, dict_features):
    ranking_dict = {}
    n_samples = min(2000, len(X_train_full))
    indices = np.random.default_rng(42).choice(len(X_train_full), size=n_samples, replace=False)
    for nut in ["N", "P", "K"]:
        modelo = cat_dict[nut]
        feats = dict_features[nut]
        idx_cols = [features_base.index(f) for f in feats]
        X_shap = X_train_full[indices][:, idx_cols]
        explainer = shap.TreeExplainer(modelo)
        shap_vals = explainer.shap_values(X_shap, check_additivity=False)
        importancia = np.abs(shap_vals).mean(axis=0)
        df_imp = pd.DataFrame({"Feature": feats, "Importance": importancia}).sort_values("Importance", ascending=False)
        ranking_dict[nut] = df_imp
    return ranking_dict

features_atuais_todas = {"N": features_base.copy(), "P": features_base.copy(), "K": features_base.copy()}
_, cat_treinados_shap = treinar_e_avaliar(features_atuais_todas, "Cálculo do SHAP")

rankings_shap = calcular_shap_individual(cat_treinados_shap, features_atuais_todas)

print("\n" + "=" * 30)
print("[10] Testando diferentes tamanhos de subset...")
print("=" * 30)

etapas_fixas = [
    ("Todas as Features", len(features_base)),
    ("Top 50 Features", 50),
    ("Top 25 Features", 25),
    ("Top 10 Features", 10),
    ("Top 5 Features", 5)
]

tabela_geral_resultados = pd.DataFrame()

for iteracao_nome, corte in etapas_fixas:
    features_da_rodada = {}
    for nut in ["N", "P", "K"]:
        features_da_rodada[nut] = rankings_shap[nut]["Feature"].head(corte).tolist()
    df_res, _ = treinar_e_avaliar(features_da_rodada, iteracao_nome, primeira_iteracao=(corte == len(features_base)))
    tabela_geral_resultados = pd.concat([tabela_geral_resultados, df_res], ignore_index=True)
    print(f"\nFeatures selecionadas - {iteracao_nome}:")
    for nut in ["N", "P", "K"]:
        print(f"\n   Top {corte} features para {nut}:")
        print(rankings_shap[nut].head(corte).to_string(index=False))

# ==========================================================
# HISTOGRAMAS TREINO/TESTE
# ==========================================================
# print("\n" + "=" * 30)
# print("Gerando histograma treino/teste")
# print("=" * 30)

# top25_features = {}
# for nut in ["N", "P", "K"]:
#     top25_features[nut] = rankings_shap[nut]["Feature"].head(25).tolist()

# X_tr_top25 = {}
# X_te_top25 = {}
# for nut in ["N", "P", "K"]:
#     idx_cols = [features_base.index(f) for f in top25_features[nut]]
#     X_tr_top25[nut] = X_train_full[:, idx_cols]
#     X_te_top25[nut] = X_test_full[:, idx_cols]

# modelos_stack_hist = {
#     "LGB": lgb.LGBMRegressor(n_estimators=500, learning_rate=0.03, max_depth=6, num_leaves=63,
#                               random_state=42, verbose=-1, n_jobs=-1),
#     "KNN": KNeighborsRegressor(n_neighbors=20, weights='distance', n_jobs=-1),
#     "MLP": MLPRegressor(hidden_layer_sizes=(50, 25), activation='relu', solver='adam', alpha=0.001,
#                         batch_size=64, learning_rate='adaptive', learning_rate_init=0.001,
#                         max_iter=500, early_stopping=True, validation_fraction=0.1,
#                         n_iter_no_change=20, random_state=42, verbose=False)
# }

# nutriente_indices = {"N": 0, "P": 1, "K": 2}

# for nut in ["N", "P", "K"]:
#     i = nutriente_indices[nut]
#     mask_tr = ~np.isnan(y_reg_original[idx_train, i])
#     mask_te = ~np.isnan(y_reg_original[idx_test, i])

#     X_tr_nut = X_tr_top25[nut][mask_tr]
#     y_tr_log = y_reg_log[idx_train][mask_tr, i]
#     X_te_nut = X_te_top25[nut][mask_te]
#     y_te_orig = y_reg_original[idx_test][mask_te, i]
#     y_tr_orig = np.expm1(y_tr_log)   # valores reais de treino

#     # Cross-validation para OOF e treino final
#     cv_g = list(kf.split(X_tr_nut))
#     oof_preds = np.zeros((len(X_tr_nut), len(modelos_stack_hist)))
#     preds_te = np.zeros((len(X_te_nut), len(modelos_stack_hist)))

#     for j, (nome, modelo) in enumerate(modelos_stack_hist.items()):
#         mod = clone(modelo)
#         oof_preds[:, j] = cross_val_predict(mod, X_tr_nut, y_tr_log, cv=cv_g, n_jobs=-1)
#         mod.fit(X_tr_nut, y_tr_log)
#         preds_te[:, j] = mod.predict(X_te_nut)

#     # Meta-modelo (XGBoost)
#     scaler = StandardScaler()
#     oof_scaled = scaler.fit_transform(oof_preds)
#     te_scaled = scaler.transform(preds_te)

#     meta = XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.05, random_state=42, n_jobs=-1)
#     meta.fit(oof_scaled, y_tr_log)

#     train_pred_log = meta.predict(oof_scaled)
#     test_pred_log = meta.predict(te_scaled)
#     train_pred_orig = np.expm1(train_pred_log)
#     test_pred_orig = np.expm1(test_pred_log)

#     fig, axes = plt.subplots(2, 1, figsize=(10, 8))

#     axes[0].hist(y_tr_orig, bins=30, alpha=0.6, label='Treino', color='steelblue')
#     axes[0].hist(y_te_orig, bins=30, alpha=0.6, label='Teste', color='darkorange')
#     axes[0].set_title(f'Distribuição dos valores reais de {nut} – Treino vs Teste')
#     axes[0].set_xlabel(f'{nut} (mg/kg)' if nut != "N" else 'N (%)')
#     axes[0].set_ylabel('Frequência')
#     axes[0].legend()

#     axes[1].hist(train_pred_orig, bins=30, alpha=0.6, label='Predições Treino', color='steelblue')
#     axes[1].hist(test_pred_orig, bins=30, alpha=0.6, label='Predições Teste', color='darkorange')
#     axes[1].set_title(f'Distribuição das predições do Stacking (Top 25 features) para {nut}')
#     axes[1].set_xlabel(f'{nut} (mg/kg)' if nut != "N" else 'N (%)')
#     axes[1].set_ylabel('Frequência')
#     axes[1].legend()

#     plt.tight_layout()
#     nome_hist = f"histograma{nut}.png"
#     plt.savefig(nome_hist, dpi=300, bbox_inches="tight")
#     plt.close()
#     print(f"Histograma treino/teste para {nut} salvo em '{nome_hist}'")

# ==========================================================
# 10. RESULTADOS FINAIS
# ==========================================================
print("\n" + "=" * 30)
print("Resultados")
print("=" * 30)

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

print(tabela_geral_resultados.to_string(index=False))

# tabela_geral_resultados.to_csv("TCC_Resultados.csv", index=False, sep=";", decimal=",")
# print("\nTabela exportada com sucesso para: 'TCC_Resultados.csv'")
# print("=" * 30)