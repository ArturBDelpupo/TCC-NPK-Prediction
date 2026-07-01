# ==========================================================
# LUCAS SOIL — SISTEMA NPK
# ==========================================================

import os
import re
import warnings
import numpy as np
import pandas as pd
from collections import defaultdict
import shap
import joblib

# ----------------------------------------------------------
# Silenciar TUDO
# ----------------------------------------------------------
warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import logging
for _logger in ["joblib", "sklearn", "lightgbm", "xgboost", "shap", "catboost"]:
    logging.getLogger(_logger).setLevel(logging.ERROR)

_orig_call = joblib.parallel.BatchedCalls.__call__
def _silent_call(self, *args, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return _orig_call(self, *args, **kwargs)
joblib.parallel.BatchedCalls.__call__ = _silent_call

print("=" * 90)
print("🚀 SISTEMA NPK V45.0 — UNIFIED RFE-SHAP + TESTE CEGO ABSOLUTO")
print("=" * 90)

# ==========================================================
# CONFIGURAÇÃO E CARREGAMENTO
# ==========================================================
ARQUIVO_DADOS = "LUCAS2015_topsoildata/LUCAS_2015_COMPLETO.csv" # 💡 Ajuste para o seu ficheiro mestre
VALOR_INVALIDO = -9999

print("\n[1] Carregando dataset...")
df = pd.read_csv(ARQUIVO_DADOS, sep=",", engine="python", on_bad_lines="skip")
df = df.loc[:, ~df.columns.duplicated()]

# Padronizar nomes de colunas que vieram do LUCAS original
df = df.rename(columns={'pH(H2O)': 'pH_H2O', 'pH(CaCl2)': 'pH_CaCl2', 'Elevation': 'Elev'})

print(f"    Dataset: {len(df)} linhas × {len(df.columns)} colunas")

print("\n[1.1] Convertendo valores inválidos (-9999) para NaN...")
cols_numericas = df.select_dtypes(include=[np.number]).columns
for col in cols_numericas:
    mask = (df[col] == VALOR_INVALIDO) | (df[col] == float(VALOR_INVALIDO))
    if mask.sum() > 0:
        df.loc[mask, col] = np.nan

cols_numericas_base = ["N", "P", "K", "EC", "pH_H2O", "TH_LAT", "TH_LONG"]
for col in cols_numericas_base:
    if col in df.columns:
        df[col] = df[col].astype(str).str.replace(",", ".")
        df[col] = pd.to_numeric(df[col], errors="coerce") 

# ==========================================================
# ==========================================================
# 2. FILTROS INTERATIVOS
# ==========================================================
print("\n" + "=" * 80)
print("📊 CONFIGURAÇÃO DE FILTROS")
print("=" * 80)

def perguntar(msg):
    return input(f"👉 {msg} (Y/N): ").strip().upper() in ["Y", "YES", "S", "SIM"]

usar_filtro = perguntar("Gostaria de usar filtro?")
if usar_filtro:
    usar_lat = perguntar("Filtro geográfico (Portugal)?")
    usar_outliers = perguntar("Remover outliers extremos (NPK)?")
    usar_pH = perguntar("Filtrar pH agrícola (4.5–8.5)?")
    usar_ec = perguntar("Filtrar EC (salinidade extrema)?")
    usar_declive = perguntar("Filtrar elevação extrema?")
    usar_ndvi = perguntar("Filtrar vegetação (NDVI > 0.1)?")
    usar_clima = perguntar("Filtrar clima extremo?")
    usar_solo = perguntar("Filtrar solos agrícolas apenas?")
    usar_solo_natural = perguntar("Filtrar solos NATURAIS (Florestas/Matagais)?")
    usar_textura = perguntar("Filtrar por Tipo de Solo específico (Textura OLM)?")

    print("\n" + "=" * 80)
    print("🔧 APLICANDO FILTROS")
    print("=" * 80)

    if usar_lat:
        df = df[(df["TH_LAT"] >= 36.95) & (df["TH_LAT"] <= 42.15)]
        print("  ✅ Filtro Portugal aplicado")

    if usar_outliers:
        for col in ["N", "P", "K"]:
            if col in df.columns:
                q1 = df[col].quantile(0.05)
                q99 = df[col].quantile(0.95)
                df = df[(df[col].isna()) | ((df[col] >= q1) & (df[col] <= q99))]
        print("  ✅ Outliers removidos (1%-99%)")

    if usar_pH and "pH_H2O" in df.columns:
        df = df[(df["pH_H2O"] >= 4.5) & (df["pH_H2O"] <= 8.5)]
        print("  ✅ Filtro de pH aplicado (4.5–8.5)")

    if usar_ec and "EC" in df.columns:
        limite_ec = df["EC"].quantile(0.99)
        df = df[df["EC"] < limite_ec]
        print(f"  ✅ Filtro EC aplicado (limite: {limite_ec:.1f})")

    if usar_declive and "Elev" in df.columns:
        limite_elev = df["Elev"].quantile(0.99)
        df = df[df["Elev"] < limite_elev]
        print("  ✅ Filtro de elevação aplicado")

    ndvi_cols = [c for c in df.columns if "NDVI" in c and re.match(r"^L_NDVI_t\d+$", c)]
    if usar_ndvi and len(ndvi_cols) > 0:
        for col in ndvi_cols: df[col] = pd.to_numeric(df[col], errors="coerce")
        df["NDVI_mean_temp"] = df[ndvi_cols].mean(axis=1, skipna=True)
        df = df[df["NDVI_mean_temp"].notna() & (df["NDVI_mean_temp"] > 0.1)]
        print("  ✅ Filtro NDVI aplicado (vegetação real)")

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
        print("  ✅ Filtro climático aplicado")

    if usar_solo and "LU1_Desc" in df.columns:
        classes_agricolas = ["Agriculture (excluding fallow land and kitchen gardens)", "Fallow land", "Kitchen gardens"]
        df = df[df["LU1_Desc"].isin(classes_agricolas)]
        print("  ✅ Filtro de uso do solo aplicado (Agricultura)")

    if usar_solo_natural and "LU1_Desc" in df.columns:
        classes_naturais = ["Forestry", "Semi-natural and natural areas not in use"]
        df = df[df["LU1_Desc"].isin(classes_naturais)]
        print("  ✅ Filtro de solo natural aplicado (Florestas)")

    if usar_textura and "OLM_TextureClass" in df.columns:
        df = df.dropna(subset=["OLM_TextureClass"])
        DICIONARIO_USDA = {
            1.0: "Argila", 2.0: "Argila Siltosa", 3.0: "Argila Arenosa",
            4.0: "Franco Argiloso", 5.0: "Franco Argilo-Siltoso",
            6.0: "Franco Argilo-Arenoso", 7.0: "Franco", 8.0: "Franco Siltoso",
            9.0: "Franco Arenoso", 10.0: "Silte", 11.0: "Areia Franca", 12.0: "Areia"
        }
        tipos_disponiveis = df["OLM_TextureClass"].dropna().unique()
        tipos_disponiveis.sort()
        print("\n  Tipos de solo disponíveis:")
        for t in tipos_disponiveis: print(f"    [{int(t)}] → {DICIONARIO_USDA.get(t, 'Classe Desconhecida')}")
        alvo_textura = input("\n  👉 Digite os números para manter (ex: 4.0, 7.0, 8.0, 9.0): ").strip()
        if alvo_textura:
            try:
                lista_alvos = [float(x.strip()) for x in alvo_textura.split(",")]
                df = df[df["OLM_TextureClass"].isin(lista_alvos)]
                print(f"  ✅ Filtro aplicado")
            except ValueError:
                print("  ⚠️ Formato inválido. Filtro ignorado.")
else:
    print("Nenhum filtro usado")

print(f"\n📊 Dataset após filtros: {len(df)} amostras")
df = df.dropna(subset=["N", "P", "K"], how="all").reset_index(drop=True)
print(f"📊 Dataset final: {len(df)} amostras válidas")

# ==========================================================
# 3. EXTRAÇÃO DE FEATURES FENOLÓGICAS E CLIMA
# ==========================================================
print("\n" + "=" * 80)
print("📈 EXTRAINDO FEATURES FENOLÓGICAS E CLIMA")
print("=" * 80)

todas_cols_brutas = []

# --- 3.1. Landsat NDVI ---
l_ndvi_cols = [c for c in df.columns if re.match(r"^L_NDVI_t\d+$", c)]
todas_cols_brutas.extend(l_ndvi_cols)
if len(l_ndvi_cols) > 0:
    ndvi = df[l_ndvi_cols]
    df["L_NDVI_Max"] = ndvi.max(axis=1).fillna(0)
    df["L_NDVI_Min"] = ndvi.min(axis=1).fillna(0)
    df["L_NDVI_Mean"] = ndvi.mean(axis=1).fillna(0)
    df["L_NDVI_Std"] = ndvi.std(axis=1).fillna(0)
    df["L_NDVI_Amp"] = df["L_NDVI_Max"] - df["L_NDVI_Min"]
    df["L_NDVI_Skew"] = ndvi.skew(axis=1).fillna(0)
    df["L_NDVI_Kurt"] = ndvi.kurt(axis=1).fillna(0)
    tempo = np.arange(len(l_ndvi_cols))
    slopes = []
    for row in ndvi.values:
        mask = ~np.isnan(row)
        if mask.sum() > 1: m, _ = np.polyfit(tempo[mask], row[mask], 1); slopes.append(m)
        else: slopes.append(0)
    df["L_NDVI_Slope"] = slopes
    peak = pd.Series(index=ndvi.index, dtype='object')
    linhas_validas = ndvi.count(axis=1) > 0
    if linhas_validas.any(): peak[linhas_validas] = ndvi[linhas_validas].idxmax(axis=1, skipna=True)
    df["L_NDVI_Peak_Time"] = peak.astype(str).str.extract(r"_t(\d+)").astype(float).fillna(0)
    print("    ✅ L_NDVI: Features extraídas")

# --- 3.2. Landsat NDMI ---
l_ndmi_cols = [c for c in df.columns if re.match(r"^L_NDMI_t\d+$", c)]
todas_cols_brutas.extend(l_ndmi_cols)
if len(l_ndmi_cols) > 0:
    ndmi = df[l_ndmi_cols]
    df["L_NDMI_Mean"] = ndmi.mean(axis=1).fillna(0)
    df["L_NDMI_Std"] = ndmi.std(axis=1).fillna(0)
    df["L_NDMI_Max"] = ndmi.max(axis=1).fillna(0)
    df["L_NDMI_Min"] = ndmi.min(axis=1).fillna(0)
    df["L_NDMI_Amp"] = df["L_NDMI_Max"] - df["L_NDMI_Min"]
    df["L_NDMI_Skew"] = ndmi.skew(axis=1).fillna(0)
    print("    ✅ L_NDMI: Features extraídas")

# --- 3.3. Landsat BSI ---
l_bsi_cols = [c for c in df.columns if re.match(r"^L_BSI_t\d+$", c)]
todas_cols_brutas.extend(l_bsi_cols)
if len(l_bsi_cols) > 0:
    bsi = df[l_bsi_cols]
    df["L_BSI_Mean"] = bsi.mean(axis=1).fillna(0)
    df["L_BSI_Std"] = bsi.std(axis=1).fillna(0)
    df["L_BSI_Max"] = bsi.max(axis=1).fillna(0)
    df["L_BSI_Min"] = bsi.min(axis=1).fillna(0)
    df["L_BSI_Amp"] = df["L_BSI_Max"] - df["L_BSI_Min"]
    df["L_BSI_Skew"] = bsi.skew(axis=1).fillna(0)
    print("    ✅ L_BSI (Solo Exposto): Features extraídas")

# --- 3.4. MODIS NDVI/EVI ---
modis_ndvi_cols = [c for c in df.columns if re.match(r"^MODIS_NDVI_t\d+$", c)]
modis_evi_cols = [c for c in df.columns if re.match(r"^MODIS_EVI_t\d+$", c)]
todas_cols_brutas.extend(modis_ndvi_cols + modis_evi_cols)
for col_group, nome in [(modis_ndvi_cols, "MODIS_NDVI"), (modis_evi_cols, "MODIS_EVI")]:
    if len(col_group) > 0:
        tmp = df[col_group]
        df[f"{nome}_Mean"] = tmp.mean(axis=1, skipna=True)
        df[f"{nome}_Std"] = tmp.std(axis=1, skipna=True).fillna(0)
        df[f"{nome}_Amp"] = tmp.max(axis=1, skipna=True) - tmp.min(axis=1, skipna=True)
print("    ✅ MODIS_NDVI/EVI: Features extraídas")

# --- 3.5. MODIS LST ---
modis_lst_day_cols = [c for c in df.columns if re.match(r"^MODIS_LST_day_t\d+$", c)]
modis_lst_night_cols = [c for c in df.columns if re.match(r"^MODIS_LST_night_t\d+$", c)]
todas_cols_brutas.extend(modis_lst_day_cols + modis_lst_night_cols)
for col_group, nome in [(modis_lst_day_cols, "MODIS_LST_day"), (modis_lst_night_cols, "MODIS_LST_night")]:
    if len(col_group) > 0:
        tmp = df[col_group]
        df[f"{nome}_Mean"] = tmp.mean(axis=1, skipna=True)
        df[f"{nome}_Std"] = tmp.std(axis=1, skipna=True).fillna(0)
        df[f"{nome}_Amp"] = tmp.max(axis=1, skipna=True) - tmp.min(axis=1, skipna=True)
print("    ✅ MODIS_LST: Features extraídas")

# --- 3.6. Clima e Landsat LST em Loop ---
prefixos_outros = ["L_LST_t", "ERA5_temp_t", "ERA5_precip_t", "ERA5_soil_moist_t"]
for pref in prefixos_outros:
    cols = [c for c in df.columns if c.startswith(pref) and not c.endswith("Mean")]
    todas_cols_brutas.extend(cols)
    if len(cols) > 0:
        tmp = df[cols]
        df[f"{pref}_Mean"] = tmp.mean(axis=1).fillna(0)
        df[f"{pref}_Std"] = tmp.std(axis=1).fillna(0)
        df[f"{pref}_Max"] = tmp.max(axis=1).fillna(0)
        df[f"{pref}_Min"] = tmp.min(axis=1).fillna(0)
        df[f"{pref}_Amp"] = df[f"{pref}_Max"] - df[f"{pref}_Min"]

# ==========================================================
# 4. FÓRMULAS AGRONÔMICAS COMPLEXAS
# ==========================================================
print("\n[4] Criando interações agronômicas...")
if "EC" in df.columns and "pH_H2O" in df.columns:
    df["RelacaoECpH"] = np.where(df["pH_H2O"] > 0, df["EC"] / df["pH_H2O"], 0)

if "ERA5_temp_t_Mean" in df.columns and "ERA5_precip_t_Mean" in df.columns:
    df["StressTermico"] = df["ERA5_temp_t_Mean"] / (df["ERA5_precip_t_Mean"] + 1.0)

if "L_NDVI_Mean" in df.columns:
    if "ERA5_temp_t_Mean" in df.columns: df["VigorClima"] = df["L_NDVI_Mean"] * df["ERA5_temp_t_Mean"]
    if "L_NDMI_Mean" in df.columns: df["Umidade_x_Vigor"] = df["L_NDMI_Mean"] * df["L_NDVI_Mean"]
    if "EC" in df.columns: df["EC_x_NDVI"] = df["EC"] * df["L_NDVI_Mean"]

if "L_NDMI_Mean" in df.columns:
    if "pH_H2O" in df.columns: df["pH_x_NDMI"] = df["pH_H2O"] * df["L_NDMI_Mean"]
    if "ERA5_precip_t_Mean" in df.columns: df["Chuva_x_NDMI"] = df["ERA5_precip_t_Mean"] * df["L_NDMI_Mean"]

if "L_BSI_Mean" in df.columns and "L_NDVI_Mean" in df.columns: df["SoloExposto_x_NDVI"] = df["L_BSI_Mean"] * df["L_NDVI_Mean"]
if "MODIS_LST_day_Mean" in df.columns and "MODIS_NDVI_Mean" in df.columns: df["Estresse_Termico_MODIS"] = df["MODIS_LST_day_Mean"] / (df["MODIS_NDVI_Mean"] + 0.001)
if "L_NDVI_Mean" in df.columns and "MODIS_NDVI_Mean" in df.columns: df["Razao_L_MODIS_NDVI"] = df["L_NDVI_Mean"] / (df["MODIS_NDVI_Mean"] + 0.001)

# ==========================================================
# 5. MATRIZ DE FEATURES E PCA
# ==========================================================
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, GroupKFold, cross_val_predict
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

print("\n[5] Montando Matriz, PCA e Split Treino/Teste (80/20)...")

colunas_excluir = set(["N","P","K","POINTID","id_original","system:index",".geo","pH_CaCl2","OC","CaCO3",
                       "Coarse", "Clay", "Sand", "Silt", "Revisited_point", "Soil_Stones", "LC1", "LU1",
                       "LC1_Desc", "LU1_Desc", "NUTS_0", "NUTS_1", "NUTS_2", "NUTS_3",
                       "NDVI_mean_temp", "Temp_mean_temp", "Chuva_mean_temp", "SG_N_profundo"] + todas_cols_brutas)

df_numerico = df.select_dtypes(include=[np.number])
features_base = [c for c in df_numerico.columns if c not in colunas_excluir and not df[c].isna().all()]

y_reg_original = df[["N", "P", "K"]].values
y_reg_log = np.log1p(y_reg_original)

idx_all = np.arange(len(df))
# Este Split é sagrado. O idx_test NUNCA será visto durante o treino.
idx_train, idx_test = train_test_split(idx_all, test_size=0.20, random_state=42)

# PCA L_NDVI
if len(l_ndvi_cols) > 0:
    imputer_pca_ndvi = SimpleImputer(strategy="mean")
    ndvi_train_imputed = imputer_pca_ndvi.fit_transform(df[l_ndvi_cols].iloc[idx_train])
    ndvi_all_imputed = imputer_pca_ndvi.transform(df[l_ndvi_cols])
    pca = PCA(n_components=8, random_state=42).fit(ndvi_train_imputed)
    emb = pca.transform(ndvi_all_imputed)
    for i in range(8): df[f"NDVI_embed_{i}"] = emb[:, i]
    features_base += [f"NDVI_embed_{i}" for i in range(8)]

# PCA L_NDMI
if len(l_ndmi_cols) > 0:
    imputer_pca_ndmi = SimpleImputer(strategy="mean")
    ndmi_train_imputed = imputer_pca_ndmi.fit_transform(df[l_ndmi_cols].iloc[idx_train])
    ndmi_all_imputed = imputer_pca_ndmi.transform(df[l_ndmi_cols])
    pca_ndmi = PCA(n_components=5, random_state=42).fit(ndmi_train_imputed)
    emb_ndmi = pca_ndmi.transform(ndmi_all_imputed)
    for i in range(5): df[f"NDMI_embed_{i}"] = emb_ndmi[:, i]
    features_base += [f"NDMI_embed_{i}" for i in range(5)]

# ==========================================================
# 6. CLASSES NPK, ZONAS GEOGRÁFICAS E IMPUTAÇÃO
# ==========================================================
y_train_clf, y_test_clf = np.full((len(idx_train), 3), -1, dtype=int), np.full((len(idx_test), 3), -1, dtype=int)

y_test_orig = y_reg_original[idx_test] # O GABARITO DO TESTE CEGO (Regressão)

for i, nut in enumerate(["N", "P", "K"]):
    st = df[nut].iloc[idx_train]
    _, bins = pd.qcut(st.rank(method='first'), q=3, labels=False, retbins=True)
    bins[0], bins[-1] = -np.inf, np.inf
    y_train_clf[:, i] = pd.cut(st.rank(method='first'), bins=bins, labels=[0, 1, 2]).astype(float).fillna(-1).astype(int).values
    
    st_teste = df[nut].iloc[idx_test]
    rk_teste = (st_teste.rank(method='first') / st_teste.rank(method='first').max() * st.rank(method='first').max())
    y_test_clf[:, i] = pd.cut(rk_teste, bins=bins, labels=[0, 1, 2]).astype(float).fillna(-1).astype(int).values # O GABARITO DO TESTE CEGO (Classificação)

geo_scaler = StandardScaler().fit(df[["TH_LAT", "TH_LONG"]].iloc[idx_train].fillna(0).values)
geo_scaled = geo_scaler.transform(df[["TH_LAT", "TH_LONG"]].fillna(0).values)
kmeans = KMeans(n_clusters=min(260, len(idx_train) // 40), random_state=42, n_init=10).fit(geo_scaled[idx_train])
zonas_all = kmeans.predict(geo_scaled)
zonas_train = zonas_all[idx_train]

zona_means_N = {}
for z in np.unique(zonas_train):
    mask_z = (zonas_train == z)
    if mask_z.sum() > 0: zona_means_N[z] = y_reg_original[idx_train][mask_z, 0].mean()
global_mean_N = np.nanmean(y_reg_original[idx_train, 0])
df["Zona_MeanN"] = [zona_means_N.get(z, global_mean_N) for z in zonas_all]
features_base.append("Zona_MeanN")

imputer = SimpleImputer(strategy="median")
X_train_full = imputer.fit_transform(df[features_base].iloc[idx_train])
X_test_full = imputer.transform(df[features_base].iloc[idx_test])

gkf = GroupKFold(n_splits=5)
print(f"    Features base: {len(features_base)} | Treino: {len(idx_train)} | Teste Cego: {len(idx_test)}")

# ==========================================================
# 7. FUNÇÕES PARA O LOOP (AVALIAÇÃO NO TESTE CEGO)
# ==========================================================
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, accuracy_score, f1_score
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, RandomForestClassifier, ExtraTreesClassifier
from xgboost import XGBRegressor, XGBClassifier
from sklearn.linear_model import Ridge
import lightgbm as lgb
from catboost import CatBoostRegressor, CatBoostClassifier
from sklearn.base import BaseEstimator, ClassifierMixin

class FlatCatBoostClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, iterations=500, learning_rate=0.03, depth=6, random_state=42, verbose=0):
        self.iterations, self.learning_rate, self.depth, self.random_state, self.verbose = iterations, learning_rate, depth, random_state, verbose
    def fit(self, X, y):
        self.model = CatBoostClassifier(iterations=self.iterations, learning_rate=self.learning_rate, depth=self.depth, random_state=self.random_state, verbose=self.verbose, thread_count=-1, allow_writing_files=False)
        self.model.fit(X, y)
        return self
    def predict(self, X): return self.model.predict(X).ravel()

def treinar_e_avaliar_teste_cego(features_selecionadas, iteracao_nome):
    print(f"\n⚙️  Iniciando etapa: {iteracao_nome} ({len(features_selecionadas)} features)")
    idx_cols = [features_base.index(f) for f in features_selecionadas]
    
    X_tr = X_train_full[:, idx_cols]
    X_te = X_test_full[:, idx_cols] # 💡 AQUI ESTÁ A MÁGICA: Usamos o X_test para a nota final
    
    resultados_iteracao = []
    
    # --- REGRESSÃO ---
    print("   [->] Modelos de REGRESSÃO (Avaliando no Teste Cego)...")
    modelos_reg = {
        "RandomForest": RandomForestRegressor(n_estimators=500, max_depth=6, random_state=42, n_jobs=-1),
        "ExtraTrees": ExtraTreesRegressor(n_estimators=500, max_depth=6, random_state=42, n_jobs=-1),
        "XGBoost": XGBRegressor(n_estimators=500, learning_rate=0.03, max_depth=6, random_state=42, n_jobs=-1),
        "LightGBM": lgb.LGBMRegressor(n_estimators=500, learning_rate=0.03, max_depth=6, num_leaves=63, random_state=42, verbose=-1, n_jobs=-1),
        "CatBoost": CatBoostRegressor(iterations=500, learning_rate=0.03, depth=6, random_state=42, verbose=0, thread_count=-1, allow_writing_files=False)
    }
    
    cat_treinados = [] 
    
    for nome, modelo in modelos_reg.items():
        for i, nut in enumerate(["N", "P", "K"]):
            y_nut_log_train = y_reg_log[idx_train, i]
            
            # Treina no X_train
            modelo.fit(X_tr, y_nut_log_train)
            
            # 💡 Previsão feita APENAS no X_test (Teste Cego)
            preds_log_test = modelo.predict(X_te)
            preds_orig_test = np.expm1(preds_log_test)
            
            # Avalia contra o y_test_orig (O Gabarito do cofre)
            r2 = r2_score(y_test_orig[:, i], preds_orig_test)
            mae = mean_absolute_error(y_test_orig[:, i], preds_orig_test)
            rmse = np.sqrt(mean_squared_error(y_test_orig[:, i], preds_orig_test))
            
            resultados_iteracao.append({"Filtro": iteracao_nome, "Modelo": f"Reg_{nome}", "Nutriente": nut, "R2": round(r2,3), "MAE": round(mae,2), "RMSE": round(rmse,2), "Acc": np.nan, "F1": np.nan})
            
            if nome == "CatBoost":
                cat_treinados.append(modelo)

    # --- CLASSIFICAÇÃO ---
    print("   [->] Modelos de CLASSIFICAÇÃO (Avaliando no Teste Cego)...")
    modelos_clf = {
        "RandomForest": RandomForestClassifier(n_estimators=500, max_depth=6, random_state=42, n_jobs=-1),
        "ExtraTrees": ExtraTreesClassifier(n_estimators=500, max_depth=6, random_state=42, n_jobs=-1),
        "XGBoost": XGBClassifier(n_estimators=500, learning_rate=0.03, max_depth=6, random_state=42, n_jobs=-1, verbosity=0, eval_metric="mlogloss"),
        "LightGBM": lgb.LGBMClassifier(n_estimators=500, learning_rate=0.03, max_depth=6, num_leaves=63, random_state=42, verbose=-1, n_jobs=-1),
        "CatBoost": FlatCatBoostClassifier(iterations=500, learning_rate=0.03, depth=6, random_state=42, verbose=0)
    }
    for nome, modelo in modelos_clf.items():
        for i, nut in enumerate(["N", "P", "K"]):
            y_nut_train = y_train_clf[:, i]
            
            # Treina no X_train
            modelo.fit(X_tr, y_nut_train)
            
            # 💡 Previsão feita APENAS no X_test (Teste Cego)
            preds_test = modelo.predict(X_te)
            
            # Avalia contra o y_test_clf (O Gabarito do cofre)
            acc = accuracy_score(y_test_clf[:, i], preds_test)
            f1 = f1_score(y_test_clf[:, i], preds_test, average='weighted')
            
            resultados_iteracao.append({"Filtro": iteracao_nome, "Modelo": f"Clf_{nome}", "Nutriente": nut, "R2": np.nan, "MAE": np.nan, "RMSE": np.nan, "Acc": round(acc,3), "F1": round(f1,3)})

    # --- STACKING ---
    print("   [->] STACKING (Avaliando no Teste Cego)...")
    modelos_stack = {
        "XGB": XGBRegressor(n_estimators=500, learning_rate=0.03, max_depth=6, random_state=42, n_jobs=-1),
        "LGB": lgb.LGBMRegressor(n_estimators=500, learning_rate=0.03, max_depth=6, num_leaves=63, random_state=42, verbose=-1, n_jobs=-1),
        "CAT": CatBoostRegressor(iterations=500, learning_rate=0.03, depth=6, random_state=42, verbose=0, thread_count=-1, allow_writing_files=False)
    }
    for nut_i, nut in enumerate(["N", "P", "K"]):
        y_nut_log_train = y_reg_log[idx_train, nut_i]
        
        cv_g = list(gkf.split(X_tr, y_nut_log_train, groups=zonas_train))
        oof_preds_nut = np.zeros((len(X_tr), len(modelos_stack)))
        preds_test_stack = np.zeros((len(X_te), len(modelos_stack)))
        
        for j, (nome_s, modelo_s) in enumerate(modelos_stack.items()):
            # 1. Cross_val para o Meta-Learner aprender
            oof_preds_nut[:, j] = cross_val_predict(modelo_s, X_tr, y_nut_log_train, cv=cv_g, n_jobs=-1)
            # 2. Treina na base toda
            m_final = type(modelo_s)(**modelo_s.get_params()).fit(X_tr, y_nut_log_train)
            # 3. Faz a previsão no X_test
            preds_test_stack[:, j] = m_final.predict(X_te)
            
        # Meta Learner Treina no OOF e Prevê no X_test_stack
        meta = Ridge(alpha=1.0).fit(oof_preds_nut, y_nut_log_train)
        preds_stack_log_test = meta.predict(preds_test_stack)
        preds_stack_orig_test = np.expm1(preds_stack_log_test)
        
        # 💡 Avalia contra o y_test_orig
        r2 = r2_score(y_test_orig[:, nut_i], preds_stack_orig_test)
        mae = mean_absolute_error(y_test_orig[:, nut_i], preds_stack_orig_test)
        rmse = np.sqrt(mean_squared_error(y_test_orig[:, nut_i], preds_stack_orig_test))
        
        resultados_iteracao.append({"Filtro": iteracao_nome, "Modelo": "Reg_Stacking", "Nutriente": nut, "R2": round(r2,3), "MAE": round(mae,2), "RMSE": round(rmse,2), "Acc": np.nan, "F1": np.nan})
            
    return pd.DataFrame(resultados_iteracao), cat_treinados

def calcular_shap_global(modelos_cat, features_selecionadas):
    print("   🔍 Calculando SHAP para extrair os próximos TOPs...")
    idx_cols = [features_base.index(f) for f in features_selecionadas]
    X_shap = X_train_full[np.random.default_rng(42).choice(len(X_train_full), size=min(800, len(X_train_full)), replace=False)][:, idx_cols]
    
    importancia_global = np.zeros(len(features_selecionadas))
    for modelo in modelos_cat: # N, P, K
        explainer = shap.TreeExplainer(modelo)
        importancia_global += np.abs(explainer.shap_values(X_shap)).mean(axis=0)
        
    df_imp = pd.DataFrame({"Feature": features_selecionadas, "Importance": importancia_global / 3})
    return df_imp.sort_values("Importance", ascending=False)

# ==========================================================
# 8. PIPELINE DE ELIMINAÇÃO RECURSIVA (TESTE CEGO + SHAP)
# ==========================================================
print("\n" + "=" * 80)
print("🌀 INICIANDO LOOP DE RFE-SHAP COM AVALIAÇÃO EM TESTE CEGO")
print("=" * 80)

etapas_rfe = [
    ("Todas as Features", len(features_base), 50),
    ("Top 50 Features", 50, 25),
    ("Top 25 Features", 25, 10),
    ("Top 10 Features", 10, 5),
    ("Top 5 Features", 5, 0)
]

features_atuais = features_base.copy()
tabela_geral_resultados = pd.DataFrame()

for iteracao_nome, num_features_atual, proximo_corte in etapas_rfe:
    
    # Treina em X_train e Tira a Nota em X_test (Teste Cego Real)
    df_res, cat_treinados = treinar_e_avaliar_teste_cego(features_atuais, iteracao_nome)
    tabela_geral_resultados = pd.concat([tabela_geral_resultados, df_res], ignore_index=True)
    
    if proximo_corte > 0:
        df_ranking_shap = calcular_shap_global(cat_treinados, features_atuais)
        print(f"\n👑 TOP {proximo_corte} EXTRAÍDO PELA SHAP NA ITERAÇÃO '{iteracao_nome}':")
        print(df_ranking_shap.head(proximo_corte).to_string(index=False))
        features_atuais = df_ranking_shap["Feature"].head(proximo_corte).tolist()

# ==========================================================
# 9. EXIBIÇÃO DA TABELA FINAL MESTRA
# ==========================================================
print("\n" + "=" * 80)
print("🏆 TABELA MESTRA DE RESULTADOS (NOTAS DO TESTE CEGO)")
print("=" * 80)

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

print(tabela_geral_resultados.to_string(index=False))

# Exporta tudo
tabela_geral_resultados.to_csv("TCC_Resultados_RFE_TesteCego.csv", index=False, sep=";", decimal=",")
print("\n✅ Tabela exportada com sucesso para: 'TCC_Resultados_RFE_TesteCego.csv'")
print("=" * 80)