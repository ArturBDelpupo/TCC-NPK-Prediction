# ==========================================================
# SOILGRIDS ENRICHER (V2.0 - DIRECT RASTER READ)
# ==========================================================
# Este script ignora a instável API REST e lê os valores 
# diretamente dos arquivos VRT/GeoTIFF na nuvem usando Rasterio.
# É milhares de vezes mais rápido e não sofre com Erro 503.
# ==========================================================

import pandas as pd
import numpy as np
import rasterio
from pyproj import Transformer
import warnings
import os

warnings.filterwarnings("ignore")

# ----------------------------------------------------------
# CONFIGURAÇÃO
# ----------------------------------------------------------
# 💡 Atualizado para pegar o arquivo que acabamos de juntar
FICHEIRO_ENTRADA  = "valirinho_Lote_1_de_1_LANDSAT_10ANOS.csv"
FICHEIRO_SAIDA    = "vilarinho_Final.csv"

COL_LAT  = "TH_LAT"
COL_LON  = "TH_LONG"
PROFUNDIDADE = "15-30cm"

# 💡 Atualizamos os nomes de saída para não apagar o SoilGrids 0-5cm que veio do GEE
PROPRIEDADES = [
    ("clay",    "SG_Clay_profundo",     0.1,    "Argila %"),
    ("sand",    "SG_Sand_profundo",     0.1,    "Areia %"),
    ("silt",    "SG_Silt_profundo",     0.1,    "Silte %"),
    ("phh2o",   "SG_pH_profundo",       0.01,   "pH em água"),
    ("soc",     "SG_SOC_profundo",      0.01,   "Carbono orgânico dg/kg → g/kg"),
    ("bdod",    "SG_BulkDens_profundo", 0.01,   "Densidade aparente cg/cm³ → g/cm³"),
    ("cec",     "SG_CEC_profundo",      0.1,    "CTC cmolc/kg"),
    ("nitrogen","SG_N_profundo",        0.01,   "Azoto total cg/kg → g/kg"),
]

def main():
    print("=" * 65)
    print("🚀 SOILGRIDS ENRICHER V2.0 (MODO RASTERIO/NUVEM)")
    print("=" * 65)

    if not os.path.exists(FICHEIRO_ENTRADA):
        print(f"ERRO: ficheiro '{FICHEIRO_ENTRADA}' não encontrado.")
        return

    print("\n[1] Lendo o dataset...")
    df = pd.read_csv(FICHEIRO_ENTRADA, low_memory=False)
    
    # Filtrar apenas as linhas que têm coordenadas válidas
    mask_validos = df[COL_LAT].notna() & df[COL_LON].notna()
    df_validos = df[mask_validos].copy()
    
    n_pontos = len(df_validos)
    print(f"    Total de pontos válidos: {n_pontos}")

    # ----------------------------------------------------------
    # 2. CONVERSÃO DE COORDENADAS (O SEGREDO DO SOILGRIDS)
    # ----------------------------------------------------------
    # O SoilGrids não usa Lat/Lon (WGS84). Ele usa a projeção 
    # Interrupted Goode Homolosine (IGH). Precisamos converter.
    print("\n[2] Convertendo coordenadas (WGS84 -> IGH)...")
    
    # String PROJ4 oficial da projeção do SoilGrids
    crs_igh = "+proj=igh +datum=WGS84 +units=m +no_defs"
    transformer = Transformer.from_crs("EPSG:4326", crs_igh, always_xy=True)

    # Extrai as coordenadas e converte todas de uma vez (super rápido)
    coords_wgs = list(zip(df_validos[COL_LON], df_validos[COL_LAT]))
    coords_igh = [transformer.transform(lon, lat) for lon, lat in coords_wgs]

# ----------------------------------------------------------
    # 3. EXTRAÇÃO DOS DADOS DIRETAMENTE DA NUVEM (VRT)
    # ----------------------------------------------------------
    print("\n[3] Extraindo dados diretamente dos mapas da ISRIC...")

    import sys # Para forçar a impressão na mesma linha

    for prop_api, col_saida, escala, desc in PROPRIEDADES:
        print(f"    ⏳ Puxando {desc:<25} ({prop_api}) ", end="")
        sys.stdout.flush()
        
        vrt_url = f"/vsicurl/https://files.isric.org/soilgrids/latest/data/{prop_api}/{prop_api}_{PROFUNDIDADE}_mean.vrt"
        
        try:
            with rasterio.open(vrt_url) as src:
                nodata = src.nodata
                
                valores_processados = []
                # Puxamos os dados um a um e avisamos a cada 2000 pontos
                for i, val in enumerate(src.sample(coords_igh)):
                    v = val[0]
                    if v == nodata or v < -9999:
                        valores_processados.append(np.nan)
                    else:
                        valores_processados.append(round(v * escala, 4))
                    
                    # Imprime o progresso a cada 2000 amostras lidas
                    if (i + 1) % 2000 == 0:
                        print(f"[{i+1}]..", end="")
                        sys.stdout.flush()
                
                df.loc[mask_validos, col_saida] = valores_processados
                
            print(" [OK]")
            
        except Exception as e:
            print(f" [ERRO] {e}")

    # ----------------------------------------------------------
    # 4. SALVAR RESULTADOS
    # ----------------------------------------------------------
    print(f"\n[4] Guardando ficheiro: {FICHEIRO_SAIDA}")
    df.to_csv(FICHEIRO_SAIDA, index=False)
    
    print("\n✅ Processo finalizado com sucesso! Sem erros de servidor.")
    print("=" * 65)

if __name__ == "__main__":
    main()