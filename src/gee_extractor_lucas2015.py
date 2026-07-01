# ==========================================================
# LUCAS SOIL — MULTI-SOURCE EXTRACTOR (V3.0)
# ==========================================================
# Melhorias:
# 1. MODIS NDVI/EVI como complemento/fallback do Sentinel-2
# 2. ERA5 precipitação como alternativa ao CHIRPS
# 3. MODIS LST como complemento do Landsat
# 4. Janela temporal expansiva para reduzir gaps
# ==========================================================

import ee
import pandas as pd
from datetime import datetime, timedelta
import re
import io

ee.Initialize(project='projeto-lucas-tcc-ic')

# ----------------------------------------------------------
# 1. Carregar Dataset Base
# ----------------------------------------------------------
print("[1] Carregando dataset LUCAS...")
with open("vilarinho_Soil_1.csv", "r", encoding="utf-8") as f:
    linhas = f.readlines()
linhas_ok = []
for i, l in enumerate(linhas):
    l = re.sub(r";+$", "", l)
    l = l.replace('""', '"')
    if i > 0 and l.startswith('"0-20 cm'):
        l = l.replace('"', '', 1)
    linhas_ok.append(l)

df = pd.read_csv(io.StringIO("".join(linhas_ok)), sep=",", engine="python", on_bad_lines="skip")
df = df.loc[:, ~df.columns.duplicated()]

df["POINTID"] = pd.to_numeric(df["POINTID"], errors="coerce")
df = df.dropna(subset=["TH_LAT", "TH_LONG", "POINTID"]).reset_index(drop=True)

# ----------------------------------------------------------
# 2. Camadas Estáticas (Solo)
# ----------------------------------------------------------
print("[2] Configurando camadas estáticas (OLM e SoilGrids)...")
olm_clay = ee.Image("OpenLandMap/SOL/SOL_CLAY-WFRACTION_USDA-3A1A1A_M/v02").select('b0').rename('OLM_Clay_pct')
olm_sand = ee.Image("OpenLandMap/SOL/SOL_SAND-WFRACTION_USDA-3A1A1A_M/v02").select('b0').rename('OLM_Sand_pct')
olm_soc = ee.Image("OpenLandMap/SOL/SOL_ORGANIC-CARBON_USDA-6A1C_M/v02").select('b0').rename('OLM_SOC')
olm_bulk = ee.Image("OpenLandMap/SOL/SOL_BULKDENS-FINEEARTH_USDA-4A1H_M/v02").select('b0').rename('OLM_BulkDens')
olm_texture = ee.Image("OpenLandMap/SOL/SOL_TEXTURE-CLASS_USDA-TT_M/v02").select('b0').rename('OLM_TextureClass')

soil_clay = ee.Image("projects/soilgrids-isric/clay_mean").select('clay_15-30cm_mean').rename('SG_Clay') #Argila
soil_sand = ee.Image("projects/soilgrids-isric/sand_mean").select('sand_15-30cm_mean').rename('SG_Sand') #Areia
soil_soc  = ee.Image("projects/soilgrids-isric/soc_mean").select('soc_15-30cm_mean').rename('SG_SOC') # Carbono orgânico (Soil organic carbon)

camadas_estaticas = (olm_clay.addBands([olm_sand, olm_soc, olm_bulk, olm_texture])
                     .addBands([soil_clay, soil_sand, soil_soc]))

# ----------------------------------------------------------
# 3. Funções Auxiliares
# ----------------------------------------------------------
def gerar_datas_bimestrais_10anos(data_base_str):
    """
    Gera 60 datas (10 anos) com intervalos de 60 dias para trás,
    a partir da data de amostragem do LUCAS.
    """
    data_alvo = datetime.strptime(data_base_str, "%d-%m-%y")
    datas_bimestrais = []
    
    # 60 períodos de 2 meses = 10 anos
    for _ in range(60):
        datas_bimestrais.append(data_alvo)
        data_alvo = data_alvo - timedelta(days=60)
        
    return datas_bimestrais

# ==========================================================
# NOVAS FUNÇÕES - FONTES ALTERNATIVAS
# ==========================================================

def extrair_modis_ndvi(roi, inicio, fim, sufixo):
    """
    Extrai NDVI e EVI do MODIS (250m, composição 16 dias).
    MODIS passa quase diariamente e tem MUITO menos gaps que Sentinel-2.
    """
    modis = (ee.ImageCollection("MODIS/061/MOD13Q1")
             .filterBounds(roi)
             .filterDate(inicio, fim)
             .select(['NDVI', 'EVI']))

    count = modis.size()

    def com_dados():
        # NDVI e EVI do MODIS vêm escalados por 0.0001
        ndvi = modis.select('NDVI').mean().multiply(0.0001).rename(f'MODIS_NDVI_{sufixo}')
        evi = modis.select('EVI').mean().multiply(0.0001).rename(f'MODIS_EVI_{sufixo}')
        return ndvi.addBands(evi).unmask(-9999)

    def sem_dados():
        return ee.Image.constant([-9999.0, -9999.0]).rename([f'MODIS_NDVI_{sufixo}', f'MODIS_EVI_{sufixo}'])

    return ee.Image(ee.Algorithms.If(count.gt(0), com_dados(), sem_dados()))


def extrair_modis_lst(roi, inicio, fim, sufixo, max_dias_expansao=90):
    """
    Extrai temperatura de superfície do MODIS (1km, diário).
    Alternativa/complemento ao Landsat LST.
    """
    # MOD11A1 - LST diário
    modis_day = (ee.ImageCollection("MODIS/061/MOD11A1")
                 .filterBounds(roi)
                 .filterDate(inicio, fim)
                 .select('LST_Day_1km'))

    # MYD11A1 - LST diário (Aqua, para mais cobertura)
    modis_night = (ee.ImageCollection("MODIS/061/MYD11A1")
                   .filterBounds(roi)
                   .filterDate(inicio, fim)
                   .select('LST_Day_1km'))

    count = modis_day.size().add(modis_night.size())

    def com_dados():
        # Escala: 0.02K -> converter para Celsius
        lst_day = modis_day.mean().multiply(0.02).subtract(273.15).rename(f'MODIS_LST_day_{sufixo}')
        lst_night = modis_night.mean().multiply(0.02).subtract(273.15).rename(f'MODIS_LST_night_{sufixo}')
        return lst_day.addBands(lst_night).unmask(-9999)

    def sem_dados():
        return ee.Image.constant([-9999.0, -9999.0]).rename([f'MODIS_LST_day_{sufixo}', f'MODIS_LST_night_{sufixo}'])

    img = ee.Image(ee.Algorithms.If(count.gt(0), com_dados(), sem_dados()))

    # Expansão se vazio
    lst_valido = img.select(f'MODIS_LST_day_{sufixo}').neq(-9999)
    novo_inicio = inicio.advance(-max_dias_expansao, "day")
    modis_exp = (ee.ImageCollection("MODIS/061/MOD11A1")
                 .filterBounds(roi)
                 .filterDate(novo_inicio, fim)
                 .select('LST_Day_1km'))
    img_exp = modis_exp.mean().multiply(0.02).subtract(273.15).rename(f'MODIS_LST_day_{sufixo}')
    img_exp = img_exp.addBands(img.select(f'MODIS_LST_night_{sufixo}'))

    return ee.Image(ee.Algorithms.If(lst_valido, img, img_exp))


def extrair_era5_precip_completo(roi, inicio, fim, sufixo):
    """
    ERA5 Land - Precipitação com cobertura global completa.
    Usar como alternativa principal ao CHIRPS.
    """
    era5 = (ee.ImageCollection("ECMWF/ERA5_LAND/DAILY_AGGR")
            .filterBounds(roi)
            .filterDate(inicio, fim))

    count = era5.size()

    def com_dados():
        precip = era5.select('total_precipitation_sum').sum().rename(f'ERA5_precip_{sufixo}')
        temp = era5.select('temperature_2m').mean().subtract(273.15).rename(f'ERA5_temp_{sufixo}')
        soil_moist = era5.select('volumetric_soil_water_layer_1').mean().rename(f'ERA5_soil_moist_{sufixo}')
        return precip.addBands([temp, soil_moist]).unmask(-9999)

    def sem_dados():
        return ee.Image.constant([-9999.0, -9999.0, -9999.0]).rename([f'ERA5_precip_{sufixo}', f'ERA5_temp_{sufixo}', f'ERA5_soil_moist_{sufixo}'])

    return ee.Image(ee.Algorithms.If(count.gt(0), com_dados(), sem_dados()))


def extrair_landsat_robusto(roi, inicio, fim, sufixo, max_dias_expansao=90):
    """
    Extrai Landsat harmonizando L5, L7 e L8.
    Calcula LST, NDVI, NDMI e BSI.
    """
    def prep_landsat(img, sensor):
        # 1. Aplicar fator de escala de refletância de superfície
        fator_mult = 0.0000275
        fator_add = -0.2
        
        # 2. Harmonizar bandas dependendo do sensor
        if sensor == 'L8':
            blue  = img.select('SR_B2').multiply(fator_mult).add(fator_add)
            red   = img.select('SR_B4').multiply(fator_mult).add(fator_add)
            nir   = img.select('SR_B5').multiply(fator_mult).add(fator_add)
            swir1 = img.select('SR_B6').multiply(fator_mult).add(fator_add)
            swir2 = img.select('SR_B7').multiply(fator_mult).add(fator_add)
            lst   = img.select('ST_B10').multiply(0.00341802).add(149.0).subtract(273.15)
        else: # Landsat 5 e 7
            blue  = img.select('SR_B1').multiply(fator_mult).add(fator_add)
            red   = img.select('SR_B3').multiply(fator_mult).add(fator_add)
            nir   = img.select('SR_B4').multiply(fator_mult).add(fator_add)
            swir1 = img.select('SR_B5').multiply(fator_mult).add(fator_add)
            swir2 = img.select('SR_B7').multiply(fator_mult).add(fator_add)
            lst   = img.select('ST_B6').multiply(0.00341802).add(149.0).subtract(273.15)

        # 3. Calcular Índices
        ndvi = nir.subtract(red).divide(nir.add(red)).rename(f'L_NDVI_{sufixo}')
        ndmi = nir.subtract(swir1).divide(nir.add(swir1)).rename(f'L_NDMI_{sufixo}')
        
        # BSI - Bare Soil Index: ((SWIR1+Red)-(NIR+Blue)) / ((SWIR1+Red)+(NIR+Blue))
        bsi_num = swir1.add(red).subtract(nir.add(blue))
        bsi_den = swir1.add(red).add(nir).add(blue)
        bsi = bsi_num.divide(bsi_den).rename(f'L_BSI_{sufixo}')
        
        lst = lst.rename(f'L_LST_{sufixo}')

        return img.addBands([lst, ndvi, ndmi, bsi]).select([f'L_LST_{sufixo}', f'L_NDVI_{sufixo}', f'L_NDMI_{sufixo}', f'L_BSI_{sufixo}'])

    def extrair_l(inicio_janela, fim_janela):
        # Filtro rigoroso de nuvens (abaixo de 40%) na imagem inteira para garantir qualidade
        l8_col = ee.ImageCollection("LANDSAT/LC08/C02/T1_L2").filterBounds(roi).filterDate(inicio_janela, fim_janela).filter(ee.Filter.lt('CLOUD_COVER', 40)).map(lambda img: prep_landsat(img, 'L8'))
        l7_col = ee.ImageCollection("LANDSAT/LE07/C02/T1_L2").filterBounds(roi).filterDate(inicio_janela, fim_janela).filter(ee.Filter.lt('CLOUD_COVER', 40)).map(lambda img: prep_landsat(img, 'L7'))
        l5_col = ee.ImageCollection("LANDSAT/LT05/C02/T1_L2").filterBounds(roi).filterDate(inicio_janela, fim_janela).filter(ee.Filter.lt('CLOUD_COVER', 40)).map(lambda img: prep_landsat(img, 'L5'))
        
        merged_col = l8_col.merge(l7_col).merge(l5_col)

        count = merged_col.size()

        def com_dados():
            return merged_col.median().unmask(-9999)

        def sem_dados():
            return ee.Image.constant([-9999.0, -9999.0, -9999.0, -9999.0]).rename([f'L_LST_{sufixo}', f'L_NDVI_{sufixo}', f'L_NDMI_{sufixo}', f'L_BSI_{sufixo}'])

        return ee.Image(ee.Algorithms.If(count.gt(0), com_dados(), sem_dados()))

    img_original = extrair_l(inicio, fim)
    ndvi_valido = img_original.select(f'L_NDVI_{sufixo}').neq(-9999)

    novo_inicio = inicio.advance(-max_dias_expansao, "day")
    img_expandida = extrair_l(novo_inicio, fim)

    return ee.Image(ee.Algorithms.If(ndvi_valido, img_original, img_expandida))

def extrair_landsat_com_modis_fallback(roi, inicio, fim, sufixo, max_dias_expansao=180):
    """
    Extrai Landsat com expansão + MODIS LST como fallback/complemento.
    """
    # Extrair Landsat
    img_landsat = extrair_landsat_robusto(roi, inicio, fim, sufixo, max_dias_expansao)

    # SEMPRE extrair MODIS LST como complemento
    img_modis_lst = extrair_modis_lst(roi, inicio, fim, sufixo, max_dias_expansao=90)

    return img_landsat.addBands(img_modis_lst)


# ----------------------------------------------------------
# 4. LOOP DE PROCESSAMENTO - MÚLTIPLOS LOTES SIMULTÂNEOS
# ----------------------------------------------------------
TAMANHO_LOTE = 500  # 💡 Mantemos 500 para garantir que a Google não cancele por falta de memória

# 👈 DEFINA AQUI QUAIS LOTES VOCÊ QUER ENVIAR PARA RODAR AO MESMO TEMPO
LOTES_PARA_RODAR = [0,1] 

# Descobre quantas linhas sobraram DEPOIS de remover as que não tinham coordenadas
total_linhas = len(df)
import math
num_lotes = math.ceil(total_linhas / TAMANHO_LOTE)

print(f"\n[3] Preparando envio simultâneo. Total de pontos válidos: {total_linhas}")
print(f"    Total de lotes possíveis: {num_lotes} (Tamanho: {TAMANHO_LOTE} pts/lote)")
print("=" * 70)

for lote_num in LOTES_PARA_RODAR:
    if lote_num < 1 or lote_num > num_lotes:
        print(f"❌ Lote {lote_num} ignorado (fora dos limites).")
        continue

    start_idx = (lote_num - 1) * TAMANHO_LOTE
    end_idx = min(start_idx + TAMANHO_LOTE, total_linhas) 
    df_subset = df.iloc[start_idx:end_idx]

    if df_subset.empty:
        continue

    print(f"🚀 Disparando Tarefa: LOTE {lote_num}/{num_lotes} (Linhas {start_idx} a {end_idx-1})... ", end="")

    features = []
    for i, row in df_subset.iterrows():
        point = ee.Geometry.Point([float(row["TH_LONG"]), float(row["TH_LAT"])])
        feat = ee.Feature(point, {"id_original": int(i), "POINTID": int(row["POINTID"])})
        features.append(feat)

    if not features:
        print("[VAZIO - PULANDO]")
        continue

    fc = ee.FeatureCollection(features)
    area_estudo = fc.geometry()

    datas_bim = gerar_datas_bimestrais_10anos("15-06-15") 
    images = []

    for i, d in enumerate(datas_bim):
        data_ref = ee.Date(d.strftime("%Y-%m-%d"))
        inicio_busca = data_ref.advance(-10, "day")
        sufixo = f"t{i+1}"

        l_bands = extrair_landsat_com_modis_fallback(area_estudo, inicio_busca, data_ref, sufixo, max_dias_expansao=60)
        era5_bands = extrair_era5_precip_completo(area_estudo, inicio_busca, data_ref, sufixo)
        modis_ndvi_bands = extrair_modis_ndvi(area_estudo, inicio_busca, data_ref, sufixo)

        images.append(l_bands.addBands(era5_bands).addBands(modis_ndvi_bands))

    stack_temporal = ee.Image.cat(images)
    stack_final = stack_temporal.addBands(camadas_estaticas)

    dataset = stack_final.sampleRegions(collection=fc, scale=30, geometries=False, tileScale=16)

    nome_arquivo = f"valirinho_Lote_{lote_num}_de_{num_lotes}_LANDSAT_10ANOS"

    task = ee.batch.Export.table.toDrive(
        collection=dataset,
        description=nome_arquivo,
        fileFormat="CSV"
    )
    
    # É ESTE COMANDO QUE FAZ O PARALELISMO NA NUVEM!
    task.start() 
    print("[ENVIADO ✅]")

print("\n" + "=" * 70)
print(f"🚀 TODAS AS {len(LOTES_PARA_RODAR)} TAREFAS FORAM DISPARADAS PARA A GOOGLE!")
print("Acesse https://code.earthengine.google.com/ para aprovar as tarefas na aba 'Tasks'.")
print("Elas rodarão simultaneamente nos servidores deles.")
print("=" * 70)