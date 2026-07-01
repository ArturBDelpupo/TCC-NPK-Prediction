# Dataset

This directory contains the datasets used in this research or the instructions required to recreate them.

## Option 1 – Build the Dataset from Scratch

If you would like to reproduce the complete dataset used in this research, follow the steps below.

### Step 1 – Request Access to the LUCAS Topsoil 2015 Dataset

The base dataset is the **LUCAS Topsoil 2015** survey provided by the European Commission.

Before proceeding, request access through the official website:

https://esdac.jrc.ec.europa.eu/content/lucas2015-topsoil-data

Once your request has been approved, download the dataset and place it in the appropriate directory.

---

### Step 2 – Extract and Enrich the Data

Run the following scripts **in the exact order shown below**:

```text
1. gee_extractor_lucas2015.py
2. enricher_soilgrids.py
3. juntos_teste_dataset.py
```

Each script performs a specific stage of the dataset construction:

| Script                       | Purpose                                                                                              |
| ---------------------------- | ---------------------------------------------------------------------------------------------------- |
| `gee_extractor_lucas2015.py` | Extracts satellite variables from Google Earth Engine (vegetation indices, climate variables, etc.). |
| `enricher_soilgrids.py`      | Enriches each sample with soil properties obtained from SoilGrids.                                   |
| `juntos_teste_dataset.py`    | Merges all extracted information into the final machine learning dataset.                            |

After executing these three scripts, the final dataset will be generated as a **CSV file** of approximately **200 MB**, containing all integrated features used throughout this research.

Because the final CSV file is approximately **200 MB**, it could not be included in this repository due to **GitHub's file size limitations**. If you prefer not to recreate the dataset, you may request access to the processed dataset as described below.

---

## Option 2 – Request the Ready-to-Use Dataset

If you prefer not to rebuild the dataset from scratch, you may request access to the complete processed dataset through the following link:

https://drive.google.com/file/d/1ef3T0Nn4_CrM7ZOKqGZ8GOUPwyTKH00o/view?usp=sharing

> **Note:** Access to the dataset is subject to approval by the repository owner and may not be granted immediately.

---

## Support

If you have any questions regarding the dataset, the methodology, or the repository, please contact:

**Artur B. Delpupo**
**Email:** [arturdelpupo.pt@gmail.com](mailto:arturdelpupo.pt@gmail.com)
