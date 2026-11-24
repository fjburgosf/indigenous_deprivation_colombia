# Material deprivation of Indigenous peoples in Colombia by territory (CNPV 2018): analysis code

This repository contains the analysis code used to measure the material deprivation of Indigenous and
non-Indigenous persons in Colombia with the microdata of the 2018 National Population and Housing Census
(Censo Nacional de Población y Vivienda, CNPV 2018), and to separate its place and ethnic components.

The repository includes only the analysis scripts. The census microdata are not redistributed.

## Data

The microdata of the CNPV 2018 are public and are distributed by the National Administrative Department of
Statistics (DANE) through its microdata catalogue (https://microdatos.dane.gov.co/). Download, for the 33
departments, the departmental files with the CSV version of the microdata (for example `05_Antioquia.zip`, which
contains `05_Antioquia_CSV.zip`), and the DDI metadata file of the census, and place them as follows:

```
data/raw/CNPV_2018/NN_Department.zip   (33 files)
data/raw/cnpv2018.ddi.xml
```

Step 01 writes the SHA-256 of every departmental file in `data/processed/provenance.json`.

## Installation

Python 3.12 was used.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

## Running the analysis

Run the scripts from the repository root, in this order:

```bash
python src/01_build_national.py
python src/02_national_analysis.py
```

| Step | Script | What it does | Output |
|---|---|---|---|
| 01 | `01_build_national.py` | Links dwellings, households and persons of the 33 departments and builds the indicators | `data/processed/households.parquet`, `indigenous.parquet`, `provenance.json` |
| 02 | `02_national_analysis.py` | Prevalences by type of territory, decomposition of inequality, gap models with fixed effects for place and comparison of resguardos with their surroundings | `results/prevalence.csv`, `decomposition.csv`, `gap_models.csv`, `resguardos.csv`, `resguardos_cesar.csv`, `pueblos.csv`, `surroundings.csv`, `summary.json` |

Step 01 takes several minutes and step 02 a few minutes on a desktop computer with 16 GB of RAM or more.

## Definitions

- Population: persons in private households, linked to their household and dwelling. Indigenous persons are those
  who self-identified as Indigenous in the census.
- Type of territory: municipal head; populated centre; rural dispersed area outside a resguardo; Indigenous
  resguardo. The census asks the question on ethnic territories only outside municipal heads.
- Indicators (household level, assigned to its members): no electricity; unimproved water source for cooking;
  unimproved sanitation; cooking in a room also used for sleeping; critical overcrowding (more than three persons
  per sleeping room). Dirt floors and walls of cane, waste or no walls are used only in a sensitivity analysis.
- Deprivation score: share of the five indicators in which the household is deprived (equal weights; indicators
  with missing values are excluded from the denominator). Multiple deprivation: two or more indicators.
- Decomposition: Theil index and generalised entropy index with parameter two, split exactly into between- and
  within-group components, for partitions by department, municipality, municipality and area, type of territory,
  detailed territory, people and people by type of territory.
- Gap models: linear models of the household score on an Indigenous household head, weighted by household size,
  with fixed effects for the municipality by area or by type of territory, household controls (education, sex and
  age group of the head, household size) and standard errors clustered by municipality.
- Groups with fewer than 30 persons are not reported.
