# Material deprivation of Indigenous peoples in Colombia by territory (CNPV 2018): analysis code

This repository contains the analysis code used to measure the material deprivation of Indigenous and
non-Indigenous persons in Colombia with the microdata of the 2018 National Population and Housing Census
(Censo Nacional de Población y Vivienda, CNPV 2018), between and within municipalities, types of territory and
Indigenous resguardos.

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
| 01 | `01_build_national.py` | Links dwellings, households and persons of the 33 departments (private households; special accommodation facilities excluded) and builds the indicators | `data/processed/households.parquet`, `indigenous.parquet`, `provenance.json` |
| 02 | `02_national_analysis.py` | Analytical population and missing indicators, prevalences by type of territory, decomposition of inequality, disparity models with fixed effects for place, robustness checks and comparison of resguardos with their surroundings | `results/population_flow.csv`, `missing.csv`, `prevalence.csv`, `decomposition.csv`, `gap_models.csv`, `gap_contrasts.csv`, `robustness.csv`, `support.csv`, `resguardos.csv`, `resguardos_cesar.csv`, `pueblos.csv`, `surroundings.csv`, `summary.json` |

Step 01 takes about half an hour and step 02 about twenty minutes on a desktop computer with 32 GB of RAM.

## Definitions

- Population: persons in private households (43,835,324 persons in 14,243,223 households, the official totals);
  the 329,093 residents of special accommodation facilities (lugares especiales de alojamiento) are excluded.
  Indigenous persons are those who self-identified as Indigenous in the census. The comparison group comprises
  non-Indigenous persons with a reported ethnicity; persons whose ethnicity was not reported are left out of the
  comparisons. Peoples: the 115 native peoples are separated from the categories of Indigenous peoples of other
  countries and the category without information on the people.
- Type of territory: municipal head; populated centre; rural dispersed area outside a resguardo; Indigenous
  resguardo. The census asks the question on ethnic territories only outside municipal heads.
- Indicators (household level, assigned to its members): no electricity; unimproved water source for preparing food
  (WHO/UNICEF JMP classification: wells without a pump and surface water; a strict variant also counts rainwater,
  water trucks and water carriers); unimproved sanitation (toilet without connection, direct discharge or none);
  cooking in a room also used for sleeping; critical overcrowding (more than three persons per sleeping room).
  Dirt floors and walls of cane, waste or no walls are used only in a sensitivity analysis.
- Deprivation score: share of the observed indicators in which the household is deprived (equal weights), for
  households with at least three observed indicators. Multiple deprivation: deprived in at least 40% of the
  observed indicators (two of five when all are observed).
- Decomposition: Theil index and generalised entropy index with parameter two, split exactly into between- and
  within-group components, for partitions by department, municipality, municipality and area, type of territory,
  detailed territory, people, people by type of territory and municipality by people, and for several versions of
  the index.
- Disparity models: linear models of the household score on an Indigenous household head, weighted by household
  size, with fixed effects for the municipality by area or by type of territory, household controls (education, sex
  and age group of the head, household size), interactions with the type of territory, Wald tests and standard
  errors clustered by municipal unit. Robustness: alternative indices, complete information, individual
  self-identification, unweighted households, strata with both groups, heads without ethnic affiliation as
  comparator and fixed effects for each resguardo.
- Groups with fewer than 30 persons are not reported.
