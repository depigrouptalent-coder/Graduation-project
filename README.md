# Data Analytics Final Project

├── data/
│   ├── external/       # External datasets (e.g., World Bank SDG indicators, weather APIs)
│   ├── interim/        # Intermediate data (partially cleaned or joined)
│   ├── processed/      # Final, canonical datasets used for the analysis models
│   └── raw/            # The original, immutable data (NAC_Sustainability_BigData.csv)
├── docs/               
│   ├── dictionaries/   # Data dictionaries explaining column meanings (AQI, kWh, etc.)
│   ├── manual/         # Methodology documentation and project guides
│   └── tasks/          # Project roadmaps and milestone tracking
├── notebooks/          # Jupyter notebooks for EDA and prototyping (e.g., 01-eda-air-quality.ipynb)
├── reports/            
│   └── figures/        # Exported charts and visualizations for the final presentation
├── src/                # Modular source code
│   ├── data/           # Scripts for data generation, collection, and ingestion
│   ├── features/       # Scripts to transform raw data into features (e.g., calculating carbon footprint)
│   └── visualization/  # Custom plotting scripts for consistent branding
├── .gitignore          # Rules to exclude large data files and environment variables
├── README.md           # This file!
└── requirements.txt    # List of dependencies (pandas, numpy, etc.)