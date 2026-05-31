# Data Access

This repository contains code, configurations, synthetic fixtures, compact manifests, curated result tables, and publication-ready figures. It does not redistribute licensed raw data.

## Required Data Sources

- WRDS CRSP monthly and daily stock files
- WRDS CRSP/Compustat Merged links and Compustat fundamentals
- WRDS OptionMetrics option quotes and security links
- WRDS Cboe/VIX and Fama-French factor libraries
- WRDS TAQ sample tables for trading-cost calibration
- WRDS IBES where available
- Official SEC Reg SHO pilot reference files
- Public macro and energy controls from FRED, BLS, BEA, and EIA

## Credentials

WRDS credentials should live in the user's normal WRDS credential mechanism, for example a `~/.pgpass` file on the compute host with permissions `600`. Optional external API keys should be supplied through runtime environment variables or an ignored `.env.local` file using the names in `.env.example`.

Do not commit `.env.local`, `.pgpass`, API keys, passwords, raw WRDS extracts, or generated private run logs.

## Recreating The Results

Use the command sequence in [docs/reproducibility.md](docs/reproducibility.md). The scripts write raw and processed licensed data under ignored `data/` folders, logs under ignored `logs/`, model binaries under ignored `outputs/models/`, and compact public artifacts under `docs/assets/`.
