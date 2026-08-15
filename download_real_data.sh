#!/bin/bash
# Re-downloads the real SCMS supply-chain shipment dataset used to validate this
# pipeline on genuine (non-synthetic) data. Already included at
# data/SCMS_Delivery_History_Dataset.csv -- this script is here in case you need
# to re-fetch it or verify the source.
#
# Original source: USAID Supply Chain Management System (SCMS) Delivery History
#   https://www.usaid.gov/opengov/developer/datasets/SCMS_Delivery_History_Dataset_20150929.csv
# Mirror used here (same file, GitHub-hosted):
#   https://github.com/jrcinco/supply-chain-shipment-price-data

set -e
mkdir -p data
curl -sL "https://raw.githubusercontent.com/jrcinco/supply-chain-shipment-price-data/master/SCMS_Delivery_History_Dataset.csv" \
  -o data/SCMS_Delivery_History_Dataset.csv
echo "Downloaded to data/SCMS_Delivery_History_Dataset.csv"
wc -l data/SCMS_Delivery_History_Dataset.csv
