# Market Analysis and Pricing Optimization for Airbnb Listings in New York

## Summary

This study aims to assist Airbnb hosts in maximizing their rental income by leveraging data-driven pricing strategies. Our analysis of Airbnb listings in New York and Paris also incorporates data from the Yelp API for nearby restaurants and attractions. The project deliverables provide Airbnb owners with actionable insights as well as a machine learning model to optimize their listing prices based on various factors.

## Group Members
- [Jeremy](https://github.com/garris9)
- [Jessica](https://github.com/jhoffmanDEV10)
- [Noelle](https://github.com/nkiesz39)

## Data

The data for this project was obtained from Kaggle. The [Airbnb Listing and Reviews](https://www.kaggle.com/datasets/mysarahmadbhat/airbnb-listings-reviews) dataset was used. The dataset contains four CSVs that can be downloaded: A Listings CSV and accompanying data dictionary and a Reviews CSV with an accompanying data dictionary. For our purpose, we focused mainly on the Listings CSV that contains over 250,000 Airbnb listings in 10 major cities. We filtered cities down to New York and Paris to make it more managable, while still keeping over 100,000 listings. 

The [Yelp Fusion API](https://www.yelp.com/developers/documentation/v3/authentication) was used to pull in data about the restaurant and attraction ratings for both cities. To keep from needing to run the API call everytime we need that data, we saved the calls to CSVs that can be found in the [data](https://github.com/garris9/AirJnJ/tree/main/data) folder of this repo. The cities Paris and New York are separated, with New York being broken down futher into the five boroughs: Manhattan, Staten Island, Queens, Bronx, and Brooklyn.

## Repo Structure
- [ETL](https://github.com/garris9/AirJnJ/tree/main/ETL): 
  - This folder contains the report for the ETL proccesses that were taken during this project.
- [Data](https://github.com/garris9/AirJnJ/tree/main/data):
  - This folder contains the CSVs that were saved from the API calls for Paris and each of the five New York boroughs. 
  - The Airbnb Listing CSV was too large to include on the repo, but can be found at [Airbnb Listing and Reviews](https://www.kaggle.com/datasets/mysarahmadbhat/airbnb-listings-reviews).
