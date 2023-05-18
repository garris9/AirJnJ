## ETL
### Data Sources
Our data was extracted from Kaggle at the following link: [airbnb-listings-reviews](https://www.kaggle.com/datasets/mysarahmadbhat/airbnb-listings-reviews)
Our API data was consumed from the Yelp Fusion API at [Yelp](https://www.yelp.com/developers/documentation/v3/authentication)

We used Azure DataBricks for ETL. 

### Extraction
We used VSCode to call the Yelp API, which was extracted in JSON format, and then saved into .csv for efficiency in future processing. The consumed API data was saved as a binary large object file in our Azure Data Lake in the cohort50 resource group, within our AirJNJ storage container.

### Transformation
We then used Python Pandas in a Jupyter notebook in Databricks to load the data from the JSON file into dataframes.


| Column 1 | Column 2 | Column 3 |
|----------|----------|----------|
|   Row 1  |   Data   |   Data   |
|   Row 2  |   Data   |   Data   |


### Load
We then loaded the data from the Jupyter Notebook into the AirJNJ SQL database for permanent storage using the following built-in:
*DataFrame.to_sql(name, con, schema=None, if_exists='fail', index=True, index_label=None, chunksize=None, dtype=None, method=None)*







