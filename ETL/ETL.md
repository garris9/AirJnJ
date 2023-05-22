## ETL
### Data Sources
Our data was extracted from Kaggle at the following link: [airbnb-listings-reviews](https://www.kaggle.com/datasets/mysarahmadbhat/airbnb-listings-reviews)
Our API data was consumed from the Yelp Fusion API at [Yelp](https://www.yelp.com/developers/documentation/v3/authentication)

We used Azure DataBricks for ETL. 

### Extraction
We used VSCode to call the Yelp API, which was extracted in JSON format, and then saved into .csv for efficiency in future processing. The consumed API data was saved as a binary large object file in our Azure Data Lake in the cohort50 resource group, within our AirJNJ storage container.

### Transformation
We then used Python Pandas in a Jupyter notebook in Databricks to load the data from the JSON file into dataframes.


| Target Table | Target Column | Target Type/Length | Source Type/Length | Source Table | Transformation Specification |
|----------|----------|----------|----------|----------|----------|
|   ML  |   listing_id   |   integer (PK)   | integer | listing.csv | no transx |
|     |   name   |   string   | string | listing.csv | no transx |
|     |   host_id   |   integer   | integer | listing.csv | no transx |
|     |   host_since   |   date   | date | listing.csv | no transx |
|     |   host_location   |   string   | string | listing.csv | no transx |
|     |   host_response_rate   |   double   | double | listing.csv | no transx |
|     |   host_acceptance_rate   |   double   | double | listing.csv | no transx |
|     |   host_total_listings_count  |   double   | integer | listing.csv | no transx |
|     |   host_has_profile_pic   |   string   | string | listing.csv | no transx |
|     |   host_identity_verified   |   string   | string | listing.csv | no transx |
|     |   neighbourhood   |   string   | string | listing.csv | no transx |
|     |   district   |   string   | string | listing.csv | replaced nulls with Paris_d |
|     |   city   |   string   | string | listing.csv | removed all cities other than NYC and Paris |
|     |   latitude   |   double   | double | listing.csv | no transx |
|     |   longitude   |   double   | double | listing.csv | no transx |
|     |   property_type   |   string   | string | listing.csv | no transx |
|     |   room_type   |   string   | string | listing.csv | no transx |
|     |   accommodates   |   integer   | integer | listing.csv | no transx |
|     |   price   |   integer   | integer | listing.csv | no transx |
|     |   price_euro   |   integer   |  |  | converted NYC prices to euro |
|     |   price_usd   |   integer   |  |  | converted Paris prices to USD |
|     |   minimum_nights   |   integer   | integer | listing.csv | no transx |
|     |   maximum_nights   |   integer   | integer | listing.csv | no transx |
|     |   review_scores_ratings   |   double   | double | listing.csv | no transx |
|     |   review_scores_accuracy   |   double   | double | listing.csv | no transx |
|     |   review_scores_cleanliness   |   double   | double | listing.csv | no transx |
|     |   review_scores_checkin   |   double   | double | listing.csv | no transx |
|     |   review_scores_communication   |   double   | double | listing.csv | no transx |
|     |   review_scores_location   |   double   | double | listing.csv | no transx |
|     |   review_scores_value   |   double   | double | listing.csv | no transx |
|     |   # of airbnb reviews   |   double   |  | review.csv | collected the count of reviews and joined them to this table on listing_id |
|     |   instant_bookable   |   string   | string | listing.csv | no transx |
|     |   # w/in 2mi   |   double   |  | yelp api | using results from yelp api, counting how many yelp attractions and restaurants are within 2 miles of the listing |
|     |   Avg yelp rating   |   double   |  | yelp api | average rating of yelp locations within 2 miles |
|     |   % Restaurants   |   double   |  | yelp api | percentage of restaurants vs attractions of nearby yelp locations |


### Load
We then loaded the data from the Jupyter Notebook into the AirJNJ SQL database for permanent storage using the following code:

ml = spark.read.option("escape", "\"").csv('/mnt/airjnj/Clean/ml.csv', header=True, inferSchema=True)

```python
ml.write.format("jdbc").option(
    "url", f"jdbc:sqlserver://{server}:1433;databaseName={database};"
    ) \
    .mode("overwrite") \
    .option("dbtable", table) \
    .option("user", airjnj_db_login) \
    .option("password", airjnj_db_password) \
    .option("driver", "com.microsoft.sqlserver.jdbc.SQLServerDriver") \
    .save()
```


