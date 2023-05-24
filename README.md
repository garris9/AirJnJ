# Market Analysis and Pricing Optimization for Airbnb Listings in New York

## Summary

This study aims to assist Airbnb hosts in maximizing their rental income by leveraging data-driven pricing strategies. Our analysis of Airbnb listings in New York and Paris also incorporates data from the Yelp API for nearby restaurants and attractions. The project deliverables provide Airbnb owners with actionable insights as well as a machine learning model to optimize their listing prices based on various factors.

## Group Members
- [Jeremy](https://github.com/garris9)
- [Jessica](https://github.com/jhoffmanDEV10)
- [Noelle](https://github.com/nkiesz39)

## Repo Structure
- [Dashboard](https://github.com/garris9/AirJnJ/tree/main/Dashboard):
  - This folder contains a screenshot of our dashboard for the project.
- [EDA](https://github.com/garris9/AirJnJ/tree/main/EDA):
  - This folder contains four notbooks that were used for Exploratory Data Analysis:
    - [EDA.ipynb](https://github.com/garris9/AirJnJ/blob/main/EDA/EDA.ipynb)
    - [airbnb_X_yelp.ipynb](https://github.com/garris9/AirJnJ/blob/main/EDA/airbnb_x_yelp.ipynb)
    - [listings_EDS.ipynb](https://github.com/garris9/AirJnJ/blob/main/EDA/listings_EDA.ipynb)
    - [yelp_EDA_nlk.ipynb](https://github.com/garris9/AirJnJ/blob/main/EDA/yelp_EDA_nlk.ipynb)
  - The EDA folder also contains an [IMG](https://github.com/garris9/AirJnJ/tree/main/EDA/IMG) folder that holds images and graphs from the initial EDA
- [ETL](https://github.com/garris9/AirJnJ/tree/main/ETL): 
  - This folder contains the report for the ETL proccesses that were taken during this project.
- [Machine Learning](https://github.com/garris9/AirJnJ/tree/main/Machine%20Learning):
  - This folder contains notebooks used for training machine learning models:
    - [Ensemble Learning](https://github.com/garris9/AirJnJ/blob/main/Machine%20Learning/ML_ensemble.ipynb)
    - [XGBoost](https://github.com/garris9/AirJnJ/blob/main/Machine%20Learning/ML_xgboost.ipynb)
    - [KNN](https://github.com/garris9/AirJnJ/blob/main/Machine%20Learning/ml.ipynb)
-[Presentation](https://github.com/garris9/AirJnJ/tree/main/Presentation) :
  - This folder contains the powerpoint slides 
- [Data](https://github.com/garris9/AirJnJ/tree/main/data):
  - This folder contains the CSVs that were saved from the API calls for Paris and each of the five New York boroughs. 
  - The Airbnb Listing CSV was too large to include on the repo, but can be found at [Airbnb Listing and Reviews](https://www.kaggle.com/datasets/mysarahmadbhat/airbnb-listings-reviews).
  - The data folder also contains an [API Calls](https://github.com/garris9/AirJnJ/tree/main/data/API%20Calls) folder where the notebook with the code and information about the API calls is kept

## Dashboard
![Dashboard](https://github.com/garris9/AirJnJ/blob/main/EDA/IMG/dash.gif)

## Guiding Questions

Our group's main aim during this project was to use data-driven insights to answer the question of: 
'What an Airbnb host should list their property at to maximize rental income?'

To answer this, we focused on the following questions:
  - Is there a correlation between AirBnB ratings and nearby Yelp restaurants and attractions?
  - Is there a difference between the New York City and Paris markets?
  - What are the best predictors of the nightly price for an AirBnB?
  - Within the NYC market are there strong differences in the five boroughs?
  - Is there a correlation between population and more or higher priced AirBnBs?
    - If not, is there correlation with city size?
  - Is there a significant difference between attractions and restaurant types in NYC and Paris?

## Data

The data for this project was obtained from Kaggle. The [Airbnb Listing and Reviews](https://www.kaggle.com/datasets/mysarahmadbhat/airbnb-listings-reviews) dataset was used. The dataset contains four CSVs that can be downloaded: A Listings CSV and accompanying data dictionary and a Reviews CSV with an accompanying data dictionary. For our purpose, we focused mainly on the Listings CSV that contains over 250,000 Airbnb listings in 10 major cities. We filtered cities down to New York and Paris to make it more managable, while still keeping over 100,000 listings. 

The [Yelp Fusion API](https://www.yelp.com/developers/documentation/v3/authentication) was used to pull in data about the restaurant and attraction ratings for both cities. To keep from needing to run the API call everytime we need that data, we saved the calls to CSVs that can be found in the [data](https://github.com/garris9/AirJnJ/tree/main/data) folder of this repo. The cities Paris and New York are separated, with New York being broken down futher into the five boroughs: Manhattan, Staten Island, Queens, Bronx, and Brooklyn.

## ETL Process

### Extraction
VSCode was used to write the code to call the Yelp API. Using the Yelp API allows only 50 places of business to come back per call. With that information, we looped through the zipcodes of each city. Parameters were also used to call attractions and restaurants separately and to ensure other types of businesses wuld not be included in the dataset. Paris and each borough of New York was extracted from the returned json and saved as a .csv for time and efficiency in future processing. The API data was saved into Azure Data Lake, within our AirJNJ storage container.
### Transform 
A Jupyter notebook was created in Databricks to load the data from the json into dataframes and to use Pandas. When reading in the AirBnB Listing data,```encoding='ISO-8859-1'``` was used to deal with the foreign characters used in the French language. The Euro symbol for the Paris prices was added. All duplicated values from each dataframe were dropped, Afterwards, all of the Yelp dataframes we concatenated together to ensure the same cleaning steps were applied to all. Null and irrelevant columns in the AirBnB Listing data such as: 'host_response_time', 'host_response_rate', 'host_acceptance_rate', 'host_has_profile_pic' were dropped. From the Yelp data the 'image_url','url','phone','display_phone' columns were dropped. 

To deal with the list of dictionaries within the Yelp Categories column, the ast library was used to pull out the title and store it in a new list. This list was uysed to create a new categories column. Since many of the business has more than one listed category, another new column called 'First Listed Category' was created. It took the first item in the list for each reataurant or attraction.
Another column called 'Restaurant or Attraction?' was added to enable filtering later on. 

To vizualize relationships, scatterplots of the data were used:

#### NYC
![ALT NYC Factors Impacting Price](https://github.com/garris9/AirJnJ/blob/main/EDA/IMG/Factors%20Impacting%20Price%20from%20NY%20Airbnbs%20copy.png)

#### Paris 
![ALT Paris Factors Impacting price](https://github.com/garris9/AirJnJ/blob/main/EDA/IMG/Factors%20Impacting%20Price%20from%20Paris%20Airbnbs%20copy.png)

Two patterns stuck out:

First, the closer a listing is to the center of the city, the higher its price. This more evident in NYC than in Paris.
Second, there seemed to be a positive correlation between price and ratings, where very low ratings correlate to very low price and very high ratings sometimes correlate to high price, but may be a poor conclusion that ratings are the strongest predictor of price. This is clear by the dense clustering of points around low priced listings with high ratings - ones that might fall into a "great value" category. A more conservative interpretation of this trend might be that listings with poor ratings are forced into lower prices if they want to continue getting business at all.

To get a look if there were any patterns for the Yelp ratings, they were mapped:
- The red dots correspond with ratings between 0 and 2.49
- The yellow dots correspond with ratings between 2.5 and 3.9
- The green dots correspon with ratings of 4+

![ALT NYC Ratings Mapped](https://github.com/garris9/AirJnJ/blob/main/EDA/IMG/NYC_Ratings_mapped.png)
To prepare for Machine Learning, a Distance formula was created to calculate the number of restaurants and attractions within a 2 mile radius of an Airbnb listing and added as a new column. The Yelp reviews for each borough and Paris were also aggregated into an average for each in placed in a new column. A new column '% Restaurants' was also created. 

### Load
The data from the Jupyter Notebook was loaded into the AirJNJ SQL database for permanent storage.


## Machine Learning
During Machine Learning, we started by looking at a correlation matrix.

![ALT Corr matrix](https://github.com/garris9/AirJnJ/blob/main/EDA/IMG/Airbnb_corr_matrix.png)

Four different machine learning models were used on the data to see which provided the best predictive model. For each model, the target value was the price, with each model having different predictors. The models that we trained were: 
- Linear Regression as a baseline
- KNearestNeighbors (KNN)
- XGBoost
- Ensemble Learning

#### Linear Regression
For the Linear Regression model, the predictors used were 'district','# w/in 2mi','Avg yelp rating', '% Restaurants','latitude','longitude'. Since District is a categorical variable, dummies were created using ```pd.get_dummies()```. The data was broken up in training and validation data. The model provided the following.
```
lr = LinearRegression()
lr.fit(X_train,y_train)
lr.score(X_test,y_test)
0.022390536758468804
```
Overall it returned a low score that did not increase much when other predictors were added.

#### KNearestNeighbors
For KNearestNeighbors, several models were fit with different parameters to try to optimize the prediction. The first using data that was split into training and validation with the same predictors as Linear Regression. The second with a scaled data to account for the slight increments in the latitude and longitude vs the larger increments in the number of restaurants within 2 miles column. It also used the same predictors as Linear Regression. 
```
# First KNN Predicitive Model
knn.score(X_test,y_test)
-0.20815964013840538

# Second KNN Model with X and y scaled
knn.score(X_test_scaled,y_test)
-0.21818449108723015
```
Neither model performed well or improved upon the other.

In an attempt to improve the score, we began to set parameters. First, a price threshold was created, getting rid of any nightly price that was above $500. A list of predictors was created to make it easier to add and delete predictors and run the model again. The most successful KNN model only included ' latitude','longitude' and 'accommodation' as the predictors. We used a loop to test various numbers in the n_neighbors parameter.
```
for n_neighbors  in [2,5,10,20, 30, 50, 100,125,150,200]:
    kn = KNeighborsRegressor(n_neighbors=n_neighbors)
    kn.fit(Xtaccom,ytaccom)
    s = kn.score(Xtaccom, ytaccom),kn.score(Xvaccom, yvaccom)
    print(n_neighbors, s)

2 (0.7417333097853893, 0.2229172472074975)
5 (0.574609290863795, 0.36531975507774483)
10 (0.5099723839627631, 0.40116003766554675)
20 (0.4716113310106692, 0.41808997461782027)
30 (0.45550917505828414, 0.4233185236772593)
50 (0.44074969444968914, 0.4209391301947214)
100 (0.42424814783313347, 0.4168745834994939)
125 (0.4193117101171542, 0.41393692879570265)
150 (0.4149444394635141, 0.41145932087238934)
200 (0.4090564563363268, 0.40718808234119297)
```
The first number is the number of n_neighbors used for the model. The second number is how well the model predicts on the training data, while the third number is how well it predicts on the unseen validation data. When the n_neighbors is less than 50, the model is highly overfit and performs poorly on the unseen data. As the n_neighbors increase, the model starts to perform better on the unseen validation data and it not being overfit. 

#### XGBoost
For XGBoost, the nulls in the 'district' column from the Airbnb data was filled with 'Paris_d', since the only nulls were for Paris. Any nulls in the 'bedroom' or the 'Avg Yelp Rating' columns were set to the mean. Similarly to KNN, a price threshold of 500 was set. The predictors used were: 
'Avg yelp rating','review_scores_rating','review_scores_cleanliness','accommodates','bedrooms','review_scores_accuracy','review_scores_checkin',
'review_scores_communication','review_scores_location','review_scores_value',
'# w/in 2mi','district','# of airbnb reviews','property_type'. Dummy variable for both the district and the property type were created. The data was split into its training a validation data. 
The XGBRegressor was imported from xgboost for use in this model.
```
xgb_model = xgb.XGBRegressor(objective='reg:linear',random_state=0,eta=.1)
xgb_model.fit(X_train, y_train)
print("Training set accuracy score:",xgb_model.score(X_train,y_train))
print("Test set accuracy score:",xgb_model.score(X_test,y_test))

Training set accuracy score: 0.5423269854523223
Test set accuracy score: 0.4966196562089644
```
XGBoost gave us overall the highest score when looking at the validation set, with only slight overfitting.

#### Ensemble Learning
In an attempt to improve upon the scores of KNN and XGBoost, we employed Ensemble Learning. It was done using the predictors from XGBoost and KNN.

```
feats = xgb_model.feature_importances_

feats = feats.tolist()

tmp = [(a,b) for a,b, in zip(feats, X_train.columns)]

tmp.sort(key=lambda x: x[0])
```
Using the feature importance from the XGBoost model, here is an abbreviated list of the learning.

```
(0.014919784851372242, 'Shared room in apartment'),
 (0.015085997059941292, 'Private room in house'),
 (0.015144539065659046, 'Room in aparthotel'),
 (0.01597871258854866, 'Entire serviced apartment'),
 (0.019094787538051605, 'Room in hotel'),
 (0.019150499254465103, 'review_scores_location'),
 (0.019154243171215057, 'Entire loft'),
 (0.01915448158979416, 'review_scores_cleanliness'),
 (0.01942456141114235, 'Brooklyn'),
 (0.021227726712822914, '# w/in 2mi'),
 (0.023188522085547447, 'Private room in townhouse'),
 (0.031567297875881195, 'Paris_d'),
 (0.036349646747112274, 'Entire apartment'),
 (0.05065086856484413, 'accommodates'),
 (0.07013404369354248, 'Manhattan'),
 (0.09304781258106232, 'Room in boutique hotel'),
 (0.09455633908510208, 'Private room in apartment'),
 (0.2328738570213318, 'bedrooms')]
```

## Conclusions 

When it comes to a correlation between AirBnB ratings and nearby Yelp restaurants and attractions there is very little correlation. One hypthesis for this is that both cities we picked to look at have extensive public transportation systems. So, when looking for an Airbnb to stay in it does not have to be right in the vicinity of the restaurants or attractions that someone wants to visit. If we had more time, we would have liked to see if another city or a suburb with limited public tranportation would have drawn the same conclusions of New York and Paris.

There does not seem to be a correlation between the number of Airbnbs and the population of the city. New York has about four times the population of Paris, but Paris has about two times more Airbnb listings. If it is not correlated with population is it correlated with the size of the city? This is also not the case. New York is about seven times larger than Paris, and about half the listings. This can be because New York recently introduced new Airbnb regulations, that require hosts to register with the city and does not allow hosts to rent out full homes where the host will not be present on the premises. In an attempt to combat single hosts with multiple listings and improve the housing crisis in the city. For more information on the new regulations [here](https://www.youtube.com/watch?v=GANfW00rFx8&t=2s) is an informational segment.  

Both the New York and Paris markets have similar prices, but Paris has more listings per capita. Within New York itself the boroughs also have similar prices to one another. However, the number of listings in each borough vary drastically. Manhattan and Brooklyn host the most with about 15 thousand and 14 thousand repectively. Staten Island hosts the fewest with only about 300. This can make sense because a majority of attraction that people come to see are in or near Manhattan. 

Based on the models that we trained, the best price predictor is the property type, whether it is a private room or the entire place, at 48%. Another good predictor was the number of bedrooms at 23%.

The Paris Yelp dataset was much smaller than that of New York's but we can see that both cities have a similar distribution of cuisine types. 

When pricing an Airbnb's nightly price, a host should mainly take into consideration the type of property they are listing and how many bedrooms are available for the renter. The location of the Airbnb was not as important as those two factors. During the project, we did not take into account the seasonality of tourism, which could be an important predictor into nightly price.

## Technologies
The following tools were used during the project:
  - Python Libraries
    - Pandas
    - Numpy
    - Seaborn
    - Matplotlib
    - Folium
    - Pyspark
    - Requests
    - Sklearn
    - json
    - Time
    - Ast
    - Geopandas
    - Plotly
    - Random
    - Geopy
  - Azure Data Studios
  - Azure Data Factory
  - Azure Databricks

## Resources
[Airbnb Listing and Reviews](https://www.kaggle.com/datasets/mysarahmadbhat/airbnb-listings-reviews)

[Yelp Fusion API Key](https://www.yelp.com/developers/documentation/v3/authentication)

[Yelp API business search](https://docs.developer.yelp.com/reference/v3_business_search)

["NYC introduces new Airbnb regulations" - FOX 5 New York](https://www.youtube.com/watch?v=GANfW00rFx8&t=2s) 
