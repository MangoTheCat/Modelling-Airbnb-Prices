
# Modeling Airbnb prices
In this post we're going to model the prices of Airbnb appartments in London. In other words, the aim is to build our own price suggestion model. We will be using data from http://insideairbnb.com/ which we collected in April 2018. This work is inspired from the [Airbnb price prediction model](https://d1no007.github.io/OptiBnB/) built by Dino Rodriguez, Chase Davis, and Ayomide Opeyemi. Normally we would be doing this in R but we thought we'd try our hand at Python for a change.

We present a shortened version here, the full version is available on our [GitHub](https://github.com/MangoTheCat/Modelling-Airbnb-Prices/tree/reproduce-results). 

## Data Preprocessing
First we import the listings gathered in the csv file.


```python
import pandas as pd

listings_file_path = 'listings.csv.gz' 
listings = pd.read_csv(listings_file_path, compression="gzip", low_memory=False)
listings.columns
```




    Index(['id', 'listing_url', 'scrape_id', 'last_scraped', 'name', 'summary',
           'space', 'description', 'experiences_offered', 'neighborhood_overview',
           'notes', 'transit', 'access', 'interaction', 'house_rules',
           'thumbnail_url', 'medium_url', 'picture_url', 'xl_picture_url',
           'host_id', 'host_url', 'host_name', 'host_since', 'host_location',
           'host_about', 'host_response_time', 'host_response_rate',
           'host_acceptance_rate', 'host_is_superhost', 'host_thumbnail_url',
           'host_picture_url', 'host_neighbourhood', 'host_listings_count',
           'host_total_listings_count', 'host_verifications',
           'host_has_profile_pic', 'host_identity_verified', 'street',
           'neighbourhood', 'neighbourhood_cleansed',
           'neighbourhood_group_cleansed', 'city', 'state', 'zipcode', 'market',
           'smart_location', 'country_code', 'country', 'latitude', 'longitude',
           'is_location_exact', 'property_type', 'room_type', 'accommodates',
           'bathrooms', 'bedrooms', 'beds', 'bed_type', 'amenities', 'square_feet',
           'price', 'weekly_price', 'monthly_price', 'security_deposit',
           'cleaning_fee', 'guests_included', 'extra_people', 'minimum_nights',
           'maximum_nights', 'calendar_updated', 'has_availability',
           'availability_30', 'availability_60', 'availability_90',
           'availability_365', 'calendar_last_scraped', 'number_of_reviews',
           'first_review', 'last_review', 'review_scores_rating',
           'review_scores_accuracy', 'review_scores_cleanliness',
           'review_scores_checkin', 'review_scores_communication',
           'review_scores_location', 'review_scores_value', 'requires_license',
           'license', 'jurisdiction_names', 'instant_bookable',
           'cancellation_policy', 'require_guest_profile_picture',
           'require_guest_phone_verification', 'calculated_host_listings_count',
           'reviews_per_month'],
          dtype='object')



The data has 95 columns or features. Our first step is to perform feature selection to reduce this number.

### Feature selection

#### Selection on Missing Data 
Features that have a high number of missing values aren't useful for our model so we should remove them.


```python
import matplotlib.pyplot as plt
%matplotlib inline

percentage_missing_data = listings.isnull().sum() / listings.shape[0]
ax = percentage_missing_data.plot(kind = 'bar', color='#E35A5C', figsize = (16, 5))
ax.set_xlabel('Feature')
ax.set_ylabel('Percent Empty / NaN')
ax.set_title('Feature Emptiness')
plt.show()
```


![](output_10_0.png)


As we can see, the features `neighbourhood_group_cleansed`, `square_feet`, `has_availability`, `license` and `jurisdiction_names` mostly have missing values. The features `neighbourhood`, `cleaning_fee` and `security_deposit` are more than 30% empty which is too much in our opinion. The `zipcode` feature also has some missing values but we can either remove these values or impute them within reasonable accuracy. 


```python
useless = ['neighbourhood', 'neighbourhood_group_cleansed', 'square_feet', 'security_deposit', 'cleaning_fee', 
           'has_availability', 'license', 'jurisdiction_names']
listings.drop(useless, axis=1, inplace=True)
```

#### Selection on Sparse Categorical Features
Let's have a look at the categorical data to see the number of unique values.


```python
categories = listings.columns[listings.dtypes == 'object']
percentage_unique = listings[categories].nunique() / listings.shape[0]

ax = percentage_unique.plot(kind = 'bar', color='#E35A5C', figsize = (16, 5))
ax.set_xlabel('Feature')
ax.set_ylabel('Percent # Unique')
ax.set_title('Feature Emptiness')
plt.show()
```


![](output_14_0.png)


We can see that the `street` and `amenities` features have a large number of unique values. It would require some natural language processing to properly wrangle these into useful features. We believe we have enough location information with `neighbourhood_cleansed` and `zipcode` so we'll remove `street`. We also remove `amenities`, `calendar_updated` and `calendar_last_updated` features as these are too complicated to process for the moment. 


```python
to_drop = ['street', 'amenities', 'calendar_last_scraped', 'calendar_updated']
listings.drop(to_drop, axis=1, inplace=True)
```

Now, let's have a look at the `zipcode` feature. The above visualisation shows us that there are lots of different postcodes, maybe too many?


```python
print("Number of Zipcodes:", listings['zipcode'].nunique())
```

    Number of Zipcodes: 24774
    

Indeed, there are too many zipcodes. If we leave this feature as is it might cause overfitting. Instead we can regroup the postcodes. At the moment, they are separated as in the following example: KT1 1PE. We'll keep the first part of the zipcode (e.g. KT1) and accept that this gives us some less precise location information.


```python
listings['zipcode'] = listings['zipcode'].str.slice(0,3)
listings['zipcode'] = listings['zipcode'].fillna("OTHER")
print("Number of Zipcodes:", listings['zipcode'].nunique())
```

    Number of Zipcodes: 461
    

A lot of zipcodes contain less than 100 appartments and a few zipcodes contain most of the appartments. Let's keep these ones.


```python
relevant_zipcodes = count_per_zipcode[count_per_zipcode > 100].index
listings_zip_filtered = listings[listings['zipcode'].isin(relevant_zipcodes)]

# Plot new zipcodes distribution
count_per_zipcode = listings_zip_filtered['zipcode'].value_counts()
ax = count_per_zipcode.plot(kind='bar', figsize = (22,4), color = '#E35A5C', alpha = 0.85)
ax.set_title("Zipcodes by Number of Listings")
ax.set_xlabel("Zipcode")
ax.set_ylabel("# of Listings")

plt.show()

print('Number of entries removed: ', listings.shape[0] - listings_zip_filtered.shape[0])
```


![](output_25_0.png)


    Number of entries removed:  5484
    

This distribution is much better, and we only removed 5484 rows from our dataframe which contained about 53904 rows.

#### Selection on Correlated Features

Next we look at correlations.

```python
import numpy as np
from sklearn import preprocessing

# Function to label encode categorical variables.
# Input: array (array of values)
# Output: array (array of encoded values)
def encode_categorical(array):
    if not array.dtype == np.dtype('float64'):
        return preprocessing.LabelEncoder().fit_transform(array) 
    else:
        return array
    
# Temporary dataframe
temp_data = listings_neighborhood_filtered.copy()

# Delete additional entries with NaN values
temp_data = temp_data.dropna(axis=0)

# Encode categorical data
temp_data = temp_data.apply(encode_categorical)
# Compute matrix of correlation coefficients
corr_matrix = temp_data.corr()
```

```python
# Display heat map 
plt.figure(figsize=(7, 7))
plt.pcolor(corr_matrix, cmap='RdBu')
plt.xlabel('Predictor Index')
plt.ylabel('Predictor Index')
plt.title('Heatmap of Correlation Matrix')
plt.colorbar()

plt.show()
```


![](output_33_0.png)


This reveals that `calculated_host_listings_count` is highly correlated with `host_total_listings_count`  so we'll keep the latter. We also see that the `availability_*` variables are correlated with each other. We'll keep `availability_365` as this one is less correlated with other variables. Finally, we decide to drop `requires_license` which has an odd correlation result of NA's which will not be useful in our model.


```python
useless = ['calculated_host_listings_count', 'availability_30', 'availability_60', 'availability_90', 'requires_license']
listings_processed = listings_neighborhood_filtered.drop(useless, axis=1)
```


### Data Splitting: Features / labels - Training set / testing set
Now we split into into features and labels and training and testing sets. We also convert the train and test dataframe into numpy arrays so that they can be used to train and test the models.


```python
# Shuffle the data to ensure a good distribution for the training and testing sets
from sklearn.utils import shuffle
listings_processed = shuffle(listings_processed)

# Extract features and labels
y = listings_processed['price']
X = listings_processed.drop('price', axis = 1)

# Training and Testing Sets
from sklearn.model_selection import train_test_split
train_X, test_X, train_y, test_y = train_test_split(X, y, random_state = 0)

train_X = np.array(train_X)
test_X = np.array(test_X)
train_y = np.array(train_y)
test_y = np.array(test_y)

train_X.shape, test_X.shape
```




    ((36185, 170), (12062, 170))



## Modelling 

Now that the data preprocessing is over, we can start the second part of this work: applying different Machine Learning models. We decided to apply 3 different models:

* Random Forest, with the RandomForestRegressor from the Scikit-learn library
* Gradient Boosting method, with the XGBRegressor from the XGBoost library
* Neural Network, with the MLPRegressor from the Scikit-learn library.

Each time, we applied the model with its default hyperparameters and we then tuned the model in order to get the best hyperparameters. The metrics we use to evaluate the models is the median absolute error due to the presence of extreme outliers and skewness in the data set.

We only show the code the Random Forest here, for the rest of the code please see the full version of this blogpost on our [GitHub](https://github.com/MangoTheCat/Modelling-Airbnb-Prices/tree/reproduce-results).

### Application of the Random Forest Regressor 

Let's start with the Random Forest model.

#### With default hyperparameters

We first create a pipeline that imputes the missing values then scales the data and finally applies the model. We then fit this pipeline to the training set.


```python
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import StandardScaler

# Create the pipeline (imputer + scaler + regressor)
my_pipeline_RF = make_pipeline(Imputer(), StandardScaler(),
                               RandomForestRegressor(random_state=42))

# Fit the model
my_pipeline_RF.fit(train_X, train_y)
```

We evaluate this model on the test set, using the median absolute error to measure the performance of the model. We'll also include the root-mean-square error (RMSE) for completeness. Since we'll be doing this repeatedly it is good practice to create a function.


```python
from sklearn.metrics import median_absolute_error
from sklearn.metrics import mean_squared_error
from math import sqrt

def evaluate_model(model, predict_set, evaluate_set):
    predictions = model.predict(predict_set)
    print("Median Absolute Error: " + str(round(median_absolute_error(predictions, evaluate_set), 2))) 
    RMSE = round(sqrt(mean_squared_error(predictions, evaluate_set)), 2)
    print("RMSE: " + str(RMSE)) 
```


```python
evaluate_model(my_pipeline_RF, test_X, test_y)
```

    Median Absolute Error: 14.2
    RMSE: 126.16
    


#### Hyperparameters tuning

We had some good results with the default hyperparameters of the Random Forest regressor. But we can improve the results with some hyperparameter tuning. There are two main methods available for this:

* Random search
* Grid search.

You have to provide a parameter grid to these methods. Then, they both try different combinations of parameters within the grid you provided. But the first one only tries several combinations whereas the second one tries all the possible combinations with the grid you provided.

What we have done is that we started with a random search to roughly evaluate a good combination of parameters. Once this is done, we use the grid search to get more precise results.

##### Randomized Search with Cross Validation 


```python
import numpy as np

# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 10, stop = 1000, num = 11)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num = 5)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]
# Create the random grid
random_grid = {'randomforestregressor__n_estimators': n_estimators,
               'randomforestregressor__max_features': max_features,
               'randomforestregressor__max_depth': max_depth,
               'randomforestregressor__min_samples_split': min_samples_split,
               'randomforestregressor__min_samples_leaf': min_samples_leaf,
               'randomforestregressor__bootstrap': bootstrap}
```


```python
# Use the random grid to search for best hyperparameters
from sklearn.model_selection import RandomizedSearchCV

# Random search of parameters, using 2 fold cross validation, 
# search across 100 different combinations, and use all available cores
rf_random = RandomizedSearchCV(estimator = my_pipeline_RF, 
                               param_distributions = random_grid, 
                               n_iter = 50, cv = 2, verbose=2,
                               random_state = 42, n_jobs = -1, 
                               scoring = 'neg_median_absolute_error')
# Fit our model
rf_random.fit(train_X, train_y)

rf_random.best_params_
```

    {'randomforestregressor__bootstrap': True,
     'randomforestregressor__max_depth': 35,
     'randomforestregressor__max_features': 'auto',
     'randomforestregressor__min_samples_leaf': 2,
     'randomforestregressor__min_samples_split': 5,
     'randomforestregressor__n_estimators': 1000}



##### Grid Search with Cross Validation 


```python
from sklearn.model_selection import GridSearchCV
# Create the parameter grid based on the results of random search 
param_grid = {
    'randomforestregressor__bootstrap': [True],
    'randomforestregressor__max_depth': [30, 35, 40], 
    'randomforestregressor__max_features': ['auto'],
    'randomforestregressor__min_samples_leaf': [2],
    'randomforestregressor__min_samples_split': [4, 5, 6],
    'randomforestregressor__n_estimators': [950, 1000, 1050] 
}

# Instantiate the grid search model
grid_search = GridSearchCV(estimator = my_pipeline_RF, 
                           param_grid = param_grid, 
                           cv = 3, n_jobs = -1, verbose = 2, 
                           scoring = 'neg_median_absolute_error')

# Fit the grid search to the data
grid_search.fit(train_X, train_y)

grid_search.best_params_
```
    {'randomforestregressor__bootstrap': True,
     'randomforestregressor__max_depth': 30,
     'randomforestregressor__max_features': 'auto',
     'randomforestregressor__min_samples_leaf': 2,
     'randomforestregressor__min_samples_split': 4,
     'randomforestregressor__n_estimators': 1050}



##### Final Model


```python
# Create the pipeline (imputer + scaler + regressor)
my_pipeline_RF_grid = make_pipeline(Imputer(), StandardScaler(),
                                      RandomForestRegressor(random_state=42,
                                                            bootstrap = True,
                                                            max_depth = 30,
                                                            max_features = 'auto',
                                                            min_samples_leaf = 2,
                                                            min_samples_split = 4,
                                                            n_estimators = 1050))

# Fit the model
my_pipeline_RF_grid.fit(train_X, train_y)

evaluate_model(my_pipeline_RF_grid, test_X, test_y)
```

    Median Absolute Error: 13.57
    RMSE: 125.04
    

We get better results with the tuned model than with default hyperparameters, but the improvement of the median absolute error is not amazing. Maybe we will have a better precision if we use another model.

### Visualision of all models' performance

![](output_80_0.png)


The tuned Random Forest and XGBoost gave the best results on the test set. Surprisingly, the Multi Layer Perceptron with default parameters gave the highest Median Absolute errors, and the tuned one did not even give better results than the default Random Forest. This is unusual, maybe the Multi Layer Perceptron needs more data to perform better, or it might need more tuning on important hyperparameters such as the hidden_layer_sizes. 

## Conclusion

In this post we modelled Airbnb apartment prices using descriptive data from the airbnb website. First, we preprocessed the data to remove any redundant features and reduce the sparsity of the data. Then we applied three different algorithms, initially with default parameters which we then tuned. In our results the tuned Random Forest and tuned XGBoost performed best. 

To further improve our models we could include more feature engineering, for example time-based features. We could also try more extensive hyperparameter tuning. If you would like to give it a go yourself, the code and data for this post can be found on [GitHub](https://github.com/MangoTheCat/Modelling-Airbnb-Prices#XGB)
