---
output:
  html_document: default
  word_document: default
---

This post aims at modelling the prices of Airbnb appartments in London. In other words, the aim is to build our own price suggestion model. The data used in this post has been downloaded from the http://insideairbnb.com/ website. The data has been collected in April 2018. This work is inspired from the [`Airbnb price prediction model built by Dino Rodriguez, Chase Davis, and Ayomide Opeyemi`](https://d1no007.github.io/OptiBnB/).

This post contains two main parts:
* the data preprocessing, which aims at cleaning the data and selecting the most useful features for the models.
* the modelling with 3 different algorithms: Random Forests, XGBoost and a Neural Network.

The libraries used are the following:
* pandas
* matplotlib
* numpy
* collection
* scikit-learn
* xgboost

Let's start with the first part, data preprocessing.

# Data Preprocessing

The first thing to do is to set the seed in order to be able to reproduce the results.


```python
import random
random.seed(42)
```

## Import data 

Then I import the listings gathered in the csv file.


```python
# Import data

import pandas as pd

main_file_path = 'listings.csv' 
data = pd.read_csv(main_file_path, low_memory=False)

print(data.columns)
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
    

As this file has 95 columns, I decide to have a look at the features names to delete the ones that directly seem useless. This is my first feature selection step.

## Feature Selection


```python
useless = ['id', 'listing_url', 'scrape_id', 'last_scraped', 'name', 'summary',
       'space', 'description', 'experiences_offered', 'neighborhood_overview',
       'notes', 'transit', 'access', 'interaction', 'house_rules',
       'thumbnail_url', 'medium_url', 'picture_url', 'xl_picture_url',
       'host_id', 'host_url', 'host_name', 'host_since', 'host_location',
       'host_about', 'host_response_time', 'host_response_rate',
       'host_acceptance_rate', 'host_is_superhost', 'host_thumbnail_url',
       'host_picture_url', 'host_neighbourhood', 'host_listings_count',
       'host_verifications', 'host_has_profile_pic', 'host_identity_verified',
       'city', 'state', 'market', 'smart_location', 'country_code', 'country',
       'is_location_exact', 'weekly_price', 'monthly_price']
data.drop(useless, axis=1, inplace=True)
```

I also decide to delete the following features as they are only available for old Airbnb appartments. Let's imagine that I am new on Airbnb and I want to rent my appartment. At that time, I won't have any review score for my appartment. This is what is called leaky predictors, and these features should not be part of my model.


```python
# Drop reviews features as they are not available for new apartments in Airbnb
useless = ['number_of_reviews', 'first_review', 'last_review', 'review_scores_rating',
           'review_scores_accuracy', 'review_scores_cleanliness',
           'review_scores_checkin', 'review_scores_communication',
           'review_scores_location', 'review_scores_value', 'reviews_per_month']
data.drop(useless, axis=1, inplace=True)
```

### Selection on numerical data

Let's have a look at the numerical variables. Using the `describe` function, we are able to spot the features which have missing values.


```python
data.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>host_total_listings_count</th>
      <th>neighbourhood_group_cleansed</th>
      <th>latitude</th>
      <th>longitude</th>
      <th>accommodates</th>
      <th>bathrooms</th>
      <th>bedrooms</th>
      <th>beds</th>
      <th>square_feet</th>
      <th>guests_included</th>
      <th>minimum_nights</th>
      <th>maximum_nights</th>
      <th>has_availability</th>
      <th>availability_30</th>
      <th>availability_60</th>
      <th>availability_90</th>
      <th>availability_365</th>
      <th>license</th>
      <th>calculated_host_listings_count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>53895.000000</td>
      <td>0.0</td>
      <td>53904.000000</td>
      <td>53904.000000</td>
      <td>53904.000000</td>
      <td>53644.000000</td>
      <td>53811.000000</td>
      <td>53731.000000</td>
      <td>582.000000</td>
      <td>53904.000000</td>
      <td>53904.000000</td>
      <td>5.390400e+04</td>
      <td>0.0</td>
      <td>53904.000000</td>
      <td>53904.000000</td>
      <td>53904.000000</td>
      <td>53904.000000</td>
      <td>0.0</td>
      <td>53904.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>15.885500</td>
      <td>NaN</td>
      <td>51.510425</td>
      <td>-0.127105</td>
      <td>3.036676</td>
      <td>1.262751</td>
      <td>1.353980</td>
      <td>1.708027</td>
      <td>577.508591</td>
      <td>1.407428</td>
      <td>3.285229</td>
      <td>2.683750e+05</td>
      <td>NaN</td>
      <td>11.830755</td>
      <td>25.337674</td>
      <td>40.514433</td>
      <td>155.849789</td>
      <td>NaN</td>
      <td>14.976106</td>
    </tr>
    <tr>
      <th>std</th>
      <td>84.700442</td>
      <td>NaN</td>
      <td>0.045454</td>
      <td>0.088346</td>
      <td>1.907429</td>
      <td>0.547699</td>
      <td>0.841912</td>
      <td>1.201165</td>
      <td>726.154243</td>
      <td>1.040308</td>
      <td>28.536837</td>
      <td>2.317162e+07</td>
      <td>NaN</td>
      <td>12.178077</td>
      <td>24.064923</td>
      <td>36.383218</td>
      <td>144.032928</td>
      <td>NaN</td>
      <td>81.750305</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.000000</td>
      <td>NaN</td>
      <td>51.292892</td>
      <td>-0.501305</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000e+00</td>
      <td>NaN</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>NaN</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>1.000000</td>
      <td>NaN</td>
      <td>51.485099</td>
      <td>-0.187191</td>
      <td>2.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>108.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>6.000000e+01</td>
      <td>NaN</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>9.000000</td>
      <td>NaN</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>1.000000</td>
      <td>NaN</td>
      <td>51.514730</td>
      <td>-0.122403</td>
      <td>2.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>484.000000</td>
      <td>1.000000</td>
      <td>2.000000</td>
      <td>1.125000e+03</td>
      <td>NaN</td>
      <td>8.000000</td>
      <td>20.000000</td>
      <td>36.000000</td>
      <td>92.000000</td>
      <td>NaN</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>3.000000</td>
      <td>NaN</td>
      <td>51.538943</td>
      <td>-0.069183</td>
      <td>4.000000</td>
      <td>1.500000</td>
      <td>2.000000</td>
      <td>2.000000</td>
      <td>819.500000</td>
      <td>1.000000</td>
      <td>3.000000</td>
      <td>1.125000e+03</td>
      <td>NaN</td>
      <td>26.000000</td>
      <td>53.000000</td>
      <td>81.000000</td>
      <td>321.000000</td>
      <td>NaN</td>
      <td>3.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>735.000000</td>
      <td>NaN</td>
      <td>51.683101</td>
      <td>0.317523</td>
      <td>16.000000</td>
      <td>8.000000</td>
      <td>10.000000</td>
      <td>16.000000</td>
      <td>10710.000000</td>
      <td>16.000000</td>
      <td>5000.000000</td>
      <td>2.147484e+09</td>
      <td>NaN</td>
      <td>30.000000</td>
      <td>60.000000</td>
      <td>90.000000</td>
      <td>365.000000</td>
      <td>NaN</td>
      <td>711.000000</td>
    </tr>
  </tbody>
</table>
</div>



As we can see, the features `neighbourhood_group_cleansed`, `license` and `has_availability` only have missing values. So I decide to delete them. The feature `square_feet` only has 600 observations which are not missing values, this is not enought as the data contains about 54000 rows. So I also delete this variable.


```python
# Drop count = 0 and count < 600 (square_feet)
useless = ['neighbourhood_group_cleansed', 'license', 'has_availability', 'square_feet']
data.drop(useless, axis=1, inplace=True)
```

### Feature Emptiness

Here, I want to have a look at the feature emptiness for all the variables (numerical and categorical). We use the `percent_empty` function on all the remaining features and plot the result.


```python
import matplotlib.pyplot as plt

def percent_empty(df):
    
    bools = df.isnull().tolist()
    percent_empty = float(bools.count(True)) / float(len(bools))
    
    return percent_empty, float(bools.count(True))

# Store emptiness for all features
emptiness = []

missing_columns = []

# Get emptiness for all features
for i in range(0, data.shape[1]):
    p, n = percent_empty(data.iloc[:,i])
    if n > 0:
        missing_columns.append(data.columns.values[i])
    emptiness.append(round((p), 2))
    
empty_dict = dict(zip(data.columns.values.tolist(), emptiness))

# Plot emptiness graph
empty = pd.DataFrame.from_dict(empty_dict, orient = 'index').sort_values(by=0)
ax = empty.plot(kind = 'bar', color='#E35A5C', figsize = (16, 5))
ax.set_xlabel('Predictor')
ax.set_ylabel('Percent Empty / NaN')
ax.set_title('Feature Emptiness')
ax.legend_.remove()

plt.show()
```


![png](plots/output_19_0.png)


As we can see, the `jurisdiction_names` feature is 100% empty, so we delete this one. The `neighbourhood`, `cleaning_fee` and `security_deposit` are more than 40% empty. I think this is too much so I decide to delete these features too. The last feature which has emptiness is `zipcode`, but it shows very few emptiness so I will be able to use this feature in my model using an imputer.


```python
# Drop neighbourhood, cleaning_fee, security_deposit and juridisction_names
useless = ['neighbourhood', 'cleaning_fee', 'security_deposit', 'jurisdiction_names']
data.drop(useless, axis=1, inplace=True)
```

### Selection on categorical data

Let's have a look at the `data` dataframe to see the remaining features.


```python
data.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>host_total_listings_count</th>
      <th>street</th>
      <th>neighbourhood_cleansed</th>
      <th>zipcode</th>
      <th>latitude</th>
      <th>longitude</th>
      <th>property_type</th>
      <th>room_type</th>
      <th>accommodates</th>
      <th>bathrooms</th>
      <th>...</th>
      <th>availability_60</th>
      <th>availability_90</th>
      <th>availability_365</th>
      <th>calendar_last_scraped</th>
      <th>requires_license</th>
      <th>instant_bookable</th>
      <th>cancellation_policy</th>
      <th>require_guest_profile_picture</th>
      <th>require_guest_phone_verification</th>
      <th>calculated_host_listings_count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.0</td>
      <td>A Thames Street, Kingston upon Thames, England...</td>
      <td>Kingston upon Thames</td>
      <td>KT1 1PE</td>
      <td>51.410036</td>
      <td>-0.306323</td>
      <td>Apartment</td>
      <td>Private room</td>
      <td>1</td>
      <td>1.0</td>
      <td>...</td>
      <td>31</td>
      <td>61</td>
      <td>61</td>
      <td>2017-03-05</td>
      <td>f</td>
      <td>f</td>
      <td>flexible</td>
      <td>f</td>
      <td>f</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1.0</td>
      <td>London Road, Kingston upon Thames, Greater Lon...</td>
      <td>Kingston upon Thames</td>
      <td>KT2 6QS</td>
      <td>51.411482</td>
      <td>-0.290704</td>
      <td>Apartment</td>
      <td>Private room</td>
      <td>2</td>
      <td>1.0</td>
      <td>...</td>
      <td>59</td>
      <td>89</td>
      <td>364</td>
      <td>2017-03-04</td>
      <td>f</td>
      <td>f</td>
      <td>moderate</td>
      <td>f</td>
      <td>f</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1.0</td>
      <td>Kingston Hill, Kingston upon Thames, KT2 7PW, ...</td>
      <td>Kingston upon Thames</td>
      <td>KT2 7PW</td>
      <td>51.415851</td>
      <td>-0.286496</td>
      <td>Apartment</td>
      <td>Private room</td>
      <td>2</td>
      <td>1.0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2017-03-05</td>
      <td>f</td>
      <td>f</td>
      <td>flexible</td>
      <td>f</td>
      <td>f</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1.0</td>
      <td>Canbury Avenue, Kingston upon Thames, KT2 6JR,...</td>
      <td>Kingston upon Thames</td>
      <td>KT2 6JR</td>
      <td>51.415723</td>
      <td>-0.292246</td>
      <td>House</td>
      <td>Private room</td>
      <td>2</td>
      <td>1.5</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2017-03-05</td>
      <td>f</td>
      <td>f</td>
      <td>flexible</td>
      <td>f</td>
      <td>f</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1.0</td>
      <td>Kingston Road, New Malden, England KT3 3RX, Un...</td>
      <td>Kingston upon Thames</td>
      <td>KT3 3RX</td>
      <td>51.404285</td>
      <td>-0.275426</td>
      <td>House</td>
      <td>Private room</td>
      <td>1</td>
      <td>1.0</td>
      <td>...</td>
      <td>59</td>
      <td>89</td>
      <td>179</td>
      <td>2017-03-05</td>
      <td>f</td>
      <td>f</td>
      <td>flexible</td>
      <td>f</td>
      <td>f</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>5 rows Ã 31 columns</p>
</div>



It makes me realise that the `street` feature would need some Natural Language Processing. As I don't have the knowledge needed here, and I think I have enough location information with `neighbourhood_cleansed` and `zipcode`, I decide to delete this feature.
I also decide to delete the dates features, I might decide to add them after if my model is not great.


```python
# Drop street as we have enought localisation info (redundant)
# Drop calendar_last_scraped and calendar_updated (date)

useless = ['street', 'calendar_last_scraped', 'calendar_updated']
data.drop(useless, axis=1, inplace=True)
```

Now, let's have a look at the zipcode feature. The visualisation of the `data` dataframe made me realise that there are lots of different zipcodes, maybe too many?


```python
# Focus on zipcode

from collections import Counter

# Get number of zipcodes
nb_counts = Counter(data.zipcode)
print("Number of Zipcodes:", len(nb_counts))
```

    Number of Zipcodes: 24775
    

Indeed, there are too many zipcodes. If I leave this feature like that, it might cause overfitting. What I decide to do is to regroup the zipcodes. At the moment, they are separated as in the following example: KT1 1PE. I decide to only keep the first part of the zipcodes (KT1) which gives me some less precise location information.


```python
for i in range(0, data.shape[0]):
    zc = data.iloc[i, 2]
    if type(zc) is str:
        zc = zc.split(" ") 
        data.iloc[i, 2] = zc[0]
    else:
        data.iloc[i, 2] = "OTHER"
```


```python
# Get number of zipcodes
nb_counts = Counter(data.zipcode)
print("Number of Zipcodes:", len(nb_counts))
```

    Number of Zipcodes: 1072
    

Now, I only have 1072 different zipcodes, which is much better than before. Let's have a look at the `data` dataframe to be sure that the zipcodes have the correct form.


```python
data.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>host_total_listings_count</th>
      <th>neighbourhood_cleansed</th>
      <th>zipcode</th>
      <th>latitude</th>
      <th>longitude</th>
      <th>property_type</th>
      <th>room_type</th>
      <th>accommodates</th>
      <th>bathrooms</th>
      <th>bedrooms</th>
      <th>...</th>
      <th>availability_30</th>
      <th>availability_60</th>
      <th>availability_90</th>
      <th>availability_365</th>
      <th>requires_license</th>
      <th>instant_bookable</th>
      <th>cancellation_policy</th>
      <th>require_guest_profile_picture</th>
      <th>require_guest_phone_verification</th>
      <th>calculated_host_listings_count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.0</td>
      <td>Kingston upon Thames</td>
      <td>KT1</td>
      <td>51.410036</td>
      <td>-0.306323</td>
      <td>Apartment</td>
      <td>Private room</td>
      <td>1</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>...</td>
      <td>1</td>
      <td>31</td>
      <td>61</td>
      <td>61</td>
      <td>f</td>
      <td>f</td>
      <td>flexible</td>
      <td>f</td>
      <td>f</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1.0</td>
      <td>Kingston upon Thames</td>
      <td>KT2</td>
      <td>51.411482</td>
      <td>-0.290704</td>
      <td>Apartment</td>
      <td>Private room</td>
      <td>2</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>...</td>
      <td>29</td>
      <td>59</td>
      <td>89</td>
      <td>364</td>
      <td>f</td>
      <td>f</td>
      <td>moderate</td>
      <td>f</td>
      <td>f</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1.0</td>
      <td>Kingston upon Thames</td>
      <td>KT2</td>
      <td>51.415851</td>
      <td>-0.286496</td>
      <td>Apartment</td>
      <td>Private room</td>
      <td>2</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>f</td>
      <td>f</td>
      <td>flexible</td>
      <td>f</td>
      <td>f</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1.0</td>
      <td>Kingston upon Thames</td>
      <td>KT2</td>
      <td>51.415723</td>
      <td>-0.292246</td>
      <td>House</td>
      <td>Private room</td>
      <td>2</td>
      <td>1.5</td>
      <td>1.0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>f</td>
      <td>f</td>
      <td>flexible</td>
      <td>f</td>
      <td>f</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1.0</td>
      <td>Kingston upon Thames</td>
      <td>KT3</td>
      <td>51.404285</td>
      <td>-0.275426</td>
      <td>House</td>
      <td>Private room</td>
      <td>1</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>...</td>
      <td>29</td>
      <td>59</td>
      <td>89</td>
      <td>179</td>
      <td>f</td>
      <td>f</td>
      <td>flexible</td>
      <td>f</td>
      <td>f</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>5 rows Ã 28 columns</p>
</div>



Now, I want to know how many appartments are contained in each zipcode.


```python
# Plot number of listings in each zipcode
import pandas as pd
tdf = pd.DataFrame.from_dict(nb_counts, orient='index').sort_values(by=0)
ax = tdf.plot(kind='bar', figsize = (30,8), color = '#E35A5C', alpha = 0.85)
ax.set_title("Zipcodes by Number of Listings")
ax.set_xlabel("Zipcode")
ax.set_ylabel("# of Listings")
plt.show()
```


![png](plots/output_35_0.png)


As we can see, lots of zipcodes only contain less than 100 appartments. Only few zipcodes contains most of the appartments. 
Let's keep these ones.


```python
# Delete zipcodes with less than 100 entries
for i in list(nb_counts):
    if nb_counts[i] < 100:
        del nb_counts[i]
        data = data[data.zipcode != i]
```


```python
# Plot new zipcodes distribution
tdf = pd.DataFrame.from_dict(nb_counts, orient='index').sort_values(by=0)
ax = tdf.plot(kind='bar', figsize = (22,4), color = '#E35A5C', alpha = 0.85)
ax.set_title("Zipcodes by Number of Listings")
ax.set_xlabel("Zipcode")
ax.set_ylabel("# of Listings")

plt.show()

print ('Number of entries removed: ', 53904 - data.shape[0])

```


![png](plots/output_38_0.png)


    Number of entries removed:  6640
    

This distribution is much better, and we only removed 6640 rows from our dataframe which contained about 54000 rows.

Now let's have a look at the distribution for the `neighbourhood_cleansed` feature.


```python
# Focus on neighbourhood_cleansed

# Get number of listings in neighborhoods
nb_counts = Counter(data.neighbourhood_cleansed)
tdf = pd.DataFrame.from_dict(nb_counts, orient='index').sort_values(by=0)

# Plot number of listings in each neighborhood
ax = tdf.plot(kind='bar', figsize = (50,10), color = '#E35A5C', alpha = 0.85)
ax.set_title("Neighborhoods by Number of Listings")
ax.set_xlabel("Neighborhood")
ax.set_ylabel("# of Listings")
plt.show()

print("Number of Neighborhoods:", len(nb_counts))
```


![png](plots/output_40_0.png)


    Number of Neighborhoods: 33
    

The distribution is fine, and there is only 33 neighborhoods. But some only contain around ten appartment, and this is useless for our model. So let's keep the neighborhoods that contain more than 100 appartments.


```python
data.shape
```




    (47264, 28)




```python
# Delete neighborhoods with less than 100 entries
for i in list(nb_counts):
    if nb_counts[i] < 100:
        del nb_counts[i]
        data = data[data.neighbourhood_cleansed != i]

# Plot new neighborhoods distribution
tdf = pd.DataFrame.from_dict(nb_counts, orient='index').sort_values(by=0)
ax = tdf.plot(kind='bar', figsize = (22,4), color = '#E35A5C', alpha = 0.85)
ax.set_title("Neighborhoods by House # (Top 22)")
ax.set_xlabel("Neighborhood")
ax.set_ylabel("# of Listings")

plt.show()

print('Number of entries removed: ', 47264 - data.shape[0])
```


![png](plots/output_43_0.png)


    Number of entries removed:  288
    

By doing this, I only removed 288 rows. I still have more than 46000 rows in my dataframe.

I can now go on with my feature selection by examining multicollinearity.

### Examine multicollinearity

This part of the code will give me a dataframe with correlation coefficients.


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
temp_data = data.copy()
# Delete additional entries with NaN values
temp_data = temp_data.dropna(axis=0)

# Encode categorical data
temp_data = temp_data.apply(encode_categorical)
# Compute matrix of correlation coefficients
corr_matrix = np.corrcoef(temp_data.T)

corr_df = pd.DataFrame(data = corr_matrix, columns = temp_data.columns, 
             index = temp_data.columns)

corr_df
```

    C:\Users\cbarret\AppData\Local\Continuum\anaconda3\lib\site-packages\numpy\lib\function_base.py:3183: RuntimeWarning: invalid value encountered in true_divide
      c /= stddev[:, None]
    C:\Users\cbarret\AppData\Local\Continuum\anaconda3\lib\site-packages\numpy\lib\function_base.py:3184: RuntimeWarning: invalid value encountered in true_divide
      c /= stddev[None, :]
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>host_total_listings_count</th>
      <th>neighbourhood_cleansed</th>
      <th>zipcode</th>
      <th>latitude</th>
      <th>longitude</th>
      <th>property_type</th>
      <th>room_type</th>
      <th>accommodates</th>
      <th>bathrooms</th>
      <th>bedrooms</th>
      <th>...</th>
      <th>availability_30</th>
      <th>availability_60</th>
      <th>availability_90</th>
      <th>availability_365</th>
      <th>requires_license</th>
      <th>instant_bookable</th>
      <th>cancellation_policy</th>
      <th>require_guest_profile_picture</th>
      <th>require_guest_phone_verification</th>
      <th>calculated_host_listings_count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>host_total_listings_count</th>
      <td>1.000000</td>
      <td>-0.013738</td>
      <td>0.051501</td>
      <td>-0.016016</td>
      <td>-0.072001</td>
      <td>0.063519</td>
      <td>-0.131653</td>
      <td>0.200126</td>
      <td>0.166858</td>
      <td>0.211850</td>
      <td>...</td>
      <td>-0.058937</td>
      <td>-0.068261</td>
      <td>-0.074833</td>
      <td>-0.070235</td>
      <td>NaN</td>
      <td>0.150995</td>
      <td>0.164600</td>
      <td>-0.017040</td>
      <td>0.005109</td>
      <td>0.675006</td>
    </tr>
    <tr>
      <th>neighbourhood_cleansed</th>
      <td>-0.013738</td>
      <td>1.000000</td>
      <td>0.077536</td>
      <td>-0.341562</td>
      <td>0.199898</td>
      <td>-0.063524</td>
      <td>-0.006576</td>
      <td>0.023728</td>
      <td>0.004714</td>
      <td>-0.004792</td>
      <td>...</td>
      <td>0.003702</td>
      <td>-0.003835</td>
      <td>-0.004243</td>
      <td>-0.003974</td>
      <td>NaN</td>
      <td>0.025385</td>
      <td>0.046251</td>
      <td>0.003457</td>
      <td>0.000373</td>
      <td>0.034339</td>
    </tr>
    <tr>
      <th>zipcode</th>
      <td>0.051501</td>
      <td>0.077536</td>
      <td>1.000000</td>
      <td>-0.452818</td>
      <td>-0.648251</td>
      <td>-0.024881</td>
      <td>-0.132323</td>
      <td>0.076068</td>
      <td>0.041722</td>
      <td>0.055485</td>
      <td>...</td>
      <td>0.007620</td>
      <td>0.002631</td>
      <td>0.000332</td>
      <td>0.015260</td>
      <td>NaN</td>
      <td>0.000722</td>
      <td>0.056807</td>
      <td>-0.013173</td>
      <td>0.009232</td>
      <td>0.021644</td>
    </tr>
    <tr>
      <th>latitude</th>
      <td>-0.016016</td>
      <td>-0.341562</td>
      <td>-0.452818</td>
      <td>1.000000</td>
      <td>0.137186</td>
      <td>-0.029977</td>
      <td>0.019271</td>
      <td>-0.024787</td>
      <td>-0.027083</td>
      <td>-0.043288</td>
      <td>...</td>
      <td>-0.032824</td>
      <td>-0.031346</td>
      <td>-0.030226</td>
      <td>-0.025894</td>
      <td>NaN</td>
      <td>0.009923</td>
      <td>-0.000559</td>
      <td>0.006033</td>
      <td>-0.004141</td>
      <td>0.001983</td>
    </tr>
    <tr>
      <th>longitude</th>
      <td>-0.072001</td>
      <td>0.199898</td>
      <td>-0.648251</td>
      <td>0.137186</td>
      <td>1.000000</td>
      <td>0.007093</td>
      <td>0.105951</td>
      <td>-0.061036</td>
      <td>-0.048459</td>
      <td>-0.062039</td>
      <td>...</td>
      <td>-0.015442</td>
      <td>-0.012797</td>
      <td>-0.009347</td>
      <td>-0.015773</td>
      <td>NaN</td>
      <td>-0.016999</td>
      <td>-0.020654</td>
      <td>0.018315</td>
      <td>0.001092</td>
      <td>-0.041229</td>
    </tr>
    <tr>
      <th>property_type</th>
      <td>0.063519</td>
      <td>-0.063524</td>
      <td>-0.024881</td>
      <td>-0.029977</td>
      <td>0.007093</td>
      <td>1.000000</td>
      <td>0.200385</td>
      <td>0.061489</td>
      <td>0.241293</td>
      <td>0.191274</td>
      <td>...</td>
      <td>0.033880</td>
      <td>0.040624</td>
      <td>0.037230</td>
      <td>0.028587</td>
      <td>NaN</td>
      <td>0.005611</td>
      <td>-0.073774</td>
      <td>0.008541</td>
      <td>-0.014637</td>
      <td>-0.031336</td>
    </tr>
    <tr>
      <th>room_type</th>
      <td>-0.131653</td>
      <td>-0.006576</td>
      <td>-0.132323</td>
      <td>0.019271</td>
      <td>0.105951</td>
      <td>0.200385</td>
      <td>1.000000</td>
      <td>-0.552309</td>
      <td>-0.133146</td>
      <td>-0.376914</td>
      <td>...</td>
      <td>0.186026</td>
      <td>0.207606</td>
      <td>0.208968</td>
      <td>0.144501</td>
      <td>NaN</td>
      <td>0.032534</td>
      <td>-0.226814</td>
      <td>0.009030</td>
      <td>-0.037406</td>
      <td>-0.143806</td>
    </tr>
    <tr>
      <th>accommodates</th>
      <td>0.200126</td>
      <td>0.023728</td>
      <td>0.076068</td>
      <td>-0.024787</td>
      <td>-0.061036</td>
      <td>0.061489</td>
      <td>-0.552309</td>
      <td>1.000000</td>
      <td>0.458911</td>
      <td>0.759394</td>
      <td>...</td>
      <td>-0.063823</td>
      <td>-0.095185</td>
      <td>-0.094233</td>
      <td>-0.029171</td>
      <td>NaN</td>
      <td>0.034965</td>
      <td>0.240081</td>
      <td>0.001646</td>
      <td>0.055264</td>
      <td>0.203671</td>
    </tr>
    <tr>
      <th>bathrooms</th>
      <td>0.166858</td>
      <td>0.004714</td>
      <td>0.041722</td>
      <td>-0.027083</td>
      <td>-0.048459</td>
      <td>0.241293</td>
      <td>-0.133146</td>
      <td>0.458911</td>
      <td>1.000000</td>
      <td>0.538818</td>
      <td>...</td>
      <td>-0.011958</td>
      <td>-0.018866</td>
      <td>-0.024146</td>
      <td>-0.013673</td>
      <td>NaN</td>
      <td>-0.001259</td>
      <td>0.085671</td>
      <td>-0.015896</td>
      <td>0.003098</td>
      <td>0.128682</td>
    </tr>
    <tr>
      <th>bedrooms</th>
      <td>0.211850</td>
      <td>-0.004792</td>
      <td>0.055485</td>
      <td>-0.043288</td>
      <td>-0.062039</td>
      <td>0.191274</td>
      <td>-0.376914</td>
      <td>0.759394</td>
      <td>0.538818</td>
      <td>1.000000</td>
      <td>...</td>
      <td>-0.070983</td>
      <td>-0.090951</td>
      <td>-0.097159</td>
      <td>-0.069897</td>
      <td>NaN</td>
      <td>-0.039755</td>
      <td>0.130258</td>
      <td>-0.009946</td>
      <td>0.019544</td>
      <td>0.123870</td>
    </tr>
    <tr>
      <th>beds</th>
      <td>0.177466</td>
      <td>0.003395</td>
      <td>0.069171</td>
      <td>-0.027013</td>
      <td>-0.070092</td>
      <td>0.138148</td>
      <td>-0.389782</td>
      <td>0.827682</td>
      <td>0.486604</td>
      <td>0.738863</td>
      <td>...</td>
      <td>-0.045556</td>
      <td>-0.069313</td>
      <td>-0.071514</td>
      <td>-0.015594</td>
      <td>NaN</td>
      <td>0.016289</td>
      <td>0.183095</td>
      <td>-0.000087</td>
      <td>0.032990</td>
      <td>0.156275</td>
    </tr>
    <tr>
      <th>bed_type</th>
      <td>0.016484</td>
      <td>0.016958</td>
      <td>0.007221</td>
      <td>-0.006280</td>
      <td>-0.000442</td>
      <td>0.002990</td>
      <td>-0.079493</td>
      <td>0.059276</td>
      <td>0.044552</td>
      <td>0.059219</td>
      <td>...</td>
      <td>-0.023812</td>
      <td>-0.024986</td>
      <td>-0.023234</td>
      <td>-0.020204</td>
      <td>NaN</td>
      <td>0.022936</td>
      <td>0.027748</td>
      <td>-0.000577</td>
      <td>0.003071</td>
      <td>0.026753</td>
    </tr>
    <tr>
      <th>amenities</th>
      <td>0.123145</td>
      <td>0.022061</td>
      <td>0.108304</td>
      <td>-0.072186</td>
      <td>-0.088790</td>
      <td>-0.031701</td>
      <td>-0.234311</td>
      <td>0.183005</td>
      <td>0.076684</td>
      <td>0.137528</td>
      <td>...</td>
      <td>-0.001473</td>
      <td>-0.011749</td>
      <td>-0.013341</td>
      <td>0.025596</td>
      <td>NaN</td>
      <td>-0.019693</td>
      <td>0.082508</td>
      <td>0.004498</td>
      <td>0.020674</td>
      <td>0.098828</td>
    </tr>
    <tr>
      <th>price</th>
      <td>-0.066191</td>
      <td>-0.023474</td>
      <td>-0.049499</td>
      <td>0.007380</td>
      <td>0.023416</td>
      <td>-0.033272</td>
      <td>0.192117</td>
      <td>-0.265193</td>
      <td>-0.158217</td>
      <td>-0.278200</td>
      <td>...</td>
      <td>0.000102</td>
      <td>0.008496</td>
      <td>0.008873</td>
      <td>-0.007275</td>
      <td>NaN</td>
      <td>0.018753</td>
      <td>-0.093060</td>
      <td>0.000242</td>
      <td>-0.028706</td>
      <td>-0.101950</td>
    </tr>
    <tr>
      <th>guests_included</th>
      <td>-0.030653</td>
      <td>0.012419</td>
      <td>0.015160</td>
      <td>0.005790</td>
      <td>-0.002472</td>
      <td>-0.003820</td>
      <td>-0.289656</td>
      <td>0.480018</td>
      <td>0.194212</td>
      <td>0.348599</td>
      <td>...</td>
      <td>-0.013624</td>
      <td>-0.027157</td>
      <td>-0.023439</td>
      <td>0.029016</td>
      <td>NaN</td>
      <td>-0.019297</td>
      <td>0.196120</td>
      <td>0.022323</td>
      <td>0.094524</td>
      <td>0.060239</td>
    </tr>
    <tr>
      <th>extra_people</th>
      <td>-0.056591</td>
      <td>0.013182</td>
      <td>0.004439</td>
      <td>-0.002158</td>
      <td>0.012007</td>
      <td>0.000438</td>
      <td>0.010600</td>
      <td>0.084464</td>
      <td>0.011485</td>
      <td>0.014340</td>
      <td>...</td>
      <td>0.066816</td>
      <td>0.074323</td>
      <td>0.079871</td>
      <td>0.093986</td>
      <td>NaN</td>
      <td>-0.005465</td>
      <td>0.119186</td>
      <td>0.037090</td>
      <td>0.049904</td>
      <td>-0.032996</td>
    </tr>
    <tr>
      <th>minimum_nights</th>
      <td>0.028281</td>
      <td>-0.008297</td>
      <td>0.013001</td>
      <td>0.017135</td>
      <td>-0.014949</td>
      <td>-0.005073</td>
      <td>-0.115240</td>
      <td>0.047342</td>
      <td>0.043238</td>
      <td>0.079624</td>
      <td>...</td>
      <td>-0.047248</td>
      <td>-0.042537</td>
      <td>-0.042612</td>
      <td>-0.033879</td>
      <td>NaN</td>
      <td>-0.028099</td>
      <td>0.071649</td>
      <td>0.003607</td>
      <td>0.001933</td>
      <td>0.018877</td>
    </tr>
    <tr>
      <th>maximum_nights</th>
      <td>0.102711</td>
      <td>0.017721</td>
      <td>0.023497</td>
      <td>0.000105</td>
      <td>-0.023045</td>
      <td>-0.041167</td>
      <td>-0.065808</td>
      <td>0.078623</td>
      <td>0.023300</td>
      <td>0.027361</td>
      <td>...</td>
      <td>0.006105</td>
      <td>-0.002623</td>
      <td>-0.004924</td>
      <td>0.088314</td>
      <td>NaN</td>
      <td>0.004953</td>
      <td>0.073872</td>
      <td>-0.046274</td>
      <td>-0.039051</td>
      <td>0.167868</td>
    </tr>
    <tr>
      <th>availability_30</th>
      <td>-0.058937</td>
      <td>0.003702</td>
      <td>0.007620</td>
      <td>-0.032824</td>
      <td>-0.015442</td>
      <td>0.033880</td>
      <td>0.186026</td>
      <td>-0.063823</td>
      <td>-0.011958</td>
      <td>-0.070983</td>
      <td>...</td>
      <td>1.000000</td>
      <td>0.951335</td>
      <td>0.909845</td>
      <td>0.662322</td>
      <td>NaN</td>
      <td>-0.040297</td>
      <td>0.013331</td>
      <td>0.033508</td>
      <td>0.028194</td>
      <td>0.040900</td>
    </tr>
    <tr>
      <th>availability_60</th>
      <td>-0.068261</td>
      <td>-0.003835</td>
      <td>0.002631</td>
      <td>-0.031346</td>
      <td>-0.012797</td>
      <td>0.040624</td>
      <td>0.207606</td>
      <td>-0.095185</td>
      <td>-0.018866</td>
      <td>-0.090951</td>
      <td>...</td>
      <td>0.951335</td>
      <td>1.000000</td>
      <td>0.978723</td>
      <td>0.714241</td>
      <td>NaN</td>
      <td>-0.042370</td>
      <td>0.018407</td>
      <td>0.038126</td>
      <td>0.031845</td>
      <td>0.036321</td>
    </tr>
    <tr>
      <th>availability_90</th>
      <td>-0.074833</td>
      <td>-0.004243</td>
      <td>0.000332</td>
      <td>-0.030226</td>
      <td>-0.009347</td>
      <td>0.037230</td>
      <td>0.208968</td>
      <td>-0.094233</td>
      <td>-0.024146</td>
      <td>-0.097159</td>
      <td>...</td>
      <td>0.909845</td>
      <td>0.978723</td>
      <td>1.000000</td>
      <td>0.741673</td>
      <td>NaN</td>
      <td>-0.034473</td>
      <td>0.027107</td>
      <td>0.042597</td>
      <td>0.038359</td>
      <td>0.041097</td>
    </tr>
    <tr>
      <th>availability_365</th>
      <td>-0.070235</td>
      <td>-0.003974</td>
      <td>0.015260</td>
      <td>-0.025894</td>
      <td>-0.015773</td>
      <td>0.028587</td>
      <td>0.144501</td>
      <td>-0.029171</td>
      <td>-0.013673</td>
      <td>-0.069897</td>
      <td>...</td>
      <td>0.662322</td>
      <td>0.714241</td>
      <td>0.741673</td>
      <td>1.000000</td>
      <td>NaN</td>
      <td>-0.027778</td>
      <td>0.095777</td>
      <td>0.063612</td>
      <td>0.076889</td>
      <td>0.085851</td>
    </tr>
    <tr>
      <th>requires_license</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>instant_bookable</th>
      <td>0.150995</td>
      <td>0.025385</td>
      <td>0.000722</td>
      <td>0.009923</td>
      <td>-0.016999</td>
      <td>0.005611</td>
      <td>0.032534</td>
      <td>0.034965</td>
      <td>-0.001259</td>
      <td>-0.039755</td>
      <td>...</td>
      <td>-0.040297</td>
      <td>-0.042370</td>
      <td>-0.034473</td>
      <td>-0.027778</td>
      <td>NaN</td>
      <td>1.000000</td>
      <td>0.044271</td>
      <td>-0.002278</td>
      <td>-0.014941</td>
      <td>0.199434</td>
    </tr>
    <tr>
      <th>cancellation_policy</th>
      <td>0.164600</td>
      <td>0.046251</td>
      <td>0.056807</td>
      <td>-0.000559</td>
      <td>-0.020654</td>
      <td>-0.073774</td>
      <td>-0.226814</td>
      <td>0.240081</td>
      <td>0.085671</td>
      <td>0.130258</td>
      <td>...</td>
      <td>0.013331</td>
      <td>0.018407</td>
      <td>0.027107</td>
      <td>0.095777</td>
      <td>NaN</td>
      <td>0.044271</td>
      <td>1.000000</td>
      <td>0.066632</td>
      <td>0.107026</td>
      <td>0.268678</td>
    </tr>
    <tr>
      <th>require_guest_profile_picture</th>
      <td>-0.017040</td>
      <td>0.003457</td>
      <td>-0.013173</td>
      <td>0.006033</td>
      <td>0.018315</td>
      <td>0.008541</td>
      <td>0.009030</td>
      <td>0.001646</td>
      <td>-0.015896</td>
      <td>-0.009946</td>
      <td>...</td>
      <td>0.033508</td>
      <td>0.038126</td>
      <td>0.042597</td>
      <td>0.063612</td>
      <td>NaN</td>
      <td>-0.002278</td>
      <td>0.066632</td>
      <td>1.000000</td>
      <td>0.700899</td>
      <td>-0.006835</td>
    </tr>
    <tr>
      <th>require_guest_phone_verification</th>
      <td>0.005109</td>
      <td>0.000373</td>
      <td>0.009232</td>
      <td>-0.004141</td>
      <td>0.001092</td>
      <td>-0.014637</td>
      <td>-0.037406</td>
      <td>0.055264</td>
      <td>0.003098</td>
      <td>0.019544</td>
      <td>...</td>
      <td>0.028194</td>
      <td>0.031845</td>
      <td>0.038359</td>
      <td>0.076889</td>
      <td>NaN</td>
      <td>-0.014941</td>
      <td>0.107026</td>
      <td>0.700899</td>
      <td>1.000000</td>
      <td>0.102366</td>
    </tr>
    <tr>
      <th>calculated_host_listings_count</th>
      <td>0.675006</td>
      <td>0.034339</td>
      <td>0.021644</td>
      <td>0.001983</td>
      <td>-0.041229</td>
      <td>-0.031336</td>
      <td>-0.143806</td>
      <td>0.203671</td>
      <td>0.128682</td>
      <td>0.123870</td>
      <td>...</td>
      <td>0.040900</td>
      <td>0.036321</td>
      <td>0.041097</td>
      <td>0.085851</td>
      <td>NaN</td>
      <td>0.199434</td>
      <td>0.268678</td>
      <td>-0.006835</td>
      <td>0.102366</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
<p>28 rows Ã 28 columns</p>
</div>



Here I create a nice plot to visualise the correlation between features, I can then report to the above dataframe to have more details if necessary.


```python
import matplotlib.pyplot as plt

# Display heat map 
plt.figure(figsize=(7, 7))
plt.pcolor(corr_matrix, cmap='RdBu')
plt.xlabel('Predictor Index')
plt.ylabel('Predictor Index')
plt.title('Heatmap of Correlation Matrix')
plt.colorbar()

plt.show()
```


![png](plots/output_49_0.png)


This nice plot makes me realise that `calculated_host_listings_count` is highly correlated with `host_total_listings_count`. So I decide to only keep the last one. I also see that the `availability_*` variables are correlated with each other. I decide to keep `availability_365` as this one is the less correlated with other variables. Finally, I decide to drop `requires_license` which has a weird correlation result and I think won't be useful in my model.


```python
# Drop calculated_host_listings_count as correlated with host_total_listings_count
# Keep availability_365 as availability variables are correlated with each other and this one is less correlated with other variables
# Drop requires_license (weird correlation result and won't impact the results)

useless = ['calculated_host_listings_count', 'availability_30', 'availability_60', 'availability_90', 'requires_license']
data.drop(useless, axis=1, inplace=True)
```

The feature selection is now done. Finally, I have a dataframe with 23 features and almost 47000 rows. I kept 87% of the rows of my initial dataframe and deleted 72 variables.


```python
print(data.shape)
```

    (46976, 23)
    

## Manipulation on price column --> integer

Now, I need to manipulate some features that relate to price as they have a wrong format: they contain the thousand separator (',') and the '$' symbol. Let's get rid of it and transform these features into integer.


```python
# Transform price column
data['price'] = data['price'].str.replace('$', '')
data['price'] = data['price'].str.replace(',', '')
data['price'] = pd.to_numeric(data['price'])
```


```python
# Transform extra_people column
data['extra_people'] = data['extra_people'].str.replace('$', '')
data['extra_people'] = data['extra_people'].str.replace(',', '')
data['extra_people'] = pd.to_numeric(data['extra_people'])
```

The next step is to shuffle the `data` dataframe to ensure a good distribution for the training and testing sets.


```python
from sklearn.utils import shuffle
data = shuffle(data)
```

Let's have a last look at the `data` dataframe:


```python
data.iloc[0:6, 0:10]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>host_total_listings_count</th>
      <th>neighbourhood_cleansed</th>
      <th>zipcode</th>
      <th>latitude</th>
      <th>longitude</th>
      <th>property_type</th>
      <th>room_type</th>
      <th>accommodates</th>
      <th>bathrooms</th>
      <th>bedrooms</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>40281</th>
      <td>1.0</td>
      <td>Tower Hamlets</td>
      <td>E2</td>
      <td>51.530636</td>
      <td>-0.071663</td>
      <td>Apartment</td>
      <td>Entire home/apt</td>
      <td>2</td>
      <td>1.0</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>15230</th>
      <td>1.0</td>
      <td>Lewisham</td>
      <td>SE14</td>
      <td>51.480032</td>
      <td>-0.047274</td>
      <td>House</td>
      <td>Private room</td>
      <td>2</td>
      <td>1.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>33497</th>
      <td>11.0</td>
      <td>Camden</td>
      <td>WC1H</td>
      <td>51.528873</td>
      <td>-0.124509</td>
      <td>Apartment</td>
      <td>Private room</td>
      <td>2</td>
      <td>1.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>33749</th>
      <td>20.0</td>
      <td>Camden</td>
      <td>NW6</td>
      <td>51.553488</td>
      <td>-0.194837</td>
      <td>House</td>
      <td>Entire home/apt</td>
      <td>14</td>
      <td>3.0</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>12492</th>
      <td>1.0</td>
      <td>Southwark</td>
      <td>SE1</td>
      <td>51.501266</td>
      <td>-0.071482</td>
      <td>Apartment</td>
      <td>Entire home/apt</td>
      <td>2</td>
      <td>1.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>18103</th>
      <td>1.0</td>
      <td>Merton</td>
      <td>SW19</td>
      <td>51.434207</td>
      <td>-0.199953</td>
      <td>House</td>
      <td>Private room</td>
      <td>2</td>
      <td>1.0</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
data.iloc[0:6, 11:22]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>bed_type</th>
      <th>amenities</th>
      <th>price</th>
      <th>guests_included</th>
      <th>extra_people</th>
      <th>minimum_nights</th>
      <th>maximum_nights</th>
      <th>availability_365</th>
      <th>instant_bookable</th>
      <th>cancellation_policy</th>
      <th>require_guest_profile_picture</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>40281</th>
      <td>Real Bed</td>
      <td>{TV,Internet,"Wireless Internet",Kitchen,Heati...</td>
      <td>70.0</td>
      <td>1</td>
      <td>0.0</td>
      <td>7</td>
      <td>20</td>
      <td>0</td>
      <td>f</td>
      <td>flexible</td>
      <td>f</td>
    </tr>
    <tr>
      <th>15230</th>
      <td>Real Bed</td>
      <td>{TV,Internet,"Wireless Internet",Kitchen,"Free...</td>
      <td>22.0</td>
      <td>1</td>
      <td>10.0</td>
      <td>5</td>
      <td>50</td>
      <td>0</td>
      <td>f</td>
      <td>moderate</td>
      <td>f</td>
    </tr>
    <tr>
      <th>33497</th>
      <td>Real Bed</td>
      <td>{Internet,"Wireless Internet",Kitchen,"Indoor ...</td>
      <td>37.0</td>
      <td>1</td>
      <td>0.0</td>
      <td>1</td>
      <td>1125</td>
      <td>156</td>
      <td>t</td>
      <td>strict</td>
      <td>f</td>
    </tr>
    <tr>
      <th>33749</th>
      <td>Real Bed</td>
      <td>{TV,Internet,"Wireless Internet",Kitchen,"Pets...</td>
      <td>180.0</td>
      <td>6</td>
      <td>4.0</td>
      <td>1</td>
      <td>1125</td>
      <td>62</td>
      <td>f</td>
      <td>strict</td>
      <td>f</td>
    </tr>
    <tr>
      <th>12492</th>
      <td>Real Bed</td>
      <td>{TV,"Cable TV","Wireless Internet",Kitchen,"Fr...</td>
      <td>100.0</td>
      <td>1</td>
      <td>0.0</td>
      <td>1</td>
      <td>1125</td>
      <td>0</td>
      <td>f</td>
      <td>flexible</td>
      <td>f</td>
    </tr>
    <tr>
      <th>18103</th>
      <td>Real Bed</td>
      <td>{TV,"Wireless Internet","Pets live on this pro...</td>
      <td>45.0</td>
      <td>1</td>
      <td>20.0</td>
      <td>2</td>
      <td>14</td>
      <td>89</td>
      <td>f</td>
      <td>strict</td>
      <td>f</td>
    </tr>
  </tbody>
</table>
</div>



I did not realised before that the `amenities` feature was here. I decide to delete this one because again, it would necessitate some Natural Language Processing.


```python
# Delete amenities feature (too complicated to process for the moment)
data.drop(['amenities'], axis=1, inplace=True)
```


```python
print(data.shape)
```

    (46976, 22)
    

## One Hot Encoding for categorical variables 

Categorical variables need to be One Hot Encoded in order to be converted in several numerical features and used in a Machine Learning model. This method is very well explained in this Kaggle notebook: https://www.kaggle.com/dansbecker/using-categorical-data-with-one-hot-encoding.


```python
# One Hot Encoding for categorical variables
data = pd.get_dummies(data)
```


```python
print(data.shape)
```

    (46976, 193)
    

## Split the data: Features / labels and Training set / testing set

I then split my dataframe into features and labels and training and testing sets.


```python
# Extract features and labels
y = data['price']
X = data.drop('price', axis = 1)

# Training and Testing Sets
from sklearn.model_selection import train_test_split
train_X, test_X, train_y, test_y = train_test_split(X, y, random_state = 0)
```

## Convert to numpy arrays

I convert the `train` and `test` dataframe into numpy arrays so that they can be used to train and test the models.


```python
import numpy as np
train_X = np.array(train_X)
test_X = np.array(test_X)
train_y = np.array(train_y)
test_y = np.array(test_y)
```


```python
train_X.shape, test_X.shape
```




    ((35232, 192), (11744, 192))




```python
%store train_X
%store train_y
%store test_X
%store test_y
```

    Stored 'train_X' (ndarray)
    Stored 'train_y' (ndarray)
    Stored 'test_X' (ndarray)
    Stored 'test_y' (ndarray)
    

Now that the data preprocessing is over, I can start the second part of this work: apply different Machine Learning models.
As a reminder, I decided to apply 3 different models:
* a Random Forest, with the RandomForestRegressor from the Scikit-learn library
* a Gradient Boosting method, with the XGBRegressor from the XGBoost library
* a Neural Network, with the MLPRegressor from the Scikit-learn library.

Each time, I applied the model with its default hyperparameters and I then tuned the model in order to get the best hyperparameters I could.

Let's start with the Random Forest model.

# Apply Random Forest regressor

As I said, the first algorithm I apply is a Random Forest regressor with the default hyperparameters.

## Without hyperparameter tuning

### Create the pipeline and fit the model

I create a pipeline that first impute the missing values, then scale the data and finally apply the model. I then fit this pipeline to the training set.


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




    Pipeline(memory=None,
         steps=[('imputer', Imputer(axis=0, copy=True, missing_values='NaN', strategy='mean', verbose=0)), ('standardscaler', StandardScaler(copy=True, with_mean=True, with_std=True)), ('randomforestregressor', RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=None,
               max_features='au...estimators=10, n_jobs=1,
               oob_score=False, random_state=42, verbose=0, warm_start=False))])




```python
%store my_pipeline_RF
```

    Stored 'my_pipeline_RF' (Pipeline)
    

### Evaluate the model

I evaluate this model on my testing set. I uses 3 different metrics, which are all interpretable in dollars: 
* the mean absolute error
* the median absolute error
* the root-mean-square error

But I decide to evaluate my model with the median absolute error due to the presence of extreme outliers and skewness in the data set.


```python
# Predict prices of the test data
predictions = my_pipeline_RF.predict(test_X)

from sklearn.metrics import mean_absolute_error
# Write the MAE
print("Mean Absolute Error test: " + str(round(mean_absolute_error(predictions, test_y), 2))) 

from sklearn.metrics import median_absolute_error
print("Median Absolute Error test: " + str(round(median_absolute_error(predictions, test_y), 2))) 

from sklearn.metrics import mean_squared_error
from math import sqrt
RMSE = round(sqrt(mean_squared_error(predictions, test_y)), 2)
# Write the RMSE
print("RMSE test: " + str(RMSE)) 
```

    Mean Absolute Error test: 27.09
    Median Absolute Error test: 14.0
    RMSE test: 66.07
    

I decide to evaluate the model on the training set too, to be sure that I have avoided overfitting.


```python
# Predict prices of the train data
predictions_train = my_pipeline_RF.predict(train_X)

from sklearn.metrics import mean_absolute_error
# Write the MEA
print("Mean Absolute Error train: " + str(round(mean_absolute_error(predictions_train, train_y), 2))) 

from sklearn.metrics import median_absolute_error
print("Median Absolute Error train: " + str(round(median_absolute_error(predictions_train, train_y), 2))) 

from sklearn.metrics import mean_squared_error
from math import sqrt
RMSE_train = round(sqrt(mean_squared_error(predictions_train, train_y)), 2)
# Write the RMSE
print("RMSE train: " + str(RMSE_train)) 
```

    Mean Absolute Error train: 11.16
    Median Absolute Error train: 5.1
    RMSE train: 37.08
    

These first results are quite good. To be sure that I have made a good feature selection I decide to have a look at the feature importances which is available with the Random Forest regressor.


```python
# Get numerical feature importances
importances = list(my_pipeline_RF.steps[2][1].feature_importances_)
# List of tuples with variable and importance
feature_list = list(data.columns.drop("price"))
feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(feature_list, importances)]
# Sort the feature importances by most important first
feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)
# Print out the feature and importances 
[print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances[0:22]]
```

    Variable: bedrooms             Importance: 0.18
    Variable: maximum_nights       Importance: 0.11
    Variable: bathrooms            Importance: 0.1
    Variable: availability_365     Importance: 0.1
    Variable: latitude             Importance: 0.07
    Variable: accommodates         Importance: 0.07
    Variable: longitude            Importance: 0.04
    Variable: property_type_Other  Importance: 0.04
    Variable: host_total_listings_count Importance: 0.03
    Variable: beds                 Importance: 0.03
    Variable: zipcode_SW1P         Importance: 0.03
    Variable: room_type_Entire home/apt Importance: 0.03
    Variable: minimum_nights       Importance: 0.02
    Variable: zipcode_SW1W         Importance: 0.02
    Variable: extra_people         Importance: 0.01
    Variable: neighbourhood_cleansed_Kensington and Chelsea Importance: 0.01
    Variable: zipcode_E8           Importance: 0.01
    Variable: zipcode_NW1          Importance: 0.01
    Variable: zipcode_SW3          Importance: 0.01
    Variable: zipcode_W2           Importance: 0.01
    Variable: guests_included      Importance: 0.0
    Variable: neighbourhood_cleansed_Barnet Importance: 0.0
    




    [None,
     None,
     None,
     None,
     None,
     None,
     None,
     None,
     None,
     None,
     None,
     None,
     None,
     None,
     None,
     None,
     None,
     None,
     None,
     None,
     None,
     None]




```python
# List of features sorted from most to least important
sorted_importances = [importance[1] for importance in feature_importances]

# Cumulative importances
cumulative_importances = sum(sorted_importances)

print(cumulative_importances)
```

    0.9300000000000004
    

# Hyperparameter tuning

I had some good results with the default hyperparameters of the Random Forest regressor. But I can improve the results with some hyperparameter tuning. There are two main methods available for this:
* Random search
* Grid search.

You have to provide a parameter grid to these methods. Then, they both try different combinations of parameters within the grid you provided. But the first one only tries several combinations whereas the second one tries all the possible combinations with the grid you provided.

What I have done is that I started with a random search to roughly evaluate a good combination of parameters. Once this is done, I use the grid search to get more precise results.


```python
from pprint import pprint
# Look at parameters used by our current forest
print('Parameters currently used:\n')
pprint(my_pipeline_RF.get_params())
```

    Parameters currently used:
    
    {'imputer': Imputer(axis=0, copy=True, missing_values='NaN', strategy='mean', verbose=0),
     'imputer__axis': 0,
     'imputer__copy': True,
     'imputer__missing_values': 'NaN',
     'imputer__strategy': 'mean',
     'imputer__verbose': 0,
     'memory': None,
     'randomforestregressor': RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=None,
               max_features='auto', max_leaf_nodes=None,
               min_impurity_decrease=0.0, min_impurity_split=None,
               min_samples_leaf=1, min_samples_split=2,
               min_weight_fraction_leaf=0.0, n_estimators=10, n_jobs=1,
               oob_score=False, random_state=42, verbose=0, warm_start=False),
     'randomforestregressor__bootstrap': True,
     'randomforestregressor__criterion': 'mse',
     'randomforestregressor__max_depth': None,
     'randomforestregressor__max_features': 'auto',
     'randomforestregressor__max_leaf_nodes': None,
     'randomforestregressor__min_impurity_decrease': 0.0,
     'randomforestregressor__min_impurity_split': None,
     'randomforestregressor__min_samples_leaf': 1,
     'randomforestregressor__min_samples_split': 2,
     'randomforestregressor__min_weight_fraction_leaf': 0.0,
     'randomforestregressor__n_estimators': 10,
     'randomforestregressor__n_jobs': 1,
     'randomforestregressor__oob_score': False,
     'randomforestregressor__random_state': 42,
     'randomforestregressor__verbose': 0,
     'randomforestregressor__warm_start': False,
     'standardscaler': StandardScaler(copy=True, with_mean=True, with_std=True),
     'standardscaler__copy': True,
     'standardscaler__with_mean': True,
     'standardscaler__with_std': True,
     'steps': [('imputer',
                Imputer(axis=0, copy=True, missing_values='NaN', strategy='mean', verbose=0)),
               ('standardscaler',
                StandardScaler(copy=True, with_mean=True, with_std=True)),
               ('randomforestregressor',
                RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=None,
               max_features='auto', max_leaf_nodes=None,
               min_impurity_decrease=0.0, min_impurity_split=None,
               min_samples_leaf=1, min_samples_split=2,
               min_weight_fraction_leaf=0.0, n_estimators=10, n_jobs=1,
               oob_score=False, random_state=42, verbose=0, warm_start=False))]}
    

## Randomized Search with Cross Validation

### Create the random grid


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

from pprint import pprint
pprint(random_grid)

%store random_grid
```

    {'randomforestregressor__bootstrap': [True, False],
     'randomforestregressor__max_depth': [10, 35, 60, 85, 110, None],
     'randomforestregressor__max_features': ['auto', 'sqrt'],
     'randomforestregressor__min_samples_leaf': [1, 2, 4],
     'randomforestregressor__min_samples_split': [2, 5, 10],
     'randomforestregressor__n_estimators': [10,
                                             109,
                                             208,
                                             307,
                                             406,
                                             505,
                                             604,
                                             703,
                                             802,
                                             901,
                                             1000]}
    Stored 'random_grid' (dict)
    

### Search for best hyperparameters


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

%store rf_random
```

    Stored 'rf_random' (RandomizedSearchCV)
    


```python
%store -r train_X
%store -r train_y
%store -r test_X
%store -r test_y
%store -r my_pipeline_RF
%store -r rf_random
```


```python
# Fit our model
rf_random.fit(train_X, train_y)
```

    Fitting 2 folds for each of 50 candidates, totalling 100 fits
    

    [Parallel(n_jobs=-1)]: Done  33 tasks      | elapsed: 36.0min
    [Parallel(n_jobs=-1)]: Done 100 out of 100 | elapsed: 165.9min finished
    




    RandomizedSearchCV(cv=2, error_score='raise',
              estimator=Pipeline(memory=None,
         steps=[('imputer', Imputer(axis=0, copy=True, missing_values='NaN', strategy='mean', verbose=0)), ('standardscaler', StandardScaler(copy=True, with_mean=True, with_std=True)), ('randomforestregressor', RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=None,
               max_features='au...estimators=10, n_jobs=1,
               oob_score=False, random_state=42, verbose=0, warm_start=False))]),
              fit_params=None, iid=True, n_iter=50, n_jobs=-1,
              param_distributions={'randomforestregressor__n_estimators': [10, 109, 208, 307, 406, 505, 604, 703, 802, 901, 1000], 'randomforestregressor__max_features': ['auto', 'sqrt'], 'randomforestregressor__max_depth': [10, 35, 60, 85, 110, None], 'randomforestregressor__min_samples_split': [2, 5, 10], 'randomforestregressor__min_samples_leaf': [1, 2, 4], 'randomforestregressor__bootstrap': [True, False]},
              pre_dispatch='2*n_jobs', random_state=42, refit=True,
              return_train_score='warn', scoring='neg_median_absolute_error',
              verbose=2)




```python
rf_random.best_params_
```




    {'randomforestregressor__bootstrap': True,
     'randomforestregressor__max_depth': 85,
     'randomforestregressor__max_features': 'auto',
     'randomforestregressor__min_samples_leaf': 1,
     'randomforestregressor__min_samples_split': 2,
     'randomforestregressor__n_estimators': 802}



## Grid Search with Cross Validation

### Create the grid


```python
from sklearn.model_selection import GridSearchCV
# Create the parameter grid based on the results of random search 
param_grid = {
    'randomforestregressor__bootstrap': [True],
    'randomforestregressor__max_depth': [80, 85, 90], 
    'randomforestregressor__max_features': ['auto'],
    'randomforestregressor__min_samples_leaf': [1],
    'randomforestregressor__min_samples_split': [2, 4],
    'randomforestregressor__n_estimators': [780, 800, 820] 
}
```


```python
%store param_grid
```

    Stored 'param_grid' (dict)
    

### Search for best hyperparameters


```python
# Instantiate the grid search model
grid_search = GridSearchCV(estimator = my_pipeline_RF, 
                           param_grid = param_grid, 
                           cv = 3, n_jobs = -1, verbose = 2, 
                           scoring = 'neg_median_absolute_error')

%store grid_search
```

    Stored 'grid_search' (GridSearchCV)
    


```python
# Fit the grid search to the data
grid_search.fit(train_X, train_y)
```

    Fitting 3 folds for each of 18 candidates, totalling 54 fits
    

    [Parallel(n_jobs=-1)]: Done  33 tasks      | elapsed: 141.8min
    [Parallel(n_jobs=-1)]: Done  54 out of  54 | elapsed: 218.5min finished
    




    GridSearchCV(cv=3, error_score='raise',
           estimator=Pipeline(memory=None,
         steps=[('imputer', Imputer(axis=0, copy=True, missing_values='NaN', strategy='mean', verbose=0)), ('standardscaler', StandardScaler(copy=True, with_mean=True, with_std=True)), ('randomforestregressor', RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=None,
               max_features='au...estimators=10, n_jobs=1,
               oob_score=False, random_state=42, verbose=0, warm_start=False))]),
           fit_params=None, iid=True, n_jobs=-1,
           param_grid={'randomforestregressor__bootstrap': [True], 'randomforestregressor__max_depth': [80, 85, 90], 'randomforestregressor__max_features': ['auto'], 'randomforestregressor__min_samples_leaf': [1], 'randomforestregressor__min_samples_split': [2, 4], 'randomforestregressor__n_estimators': [780, 800, 820]},
           pre_dispatch='2*n_jobs', refit=True, return_train_score='warn',
           scoring='neg_median_absolute_error', verbose=2)




```python
grid_search.best_params_
```




    {'randomforestregressor__bootstrap': True,
     'randomforestregressor__max_depth': 80,
     'randomforestregressor__max_features': 'auto',
     'randomforestregressor__min_samples_leaf': 1,
     'randomforestregressor__min_samples_split': 2,
     'randomforestregressor__n_estimators': 820}




```python
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import StandardScaler

# Create the pipeline (imputer + scaler + regressor)
my_pipeline_RF_grid = make_pipeline(Imputer(), StandardScaler(),
                                      RandomForestRegressor(random_state=42,
                                                            bootstrap = True,
                                                            max_depth = 80,
                                                            max_features = 'auto',
                                                            min_samples_leaf = 1,
                                                            min_samples_split = 2,
                                                            n_estimators = 820))

# Fit the model
my_pipeline_RF_grid.fit(train_X, train_y)
```




    Pipeline(memory=None,
         steps=[('imputer', Imputer(axis=0, copy=True, missing_values='NaN', strategy='mean', verbose=0)), ('standardscaler', StandardScaler(copy=True, with_mean=True, with_std=True)), ('randomforestregressor', RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=80,
               max_features='auto...stimators=820, n_jobs=1,
               oob_score=False, random_state=42, verbose=0, warm_start=False))])



### Evaluate the model

Now let's evaluate the tuned model.


```python
# Predict prices of the test data
predictions_grid = my_pipeline_RF_grid.predict(test_X)

from sklearn.metrics import mean_absolute_error
# Write the MEA
print("Mean Absolute Error test: " + str(round(mean_absolute_error(predictions_grid, test_y), 2))) 

from sklearn.metrics import median_absolute_error
print("Median Absolute Error test: " + str(round(median_absolute_error(predictions_grid, test_y), 2))) 

from sklearn.metrics import mean_squared_error
from math import sqrt
RMSE = round(sqrt(mean_squared_error(predictions_grid, test_y)), 2)
# Write the RMSE
print("RMSE test: " + str(RMSE)) 
```

    Mean Absolute Error test: 25.34
    Median Absolute Error test: 13.35
    RMSE test: 59.74
    

I get better results with the tuned model than with default hyperparameters, but the improvement is not amazing. Maybe I will have a better precision if I use another model.


# Apply XGBRegressor

Let's try with the XGBoost gradient boosting model. This model often produces really good results in Kaggle competitions. The first step is to use it with the default hyperparameters.


```python
from xgboost import XGBRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Imputer
# Multi-layer Perceptron is sensitive to feature scaling, so it is highly recommended to scale your data
from sklearn.preprocessing import StandardScaler

# Create the pipeline: Imputation + Scale + MLP regressor
my_pipeline_XGB = make_pipeline(Imputer(), StandardScaler(), 
                                XGBRegressor(random_state = 42))

# Fit the model
my_pipeline_XGB.fit(train_X, train_y)
%store my_pipeline_XGB
```

    Stored 'my_pipeline_XGB' (Pipeline)
    


```python
# Predict prices of the test data
predictions = my_pipeline_XGB.predict(test_X)

from sklearn.metrics import mean_absolute_error
# Write the MAE
print("Mean Absolute Error test: " + str(round(mean_absolute_error(predictions, test_y), 2))) 

from sklearn.metrics import median_absolute_error
print("Median Absolute Error test: " + str(round(median_absolute_error(predictions, test_y), 2))) 

from sklearn.metrics import mean_squared_error
from math import sqrt
RMSE = round(sqrt(mean_squared_error(predictions, test_y)), 2)
# Write the RMSE
print("RMSE test: " + str(RMSE)) 
```

    Mean Absolute Error test: 27.41
    Median Absolute Error test: 16.15
    RMSE test: 57.81
    


```python
# Predict prices of the train data
predictions_train = my_pipeline_XGB.predict(train_X)

from sklearn.metrics import mean_absolute_error
# Write the MEA
print("Mean Absolute Error train: " + str(round(mean_absolute_error(predictions_train, train_y), 2))) 

from sklearn.metrics import median_absolute_error
print("Median Absolute Error train: " + str(round(median_absolute_error(predictions_train, train_y), 2))) 

from sklearn.metrics import mean_squared_error
from math import sqrt
RMSE_train = round(sqrt(mean_squared_error(predictions_train, train_y)), 2)
# Write the RMSE
print("RMSE train: " + str(RMSE_train)) 
```

    Mean Absolute Error train: 27.54
    Median Absolute Error train: 16.14
    RMSE train: 71.46
    

For the moment, the tuned and even not tuned Random Forest models give better results. I want to see if hyperparameter tuning will make this model better than the Random Forest one.

# Hyperparameter tuning


```python
from pprint import pprint
# Look at parameters used by our current model
print('Parameters currently used:\n')
pprint(my_pipeline_XGB.get_params())
```

    Parameters currently used:
    
    {'imputer': Imputer(axis=0, copy=True, missing_values='NaN', strategy='mean', verbose=0),
     'imputer__axis': 0,
     'imputer__copy': True,
     'imputer__missing_values': 'NaN',
     'imputer__strategy': 'mean',
     'imputer__verbose': 0,
     'memory': None,
     'standardscaler': StandardScaler(copy=True, with_mean=True, with_std=True),
     'standardscaler__copy': True,
     'standardscaler__with_mean': True,
     'standardscaler__with_std': True,
     'steps': [('imputer',
                Imputer(axis=0, copy=True, missing_values='NaN', strategy='mean', verbose=0)),
               ('standardscaler',
                StandardScaler(copy=True, with_mean=True, with_std=True)),
               ('xgbregressor',
                XGBRegressor(base_score=0.5, booster='gbtree', colsample_bylevel=1,
           colsample_bytree=1, gamma=0, learning_rate=0.1, max_delta_step=0,
           max_depth=3, min_child_weight=1, missing=None, n_estimators=100,
           n_jobs=1, nthread=None, objective='reg:linear', random_state=42,
           reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,
           silent=True, subsample=1))],
     'xgbregressor': XGBRegressor(base_score=0.5, booster='gbtree', colsample_bylevel=1,
           colsample_bytree=1, gamma=0, learning_rate=0.1, max_delta_step=0,
           max_depth=3, min_child_weight=1, missing=None, n_estimators=100,
           n_jobs=1, nthread=None, objective='reg:linear', random_state=42,
           reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,
           silent=True, subsample=1),
     'xgbregressor__base_score': 0.5,
     'xgbregressor__booster': 'gbtree',
     'xgbregressor__colsample_bylevel': 1,
     'xgbregressor__colsample_bytree': 1,
     'xgbregressor__gamma': 0,
     'xgbregressor__learning_rate': 0.1,
     'xgbregressor__max_delta_step': 0,
     'xgbregressor__max_depth': 3,
     'xgbregressor__min_child_weight': 1,
     'xgbregressor__missing': None,
     'xgbregressor__n_estimators': 100,
     'xgbregressor__n_jobs': 1,
     'xgbregressor__nthread': None,
     'xgbregressor__objective': 'reg:linear',
     'xgbregressor__random_state': 42,
     'xgbregressor__reg_alpha': 0,
     'xgbregressor__reg_lambda': 1,
     'xgbregressor__scale_pos_weight': 1,
     'xgbregressor__seed': None,
     'xgbregressor__silent': True,
     'xgbregressor__subsample': 1}
    

## Grid search


```python
from sklearn.model_selection import GridSearchCV
# Create the parameter grid based on the results of random search 
param_grid = {'xgbregressor__learning_rate': [0.1, 0.05], 
              'xgbregressor__max_depth': [5, 7, 9],
              'xgbregressor__n_estimators': [100, 500, 900]}
%store param_grid
```

    Stored 'param_grid' (dict)
    


```python
# Instantiate the grid search model
grid_search = GridSearchCV(estimator = my_pipeline_XGB,
                           param_grid = param_grid, 
                           cv = 3, n_jobs = -1, verbose = 2, 
                           scoring = 'neg_median_absolute_error')

%store grid_search
```

    Stored 'grid_search' (GridSearchCV)
    


```python
# Fit the grid search to the data
grid_search.fit(train_X, train_y)
```

    Fitting 3 folds for each of 18 candidates, totalling 54 fits
    

    [Parallel(n_jobs=-1)]: Done  33 tasks      | elapsed: 40.0min
    [Parallel(n_jobs=-1)]: Done  54 out of  54 | elapsed: 76.0min finished
    




    GridSearchCV(cv=3, error_score='raise',
           estimator=Pipeline(memory=None,
         steps=[('imputer', Imputer(axis=0, copy=True, missing_values='NaN', strategy='mean', verbose=0)), ('standardscaler', StandardScaler(copy=True, with_mean=True, with_std=True)), ('xgbregressor', XGBRegressor(base_score=0.5, booster='gbtree', colsample_bylevel=1,
           colsample_bytree=1, gamma=0, lea...
           reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,
           silent=True, subsample=1))]),
           fit_params=None, iid=True, n_jobs=-1,
           param_grid={'xgbregressor__learning_rate': [0.1, 0.05], 'xgbregressor__max_depth': [5, 7, 9], 'xgbregressor__n_estimators': [100, 500, 900]},
           pre_dispatch='2*n_jobs', refit=True, return_train_score='warn',
           scoring='neg_median_absolute_error', verbose=2)




```python
grid_search.best_score_
```




    -13.640864690144857




```python
grid_search.best_params_
```




    {'xgbregressor__learning_rate': 0.05,
     'xgbregressor__max_depth': 9,
     'xgbregressor__n_estimators': 900}




```python
from xgboost import XGBRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Imputer
# Multi-layer Perceptron is sensitive to feature scaling, so it is highly recommended to scale your data
from sklearn.preprocessing import StandardScaler

# Create the pipeline: Imputation + Scale + MLP regressor
my_pipeline_XGB_grid = make_pipeline(Imputer(), StandardScaler(), 
                                     XGBRegressor(random_state = 42,
                                                  learning_rate = 0.05,
                                                  max_depth = 9,
                                                  n_estimators = 900))

# Fit the model
my_pipeline_XGB_grid.fit(train_X, train_y)
%store my_pipeline_XGB_grid
```

    Stored 'my_pipeline_XGB_grid' (Pipeline)
    

## Evaluate the tuned model


```python
# Predict prices of test data
predictions_grid = my_pipeline_XGB_grid.predict(test_X)
%store predictions_grid

from sklearn.metrics import mean_absolute_error
# Write the MAE
print("Mean Absolute Error : " + str(round(mean_absolute_error(predictions_grid, test_y), 2)))

from sklearn.metrics import median_absolute_error
print("Median Absolute Error test: " + str(round(median_absolute_error(predictions_grid, test_y), 2))) 

from sklearn.metrics import mean_squared_error
from math import sqrt
RMSE = round(sqrt(mean_squared_error(predictions_grid, test_y)), 2)
# Write the RMSE
print("RMSE test: " + str(RMSE)) 
```

    Stored 'predictions_grid' (ndarray)
    Mean Absolute Error : 25.37
    Median Absolute Error test: 13.53
    RMSE test: 78.18
    


```python
# Predict prices of the train data
predictions_train_grid = my_pipeline_XGB_grid.predict(train_X)

from sklearn.metrics import mean_absolute_error
# Write the MEA
print("Mean Absolute Error train: " + str(round(mean_absolute_error(predictions_train_grid, train_y), 2))) 

from sklearn.metrics import median_absolute_error
print("Median Absolute Error train: " + str(round(median_absolute_error(predictions_train_grid, train_y), 2))) 

from sklearn.metrics import mean_squared_error
from math import sqrt
RMSE_train = round(sqrt(mean_squared_error(predictions_train_grid, train_y)), 2)
# Write the RMSE
print("RMSE train: " + str(RMSE_train)) 
```

    Mean Absolute Error train: 13.48
    Median Absolute Error train: 8.94
    RMSE train: 21.06
    

The tuned XGBoost model gives better results than the not tuned Random Forest model and gives almost the same results than the tuned Random Forest model. But for the moment, the tuned Random Forest model is the best one.


# Apply MLP regressor


Now let's try a Neural Network, or to be more precise, a multilayer perceptron which is a class of Neural Network. I apply this regressor with default hyperparameters except from the maximum numer of iteration in order to let him run until the end.


```python
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Imputer
# Multi-layer Perceptron is sensitive to feature scaling, so it is highly recommended to scale your data
from sklearn.preprocessing import StandardScaler

# Create the pipeline: Imputation + Scale + Feature Selection + MLP regressor
my_pipeline_NN = make_pipeline(Imputer(), StandardScaler(), 
                               MLPRegressor(random_state = 42,
                                            max_iter = 400,
                                            verbose = True))

# Fit the model
my_pipeline_NN.fit(train_X, train_y)
```

    Iteration 1, loss = 10492.41418106
    Iteration 2, loss = 6536.03461242
    Iteration 3, loss = 5030.78339190
    Iteration 4, loss = 4800.55724882
    Iteration 5, loss = 4723.13221960
    Iteration 6, loss = 4673.96808543
    Iteration 7, loss = 4641.57051599
    Iteration 8, loss = 4612.84974409
    Iteration 9, loss = 4585.72440393
    Iteration 10, loss = 4559.15715541
    Iteration 11, loss = 4536.04631480
    Iteration 12, loss = 4511.37422972
    Iteration 13, loss = 4487.52690173
    Iteration 14, loss = 4463.52338122
    Iteration 15, loss = 4443.28330483
    Iteration 16, loss = 4419.13173947
    Iteration 17, loss = 4407.62356507
    Iteration 18, loss = 4380.29172370
    Iteration 19, loss = 4363.31869460
    Iteration 20, loss = 4343.85800349
    Iteration 21, loss = 4329.67833578
    Iteration 22, loss = 4310.76057020
    Iteration 23, loss = 4294.20649263
    Iteration 24, loss = 4280.43553779
    Iteration 25, loss = 4265.38350690
    Iteration 26, loss = 4250.74433356
    Iteration 27, loss = 4238.21501377
    Iteration 28, loss = 4227.25031136
    Iteration 29, loss = 4210.09332730
    Iteration 30, loss = 4198.83256844
    Iteration 31, loss = 4185.89556756
    Iteration 32, loss = 4176.26223348
    Iteration 33, loss = 4163.09525347
    Iteration 34, loss = 4150.92840212
    Iteration 35, loss = 4142.86704412
    Iteration 36, loss = 4128.46324949
    Iteration 37, loss = 4119.58738016
    Iteration 38, loss = 4107.32203383
    Iteration 39, loss = 4095.61799268
    Iteration 40, loss = 4087.42491081
    Iteration 41, loss = 4075.27234369
    Iteration 42, loss = 4066.08053223
    Iteration 43, loss = 4058.14054964
    Iteration 44, loss = 4047.08179336
    Iteration 45, loss = 4037.25915327
    Iteration 46, loss = 4026.78732887
    Iteration 47, loss = 4020.68938922
    Iteration 48, loss = 4007.72581644
    Iteration 49, loss = 4003.46523003
    Iteration 50, loss = 3993.71659271
    Iteration 51, loss = 3982.23517638
    Iteration 52, loss = 3973.50484591
    Iteration 53, loss = 3962.79712038
    Iteration 54, loss = 3958.27758728
    Iteration 55, loss = 3946.82221812
    Iteration 56, loss = 3939.78819822
    Iteration 57, loss = 3931.52206491
    Iteration 58, loss = 3932.41585960
    Iteration 59, loss = 3914.85369506
    Iteration 60, loss = 3908.03624648
    Iteration 61, loss = 3900.78388874
    Iteration 62, loss = 3893.55260357
    Iteration 63, loss = 3888.33522917
    Iteration 64, loss = 3878.34556041
    Iteration 65, loss = 3871.04201065
    Iteration 66, loss = 3863.50612069
    Iteration 67, loss = 3856.69420895
    Iteration 68, loss = 3848.47205384
    Iteration 69, loss = 3845.16280149
    Iteration 70, loss = 3837.92354937
    Iteration 71, loss = 3827.42774532
    Iteration 72, loss = 3822.92630776
    Iteration 73, loss = 3813.69028447
    Iteration 74, loss = 3810.42917709
    Iteration 75, loss = 3800.40866223
    Iteration 76, loss = 3796.71152960
    Iteration 77, loss = 3784.81550308
    Iteration 78, loss = 3785.03574197
    Iteration 79, loss = 3776.04856856
    Iteration 80, loss = 3768.60981803
    Iteration 81, loss = 3765.34873298
    Iteration 82, loss = 3751.95023289
    Iteration 83, loss = 3748.40359440
    Iteration 84, loss = 3741.90214212
    Iteration 85, loss = 3736.16117448
    Iteration 86, loss = 3731.82732066
    Iteration 87, loss = 3726.57571598
    Iteration 88, loss = 3715.99661701
    Iteration 89, loss = 3714.71968139
    Iteration 90, loss = 3707.34095719
    Iteration 91, loss = 3700.04463281
    Iteration 92, loss = 3690.50380986
    Iteration 93, loss = 3689.83623665
    Iteration 94, loss = 3681.80791318
    Iteration 95, loss = 3678.53079105
    Iteration 96, loss = 3669.73609751
    Iteration 97, loss = 3669.03567335
    Iteration 98, loss = 3659.14070368
    Iteration 99, loss = 3653.17648594
    Iteration 100, loss = 3652.26206594
    Iteration 101, loss = 3640.15930247
    Iteration 102, loss = 3637.90452072
    Iteration 103, loss = 3631.36147954
    Iteration 104, loss = 3626.75145103
    Iteration 105, loss = 3621.10899573
    Iteration 106, loss = 3614.97522644
    Iteration 107, loss = 3608.61215357
    Iteration 108, loss = 3606.53177020
    Iteration 109, loss = 3599.65897200
    Iteration 110, loss = 3592.96170688
    Iteration 111, loss = 3587.99968642
    Iteration 112, loss = 3582.61601169
    Iteration 113, loss = 3577.62407230
    Iteration 114, loss = 3573.80462969
    Iteration 115, loss = 3569.56666702
    Iteration 116, loss = 3563.12034930
    Iteration 117, loss = 3557.87937972
    Iteration 118, loss = 3550.42806384
    Iteration 119, loss = 3548.00310881
    Iteration 120, loss = 3545.91109167
    Iteration 121, loss = 3539.61676008
    Iteration 122, loss = 3532.62911810
    Iteration 123, loss = 3526.46880243
    Iteration 124, loss = 3523.94817407
    Iteration 125, loss = 3517.01563517
    Iteration 126, loss = 3511.39842577
    Iteration 127, loss = 3507.68103859
    Iteration 128, loss = 3505.27778328
    Iteration 129, loss = 3498.04880614
    Iteration 130, loss = 3493.15748064
    Iteration 131, loss = 3491.30464507
    Iteration 132, loss = 3486.00567645
    Iteration 133, loss = 3480.30041433
    Iteration 134, loss = 3477.10829476
    Iteration 135, loss = 3473.49457001
    Iteration 136, loss = 3468.59272826
    Iteration 137, loss = 3465.36176287
    Iteration 138, loss = 3462.49420803
    Iteration 139, loss = 3455.52242101
    Iteration 140, loss = 3451.05841171
    Iteration 141, loss = 3445.99100304
    Iteration 142, loss = 3442.63446617
    Iteration 143, loss = 3439.12706119
    Iteration 144, loss = 3434.79847326
    Iteration 145, loss = 3431.23649815
    Iteration 146, loss = 3439.43370487
    Iteration 147, loss = 3421.95707290
    Iteration 148, loss = 3415.37866746
    Iteration 149, loss = 3408.88721864
    Iteration 150, loss = 3404.47891620
    Iteration 151, loss = 3403.51332768
    Iteration 152, loss = 3398.66461276
    Iteration 153, loss = 3396.96311948
    Iteration 154, loss = 3394.06321149
    Iteration 155, loss = 3390.06515840
    Iteration 156, loss = 3392.22514570
    Iteration 157, loss = 3379.08008192
    Iteration 158, loss = 3382.45199949
    Iteration 159, loss = 3379.30669854
    Iteration 160, loss = 3370.65789314
    Iteration 161, loss = 3371.97605602
    Iteration 162, loss = 3365.48013461
    Iteration 163, loss = 3364.28349104
    Iteration 164, loss = 3359.73161575
    Iteration 165, loss = 3356.67365240
    Iteration 166, loss = 3349.13003624
    Iteration 167, loss = 3352.61788039
    Iteration 168, loss = 3343.38009534
    Iteration 169, loss = 3339.32015594
    Iteration 170, loss = 3337.33702128
    Iteration 171, loss = 3331.58963289
    Iteration 172, loss = 3330.39460636
    Iteration 173, loss = 3327.40367472
    Iteration 174, loss = 3320.51804000
    Iteration 175, loss = 3316.64077785
    Iteration 176, loss = 3315.08358421
    Iteration 177, loss = 3314.02102161
    Iteration 178, loss = 3309.65524308
    Iteration 179, loss = 3311.25372451
    Iteration 180, loss = 3304.31028659
    Iteration 181, loss = 3298.41702436
    Iteration 182, loss = 3296.03241052
    Iteration 183, loss = 3293.76455963
    Iteration 184, loss = 3290.48964342
    Iteration 185, loss = 3284.90045206
    Iteration 186, loss = 3282.68836180
    Iteration 187, loss = 3276.90859218
    Iteration 188, loss = 3276.57053005
    Iteration 189, loss = 3269.50459449
    Iteration 190, loss = 3272.96867867
    Iteration 191, loss = 3269.28304970
    Iteration 192, loss = 3258.25919524
    Iteration 193, loss = 3260.32555892
    Iteration 194, loss = 3258.33814408
    Iteration 195, loss = 3256.00608430
    Iteration 196, loss = 3248.90741117
    Iteration 197, loss = 3246.39883857
    Iteration 198, loss = 3240.05312902
    Iteration 199, loss = 3243.66324627
    Iteration 200, loss = 3242.46281985
    Iteration 201, loss = 3235.42228886
    Iteration 202, loss = 3235.68526935
    Iteration 203, loss = 3232.88839057
    Iteration 204, loss = 3224.83167799
    Iteration 205, loss = 3222.79884908
    Iteration 206, loss = 3220.17116771
    Iteration 207, loss = 3216.66906796
    Iteration 208, loss = 3211.44735502
    Iteration 209, loss = 3213.12534859
    Iteration 210, loss = 3203.17643273
    Iteration 211, loss = 3207.83405182
    Iteration 212, loss = 3205.37969636
    Iteration 213, loss = 3202.25109742
    Iteration 214, loss = 3199.24127268
    Iteration 215, loss = 3194.97354918
    Iteration 216, loss = 3191.93474285
    Iteration 217, loss = 3187.28390243
    Iteration 218, loss = 3182.09664888
    Iteration 219, loss = 3188.32198996
    Iteration 220, loss = 3180.57205441
    Iteration 221, loss = 3173.56995590
    Iteration 222, loss = 3176.73612023
    Iteration 223, loss = 3173.56820601
    Iteration 224, loss = 3167.84061061
    Iteration 225, loss = 3167.13798652
    Iteration 226, loss = 3156.97982904
    Iteration 227, loss = 3161.87884204
    Iteration 228, loss = 3158.97614606
    Iteration 229, loss = 3154.86552929
    Iteration 230, loss = 3155.93087230
    Iteration 231, loss = 3147.69407649
    Iteration 232, loss = 3148.39132383
    Iteration 233, loss = 3142.16978089
    Iteration 234, loss = 3142.13796039
    Iteration 235, loss = 3141.86493320
    Iteration 236, loss = 3134.47228770
    Iteration 237, loss = 3134.24202159
    Iteration 238, loss = 3131.12587305
    Iteration 239, loss = 3126.75502744
    Iteration 240, loss = 3127.08085497
    Iteration 241, loss = 3127.54202213
    Iteration 242, loss = 3120.64297074
    Iteration 243, loss = 3118.76277010
    Iteration 244, loss = 3115.38656733
    Iteration 245, loss = 3114.67695942
    Iteration 246, loss = 3112.39170395
    Iteration 247, loss = 3103.46228882
    Iteration 248, loss = 3110.81830787
    Iteration 249, loss = 3100.66584355
    Iteration 250, loss = 3103.96442778
    Iteration 251, loss = 3097.78140663
    Iteration 252, loss = 3094.71092047
    Iteration 253, loss = 3103.60049290
    Iteration 254, loss = 3091.86017343
    Iteration 255, loss = 3089.60701725
    Iteration 256, loss = 3083.75011805
    Iteration 257, loss = 3083.19794733
    Iteration 258, loss = 3079.43916792
    Iteration 259, loss = 3081.25579462
    Iteration 260, loss = 3076.31668567
    Iteration 261, loss = 3073.52507087
    Iteration 262, loss = 3073.43665141
    Iteration 263, loss = 3070.94309316
    Iteration 264, loss = 3066.50040269
    Iteration 265, loss = 3065.39134880
    Iteration 266, loss = 3059.26315212
    Iteration 267, loss = 3061.09830053
    Iteration 268, loss = 3054.32926090
    Iteration 269, loss = 3057.89151585
    Iteration 270, loss = 3050.26474545
    Iteration 271, loss = 3048.75847605
    Iteration 272, loss = 3047.58609899
    Iteration 273, loss = 3042.69109886
    Iteration 274, loss = 3044.18340494
    Iteration 275, loss = 3041.31169256
    Iteration 276, loss = 3039.58894758
    Iteration 277, loss = 3035.22187652
    Iteration 278, loss = 3036.08851411
    Iteration 279, loss = 3028.91607953
    Iteration 280, loss = 3026.81882819
    Iteration 281, loss = 3029.04078809
    Iteration 282, loss = 3024.20159254
    Iteration 283, loss = 3025.13078056
    Iteration 284, loss = 3021.92859916
    Iteration 285, loss = 3018.94570762
    Iteration 286, loss = 3017.91595546
    Iteration 287, loss = 3014.35161359
    Iteration 288, loss = 3012.58358740
    Iteration 289, loss = 3010.01351118
    Iteration 290, loss = 3008.39585163
    Iteration 291, loss = 3005.29290874
    Iteration 292, loss = 3004.84732487
    Iteration 293, loss = 3001.58221336
    Iteration 294, loss = 3002.68628218
    Iteration 295, loss = 2995.67026353
    Iteration 296, loss = 2997.76328235
    Iteration 297, loss = 2990.26554201
    Iteration 298, loss = 2991.02611601
    Iteration 299, loss = 2989.93947905
    Iteration 300, loss = 2982.29472036
    Iteration 301, loss = 2986.02778593
    Iteration 302, loss = 2980.95732051
    Iteration 303, loss = 2981.14934025
    Iteration 304, loss = 2979.92745017
    Iteration 305, loss = 2979.78733246
    Iteration 306, loss = 2973.31954922
    Iteration 307, loss = 2974.86305274
    Iteration 308, loss = 2969.18891938
    Iteration 309, loss = 2966.85009784
    Iteration 310, loss = 2965.96129348
    Iteration 311, loss = 2968.78985489
    Iteration 312, loss = 2960.21432731
    Iteration 313, loss = 2955.49474541
    Iteration 314, loss = 2958.63439592
    Iteration 315, loss = 2957.85806347
    Iteration 316, loss = 2955.50272449
    Training loss did not improve more than tol=0.000100 for two consecutive epochs. Stopping.
    




    Pipeline(memory=None,
         steps=[('imputer', Imputer(axis=0, copy=True, missing_values='NaN', strategy='mean', verbose=0)), ('standardscaler', StandardScaler(copy=True, with_mean=True, with_std=True)), ('mlpregressor', MLPRegressor(activation='relu', alpha=0.0001, batch_size='auto', beta_1=0.9,
           beta_2=0.999, early_sto...
           solver='adam', tol=0.0001, validation_fraction=0.1, verbose=True,
           warm_start=False))])



## Evaluate our model


```python
# Predict prices of the test data
predictions = my_pipeline_NN.predict(test_X)

from sklearn.metrics import mean_absolute_error
# Write the MAE
print("Mean Absolute Error test: " + str(round(mean_absolute_error(predictions, test_y), 2))) 

from sklearn.metrics import median_absolute_error
print("Median Absolute Error test: " + str(round(median_absolute_error(predictions, test_y), 2))) 

from sklearn.metrics import mean_squared_error
from math import sqrt
RMSE = round(sqrt(mean_squared_error(predictions, test_y)), 2)
# Write the RMSE
print("RMSE test: " + str(RMSE)) 

%store my_pipeline_NN
%store predictions
```

    Mean Absolute Error test: 33.04
    Median Absolute Error test: 19.49
    RMSE test: 64.27
    Stored 'my_pipeline_NN' (Pipeline)
    Stored 'predictions' (ndarray)
    


```python
# Predict prices of the train data
predictions_train = my_pipeline_NN.predict(train_X)

from sklearn.metrics import mean_absolute_error
# Write the MEA
print("Mean Absolute Error train: " + str(round(mean_absolute_error(predictions_train, train_y), 2))) 

from sklearn.metrics import median_absolute_error
print("Median Absolute Error test: " + str(round(median_absolute_error(predictions_train, train_y), 2))) 

from sklearn.metrics import mean_squared_error
from math import sqrt
RMSE_train = round(sqrt(mean_squared_error(predictions_train, train_y)), 2)
# Write the RMSE
print("RMSE train: " + str(RMSE_train)) 

%store predictions_train
```

    Mean Absolute Error train: 27.2
    Median Absolute Error test: 16.61
    RMSE train: 76.38
    Stored 'predictions_train' (ndarray)
    

The results are not very good compared to the two previous models. Let's try to tune this neural network, maybe the default parameters are very bad for the data.

# Improve hyperparameters


```python
# Which parameters in my pipeline?
my_pipeline_NN.get_params()
```




    {'imputer': Imputer(axis=0, copy=True, missing_values='NaN', strategy='mean', verbose=0),
     'imputer__axis': 0,
     'imputer__copy': True,
     'imputer__missing_values': 'NaN',
     'imputer__strategy': 'mean',
     'imputer__verbose': 0,
     'memory': None,
     'mlpregressor': MLPRegressor(activation='relu', alpha=0.0001, batch_size='auto', beta_1=0.9,
            beta_2=0.999, early_stopping=False, epsilon=1e-08,
            hidden_layer_sizes=(100,), learning_rate='constant',
            learning_rate_init=0.001, max_iter=400, momentum=0.9,
            nesterovs_momentum=True, power_t=0.5, random_state=42, shuffle=True,
            solver='adam', tol=0.0001, validation_fraction=0.1, verbose=True,
            warm_start=False),
     'mlpregressor__activation': 'relu',
     'mlpregressor__alpha': 0.0001,
     'mlpregressor__batch_size': 'auto',
     'mlpregressor__beta_1': 0.9,
     'mlpregressor__beta_2': 0.999,
     'mlpregressor__early_stopping': False,
     'mlpregressor__epsilon': 1e-08,
     'mlpregressor__hidden_layer_sizes': (100,),
     'mlpregressor__learning_rate': 'constant',
     'mlpregressor__learning_rate_init': 0.001,
     'mlpregressor__max_iter': 400,
     'mlpregressor__momentum': 0.9,
     'mlpregressor__nesterovs_momentum': True,
     'mlpregressor__power_t': 0.5,
     'mlpregressor__random_state': 42,
     'mlpregressor__shuffle': True,
     'mlpregressor__solver': 'adam',
     'mlpregressor__tol': 0.0001,
     'mlpregressor__validation_fraction': 0.1,
     'mlpregressor__verbose': True,
     'mlpregressor__warm_start': False,
     'standardscaler': StandardScaler(copy=True, with_mean=True, with_std=True),
     'standardscaler__copy': True,
     'standardscaler__with_mean': True,
     'standardscaler__with_std': True,
     'steps': [('imputer',
       Imputer(axis=0, copy=True, missing_values='NaN', strategy='mean', verbose=0)),
      ('standardscaler', StandardScaler(copy=True, with_mean=True, with_std=True)),
      ('mlpregressor',
       MLPRegressor(activation='relu', alpha=0.0001, batch_size='auto', beta_1=0.9,
              beta_2=0.999, early_stopping=False, epsilon=1e-08,
              hidden_layer_sizes=(100,), learning_rate='constant',
              learning_rate_init=0.001, max_iter=400, momentum=0.9,
              nesterovs_momentum=True, power_t=0.5, random_state=42, shuffle=True,
              solver='adam', tol=0.0001, validation_fraction=0.1, verbose=True,
              warm_start=False))]}



I did a random search before the grid search but I decided not to include it in the notebook to make it more concise and clearer.

## Grid search - Activation: logistic, tanh


```python
from sklearn.model_selection import GridSearchCV
# Create the parameter grid based on the results of random search 
param_grid = {
    'mlpregressor__activation': ['logistic', 'tanh'],
    'mlpregressor__solver': ['sgd', 'adam'],
    'mlpregressor__early_stopping': [True, False],
    'mlpregressor__hidden_layer_sizes': [(50,), (100,)],
    'mlpregressor__learning_rate_init': [0.001, 0.0001],
}
%store param_grid
```

    Stored 'param_grid' (dict)
    


```python
# Instantiate the grid search model
grid_search = GridSearchCV(estimator = my_pipeline_NN,
                           param_grid = param_grid, 
                           cv = 3, n_jobs = -1, verbose = 2,
                           scoring = 'neg_median_absolute_error')

%store grid_search
```

    Stored 'grid_search' (GridSearchCV)
    


```python
# Fit the grid search to the data
grid_search.fit(train_X, train_y)
```

    Fitting 3 folds for each of 32 candidates, totalling 96 fits
    

    [Parallel(n_jobs=-1)]: Done  33 tasks      | elapsed: 11.5min
    [Parallel(n_jobs=-1)]: Done  96 out of  96 | elapsed: 42.7min finished
    

    Iteration 1, loss = 12365.33617118
    Validation score: -0.937196
    Iteration 2, loss = 12258.92656260
    Validation score: -0.914348
    Iteration 3, loss = 12137.62984836
    Validation score: -0.888118
    Iteration 4, loss = 12001.03568810
    Validation score: -0.858977
    Iteration 5, loss = 11854.14995127
    Validation score: -0.828284
    Iteration 6, loss = 11703.11602306
    Validation score: -0.797159
    Iteration 7, loss = 11551.81977378
    Validation score: -0.766160
    Iteration 8, loss = 11401.97749265
    Validation score: -0.735443
    Iteration 9, loss = 11253.92390169
    Validation score: -0.704980
    Iteration 10, loss = 11107.30596336
    Validation score: -0.674745
    Iteration 11, loss = 10962.24129882
    Validation score: -0.644778
    Iteration 12, loss = 10818.65996750
    Validation score: -0.615148
    Iteration 13, loss = 10677.17168712
    Validation score: -0.585960
    Iteration 14, loss = 10538.34917241
    Validation score: -0.557336
    Iteration 15, loss = 10402.56061390
    Validation score: -0.529516
    Iteration 16, loss = 10270.31448554
    Validation score: -0.502348
    Iteration 17, loss = 10141.97194954
    Validation score: -0.476119
    Iteration 18, loss = 10017.75691617
    Validation score: -0.450782
    Iteration 19, loss = 9897.75851920
    Validation score: -0.426382
    Iteration 20, loss = 9782.02949136
    Validation score: -0.402807
    Iteration 21, loss = 9670.63465322
    Validation score: -0.380221
    Iteration 22, loss = 9563.46361110
    Validation score: -0.358504
    Iteration 23, loss = 9460.47149935
    Validation score: -0.337718
    Iteration 24, loss = 9361.52511966
    Validation score: -0.317717
    Iteration 25, loss = 9266.48372661
    Validation score: -0.298647
    Iteration 26, loss = 9175.23319237
    Validation score: -0.280239
    Iteration 27, loss = 9087.84795319
    Validation score: -0.262653
    Iteration 28, loss = 9003.94042188
    Validation score: -0.245749
    Iteration 29, loss = 8923.30607547
    Validation score: -0.229617
    Iteration 30, loss = 8845.62061515
    Validation score: -0.213970
    Iteration 31, loss = 8770.74216987
    Validation score: -0.198962
    Iteration 32, loss = 8698.48134854
    Validation score: -0.184460
    Iteration 33, loss = 8628.70864004
    Validation score: -0.170450
    Iteration 34, loss = 8561.27880824
    Validation score: -0.156950
    Iteration 35, loss = 8496.07701534
    Validation score: -0.143939
    Iteration 36, loss = 8432.98432886
    Validation score: -0.131286
    Iteration 37, loss = 8371.99344900
    Validation score: -0.119095
    Iteration 38, loss = 8312.87189140
    Validation score: -0.107318
    Iteration 39, loss = 8255.71935524
    Validation score: -0.095926
    Iteration 40, loss = 8200.30983825
    Validation score: -0.084935
    Iteration 41, loss = 8146.56455772
    Validation score: -0.074204
    Iteration 42, loss = 8094.40518687
    Validation score: -0.063865
    Iteration 43, loss = 8043.74345145
    Validation score: -0.053843
    Iteration 44, loss = 7994.55783715
    Validation score: -0.044091
    Iteration 45, loss = 7946.82822585
    Validation score: -0.034709
    Iteration 46, loss = 7900.37118877
    Validation score: -0.025502
    Iteration 47, loss = 7855.21995002
    Validation score: -0.016583
    Iteration 48, loss = 7811.19902415
    Validation score: -0.007979
    Iteration 49, loss = 7768.50047531
    Validation score: 0.000501
    Iteration 50, loss = 7726.90540784
    Validation score: 0.008686
    Iteration 51, loss = 7686.45936824
    Validation score: 0.016592
    Iteration 52, loss = 7647.16981802
    Validation score: 0.024291
    Iteration 53, loss = 7608.90183796
    Validation score: 0.031838
    Iteration 54, loss = 7571.68075989
    Validation score: 0.039109
    Iteration 55, loss = 7535.46143319
    Validation score: 0.046171
    Iteration 56, loss = 7500.19146583
    Validation score: 0.053008
    Iteration 57, loss = 7465.90793467
    Validation score: 0.059803
    Iteration 58, loss = 7432.60915323
    Validation score: 0.066296
    Iteration 59, loss = 7400.20673128
    Validation score: 0.072630
    Iteration 60, loss = 7368.64044899
    Validation score: 0.078704
    Iteration 61, loss = 7338.00453765
    Validation score: 0.084702
    Iteration 62, loss = 7308.30755470
    Validation score: 0.090472
    Iteration 63, loss = 7279.48390714
    Validation score: 0.096069
    Iteration 64, loss = 7251.42775251
    Validation score: 0.101476
    Iteration 65, loss = 7223.99891306
    Validation score: 0.106808
    Iteration 66, loss = 7197.56752633
    Validation score: 0.111891
    Iteration 67, loss = 7171.99590326
    Validation score: 0.116855
    Iteration 68, loss = 7147.01078849
    Validation score: 0.121664
    Iteration 69, loss = 7122.61480382
    Validation score: 0.126344
    Iteration 70, loss = 7098.69556832
    Validation score: 0.130974
    Iteration 71, loss = 7075.23749534
    Validation score: 0.135494
    Iteration 72, loss = 7052.17323353
    Validation score: 0.139925
    Iteration 73, loss = 7029.48515578
    Validation score: 0.144238
    Iteration 74, loss = 7007.03065370
    Validation score: 0.148547
    Iteration 75, loss = 6984.68653942
    Validation score: 0.152779
    Iteration 76, loss = 6962.54261146
    Validation score: 0.157061
    Iteration 77, loss = 6940.51448976
    Validation score: 0.161335
    Iteration 78, loss = 6918.50416273
    Validation score: 0.165512
    Iteration 79, loss = 6896.48330770
    Validation score: 0.169686
    Iteration 80, loss = 6874.47955389
    Validation score: 0.173841
    Iteration 81, loss = 6852.50836475
    Validation score: 0.177957
    Iteration 82, loss = 6830.66874197
    Validation score: 0.182131
    Iteration 83, loss = 6808.92546320
    Validation score: 0.186237
    Iteration 84, loss = 6787.50586283
    Validation score: 0.190287
    Iteration 85, loss = 6766.19510079
    Validation score: 0.194226
    Iteration 86, loss = 6745.11271223
    Validation score: 0.198153
    Iteration 87, loss = 6724.26838805
    Validation score: 0.202033
    Iteration 88, loss = 6703.53346060
    Validation score: 0.205923
    Iteration 89, loss = 6683.28688617
    Validation score: 0.209701
    Iteration 90, loss = 6663.24659260
    Validation score: 0.213424
    Iteration 91, loss = 6643.52667223
    Validation score: 0.217125
    Iteration 92, loss = 6624.13401851
    Validation score: 0.220700
    Iteration 93, loss = 6605.12056243
    Validation score: 0.224234
    Iteration 94, loss = 6586.30349488
    Validation score: 0.227753
    Iteration 95, loss = 6567.81754283
    Validation score: 0.231105
    Iteration 96, loss = 6549.79782186
    Validation score: 0.234498
    Iteration 97, loss = 6531.97619104
    Validation score: 0.237681
    Iteration 98, loss = 6514.52690310
    Validation score: 0.240918
    Iteration 99, loss = 6497.21713448
    Validation score: 0.244114
    Iteration 100, loss = 6480.16091323
    Validation score: 0.247279
    Iteration 101, loss = 6463.31703815
    Validation score: 0.250379
    Iteration 102, loss = 6446.75245099
    Validation score: 0.253424
    Iteration 103, loss = 6430.50103487
    Validation score: 0.256450
    Iteration 104, loss = 6414.45533294
    Validation score: 0.259386
    Iteration 105, loss = 6398.65353067
    Validation score: 0.262305
    Iteration 106, loss = 6382.93668961
    Validation score: 0.265152
    Iteration 107, loss = 6367.63190244
    Validation score: 0.268018
    Iteration 108, loss = 6352.59504195
    Validation score: 0.270741
    Iteration 109, loss = 6337.80766837
    Validation score: 0.273472
    Iteration 110, loss = 6323.18471384
    Validation score: 0.276090
    Iteration 111, loss = 6308.68624683
    Validation score: 0.278722
    Iteration 112, loss = 6294.33035312
    Validation score: 0.281266
    Iteration 113, loss = 6280.19294767
    Validation score: 0.283802
    Iteration 114, loss = 6266.26247251
    Validation score: 0.286307
    Iteration 115, loss = 6252.54878510
    Validation score: 0.288741
    Iteration 116, loss = 6238.98620104
    Validation score: 0.291196
    Iteration 117, loss = 6225.71184972
    Validation score: 0.293481
    Iteration 118, loss = 6212.61608908
    Validation score: 0.295847
    Iteration 119, loss = 6199.58156017
    Validation score: 0.298191
    Iteration 120, loss = 6186.79504091
    Validation score: 0.300466
    Iteration 121, loss = 6174.26129970
    Validation score: 0.302627
    Iteration 122, loss = 6161.85633101
    Validation score: 0.304781
    Iteration 123, loss = 6149.56691081
    Validation score: 0.306989
    Iteration 124, loss = 6137.47492781
    Validation score: 0.309111
    Iteration 125, loss = 6125.60834220
    Validation score: 0.311165
    Iteration 126, loss = 6113.71411155
    Validation score: 0.313228
    Iteration 127, loss = 6102.10316629
    Validation score: 0.315304
    Iteration 128, loss = 6090.52957042
    Validation score: 0.317323
    Iteration 129, loss = 6079.00854136
    Validation score: 0.319315
    Iteration 130, loss = 6067.79037422
    Validation score: 0.321248
    Iteration 131, loss = 6056.60260272
    Validation score: 0.323200
    Iteration 132, loss = 6045.52919107
    Validation score: 0.325126
    Iteration 133, loss = 6034.60401224
    Validation score: 0.327034
    Iteration 134, loss = 6023.83650280
    Validation score: 0.328851
    Iteration 135, loss = 6013.17072157
    Validation score: 0.330672
    Iteration 136, loss = 6002.61053938
    Validation score: 0.332531
    Iteration 137, loss = 5992.28678729
    Validation score: 0.334300
    Iteration 138, loss = 5981.96491218
    Validation score: 0.336040
    Iteration 139, loss = 5971.81014402
    Validation score: 0.337793
    Iteration 140, loss = 5961.85750763
    Validation score: 0.339486
    Iteration 141, loss = 5951.96602553
    Validation score: 0.341134
    Iteration 142, loss = 5942.16557181
    Validation score: 0.342839
    Iteration 143, loss = 5932.53631927
    Validation score: 0.344392
    Iteration 144, loss = 5923.00724991
    Validation score: 0.346040
    Iteration 145, loss = 5913.54966999
    Validation score: 0.347677
    Iteration 146, loss = 5904.17020396
    Validation score: 0.349274
    Iteration 147, loss = 5894.70347098
    Validation score: 0.350683
    Iteration 148, loss = 5885.59242341
    Validation score: 0.352269
    Iteration 149, loss = 5876.49284059
    Validation score: 0.353764
    Iteration 150, loss = 5867.56108113
    Validation score: 0.355263
    Iteration 151, loss = 5858.62778683
    Validation score: 0.356758
    Iteration 152, loss = 5849.82209194
    Validation score: 0.358215
    Iteration 153, loss = 5841.10402451
    Validation score: 0.359670
    Iteration 154, loss = 5832.42139752
    Validation score: 0.361174
    Iteration 155, loss = 5823.75467059
    Validation score: 0.362570
    Iteration 156, loss = 5815.17635526
    Validation score: 0.363980
    Iteration 157, loss = 5806.60385880
    Validation score: 0.365415
    Iteration 158, loss = 5798.13552983
    Validation score: 0.366786
    Iteration 159, loss = 5789.89083318
    Validation score: 0.368178
    Iteration 160, loss = 5781.67885824
    Validation score: 0.369497
    Iteration 161, loss = 5773.61196270
    Validation score: 0.370802
    Iteration 162, loss = 5765.39778373
    Validation score: 0.372094
    Iteration 163, loss = 5757.33863016
    Validation score: 0.373425
    Iteration 164, loss = 5749.25658832
    Validation score: 0.374611
    Iteration 165, loss = 5741.37073635
    Validation score: 0.375929
    Iteration 166, loss = 5733.57941156
    Validation score: 0.377075
    Iteration 167, loss = 5725.81148402
    Validation score: 0.378287
    Iteration 168, loss = 5718.20641974
    Validation score: 0.379544
    Iteration 169, loss = 5710.64365188
    Validation score: 0.380696
    Iteration 170, loss = 5703.21731023
    Validation score: 0.381900
    Iteration 171, loss = 5695.79184861
    Validation score: 0.383026
    Iteration 172, loss = 5688.59160629
    Validation score: 0.384118
    Iteration 173, loss = 5681.39381390
    Validation score: 0.385254
    Iteration 174, loss = 5674.18813247
    Validation score: 0.386347
    Iteration 175, loss = 5667.16609371
    Validation score: 0.387436
    Iteration 176, loss = 5660.22146210
    Validation score: 0.388522
    Iteration 177, loss = 5653.36401399
    Validation score: 0.389555
    Iteration 178, loss = 5646.73318244
    Validation score: 0.390574
    Iteration 179, loss = 5639.87713611
    Validation score: 0.391666
    Iteration 180, loss = 5633.23959550
    Validation score: 0.392671
    Iteration 181, loss = 5626.55561484
    Validation score: 0.393736
    Iteration 182, loss = 5619.94063729
    Validation score: 0.394554
    Iteration 183, loss = 5613.71659900
    Validation score: 0.395496
    Iteration 184, loss = 5606.98879103
    Validation score: 0.396485
    Iteration 185, loss = 5600.80951445
    Validation score: 0.397361
    Iteration 186, loss = 5594.45398151
    Validation score: 0.398369
    Iteration 187, loss = 5588.24044205
    Validation score: 0.399292
    Iteration 188, loss = 5582.08579409
    Validation score: 0.400185
    Iteration 189, loss = 5576.01648851
    Validation score: 0.401009
    Iteration 190, loss = 5569.86114214
    Validation score: 0.401905
    Iteration 191, loss = 5564.06229346
    Validation score: 0.402825
    Iteration 192, loss = 5557.98275165
    Validation score: 0.403640
    Iteration 193, loss = 5552.23355415
    Validation score: 0.404403
    Iteration 194, loss = 5546.49807091
    Validation score: 0.405196
    Iteration 195, loss = 5540.77041597
    Validation score: 0.406002
    Iteration 196, loss = 5535.14189539
    Validation score: 0.406846
    Iteration 197, loss = 5529.50040725
    Validation score: 0.407595
    Iteration 198, loss = 5523.96765099
    Validation score: 0.408359
    Iteration 199, loss = 5518.50362379
    Validation score: 0.409110
    Iteration 200, loss = 5513.01215904
    Validation score: 0.409887
    Iteration 201, loss = 5507.72310493
    Validation score: 0.410538
    Iteration 202, loss = 5502.42512804
    Validation score: 0.411293
    Iteration 203, loss = 5497.12394289
    Validation score: 0.411948
    Iteration 204, loss = 5491.98156348
    Validation score: 0.412670
    Iteration 205, loss = 5486.73260455
    Validation score: 0.413413
    Iteration 206, loss = 5481.59140313
    Validation score: 0.414084
    Iteration 207, loss = 5476.43894566
    Validation score: 0.414815
    Iteration 208, loss = 5471.47649446
    Validation score: 0.415512
    Iteration 209, loss = 5466.45886821
    Validation score: 0.416254
    Iteration 210, loss = 5461.36830099
    Validation score: 0.416876
    Iteration 211, loss = 5456.52239208
    Validation score: 0.417591
    Iteration 212, loss = 5451.55135689
    Validation score: 0.418285
    Iteration 213, loss = 5446.74304519
    Validation score: 0.418977
    Iteration 214, loss = 5441.86689015
    Validation score: 0.419588
    Iteration 215, loss = 5437.18008011
    Validation score: 0.420275
    Iteration 216, loss = 5432.28647872
    Validation score: 0.420917
    Iteration 217, loss = 5427.57965886
    Validation score: 0.421553
    Iteration 218, loss = 5422.97879958
    Validation score: 0.422189
    Iteration 219, loss = 5418.28690320
    Validation score: 0.422784
    Iteration 220, loss = 5413.59247877
    Validation score: 0.423404
    Iteration 221, loss = 5409.12638511
    Validation score: 0.424030
    Iteration 222, loss = 5404.49207221
    Validation score: 0.424654
    Iteration 223, loss = 5400.04177846
    Validation score: 0.425195
    Iteration 224, loss = 5395.61795243
    Validation score: 0.425779
    Iteration 225, loss = 5391.26397131
    Validation score: 0.426403
    Iteration 226, loss = 5386.89169128
    Validation score: 0.427000
    Iteration 227, loss = 5382.43809283
    Validation score: 0.427506
    Iteration 228, loss = 5378.24317138
    Validation score: 0.428137
    Iteration 229, loss = 5373.95258815
    Validation score: 0.428674
    Iteration 230, loss = 5369.83186885
    Validation score: 0.429240
    Iteration 231, loss = 5365.60406736
    Validation score: 0.429749
    Iteration 232, loss = 5361.41335831
    Validation score: 0.430307
    Iteration 233, loss = 5357.25381633
    Validation score: 0.430904
    Iteration 234, loss = 5353.03526612
    Validation score: 0.431446
    Iteration 235, loss = 5348.88245068
    Validation score: 0.431930
    Iteration 236, loss = 5344.93524625
    Validation score: 0.432511
    Iteration 237, loss = 5340.83437266
    Validation score: 0.433017
    Iteration 238, loss = 5336.91522930
    Validation score: 0.433629
    Iteration 239, loss = 5332.95591936
    Validation score: 0.434030
    Iteration 240, loss = 5329.01896623
    Validation score: 0.434604
    Iteration 241, loss = 5324.88487479
    Validation score: 0.434981
    Iteration 242, loss = 5321.18689372
    Validation score: 0.435453
    Iteration 243, loss = 5317.24994130
    Validation score: 0.435929
    Iteration 244, loss = 5313.30734559
    Validation score: 0.436372
    Iteration 245, loss = 5309.50316366
    Validation score: 0.437013
    Iteration 246, loss = 5305.67036024
    Validation score: 0.437391
    Iteration 247, loss = 5301.90736946
    Validation score: 0.437870
    Iteration 248, loss = 5298.11764775
    Validation score: 0.438279
    Iteration 249, loss = 5294.45233936
    Validation score: 0.438782
    Iteration 250, loss = 5290.89879623
    Validation score: 0.439216
    Iteration 251, loss = 5287.09416297
    Validation score: 0.439723
    Iteration 252, loss = 5283.38525207
    Validation score: 0.440172
    Iteration 253, loss = 5279.83981296
    Validation score: 0.440573
    Iteration 254, loss = 5276.33839881
    Validation score: 0.441187
    Iteration 255, loss = 5272.70087877
    Validation score: 0.441478
    Iteration 256, loss = 5269.17801673
    Validation score: 0.441978
    Iteration 257, loss = 5265.79227851
    Validation score: 0.442260
    Iteration 258, loss = 5262.23358686
    Validation score: 0.442788
    Iteration 259, loss = 5258.89909519
    Validation score: 0.443227
    Iteration 260, loss = 5255.42895097
    Validation score: 0.443585
    Iteration 261, loss = 5252.27806829
    Validation score: 0.444017
    Iteration 262, loss = 5248.85174036
    Validation score: 0.444442
    Iteration 263, loss = 5245.65708715
    Validation score: 0.444768
    Iteration 264, loss = 5242.11409062
    Validation score: 0.445261
    Iteration 265, loss = 5238.87249658
    Validation score: 0.445678
    Iteration 266, loss = 5235.59832550
    Validation score: 0.446061
    Iteration 267, loss = 5232.38630355
    Validation score: 0.446501
    Iteration 268, loss = 5229.13583023
    Validation score: 0.447005
    Iteration 269, loss = 5225.87369031
    Validation score: 0.447261
    Iteration 270, loss = 5222.70638268
    Validation score: 0.447687
    Iteration 271, loss = 5219.49001126
    Validation score: 0.448106
    Iteration 272, loss = 5216.18203035
    Validation score: 0.448493
    Iteration 273, loss = 5213.03364798
    Validation score: 0.448847
    Iteration 274, loss = 5209.90951175
    Validation score: 0.449419
    Iteration 275, loss = 5206.70190560
    Validation score: 0.449484
    Iteration 276, loss = 5203.71591283
    Validation score: 0.449901
    Iteration 277, loss = 5200.53676790
    Validation score: 0.450362
    Iteration 278, loss = 5197.56321373
    Validation score: 0.450703
    Iteration 279, loss = 5194.42296936
    Validation score: 0.451062
    Iteration 280, loss = 5191.33540196
    Validation score: 0.451362
    Iteration 281, loss = 5188.33338518
    Validation score: 0.451745
    Iteration 282, loss = 5185.44982017
    Validation score: 0.452159
    Iteration 283, loss = 5182.35191876
    Validation score: 0.452568
    Iteration 284, loss = 5179.47808095
    Validation score: 0.452874
    Iteration 285, loss = 5176.53203324
    Validation score: 0.453238
    Iteration 286, loss = 5173.47161839
    Validation score: 0.453579
    Iteration 287, loss = 5170.61997462
    Validation score: 0.453826
    Iteration 288, loss = 5167.67281453
    Validation score: 0.454247
    Iteration 289, loss = 5164.87658108
    Validation score: 0.454660
    Iteration 290, loss = 5162.02441356
    Validation score: 0.454890
    Iteration 291, loss = 5159.08528624
    Validation score: 0.455370
    Iteration 292, loss = 5156.31272337
    Validation score: 0.455572
    Iteration 293, loss = 5153.20212458
    Validation score: 0.455873
    Iteration 294, loss = 5150.45700491
    Validation score: 0.456237
    Iteration 295, loss = 5147.56883546
    Validation score: 0.456596
    Iteration 296, loss = 5144.73541611
    Validation score: 0.456891
    Iteration 297, loss = 5142.10637014
    Validation score: 0.457223
    Iteration 298, loss = 5139.24782409
    Validation score: 0.457573
    Iteration 299, loss = 5136.60060050
    Validation score: 0.457786
    Iteration 300, loss = 5133.83401079
    Validation score: 0.458275
    Iteration 301, loss = 5131.03793592
    Validation score: 0.458475
    Iteration 302, loss = 5128.50451791
    Validation score: 0.458777
    Iteration 303, loss = 5125.66779116
    Validation score: 0.459047
    Iteration 304, loss = 5123.07696274
    Validation score: 0.459364
    Iteration 305, loss = 5120.44158280
    Validation score: 0.459657
    Iteration 306, loss = 5117.92541588
    Validation score: 0.459730
    Iteration 307, loss = 5115.34407097
    Validation score: 0.460247
    Iteration 308, loss = 5112.55979974
    Validation score: 0.460344
    Iteration 309, loss = 5110.06751305
    Validation score: 0.460601
    Iteration 310, loss = 5107.34366911
    Validation score: 0.460974
    Iteration 311, loss = 5104.81413587
    Validation score: 0.461166
    Iteration 312, loss = 5102.22875702
    Validation score: 0.461590
    Iteration 313, loss = 5099.50027094
    Validation score: 0.461616
    Iteration 314, loss = 5097.04865508
    Validation score: 0.461889
    Iteration 315, loss = 5094.52895422
    Validation score: 0.462250
    Iteration 316, loss = 5091.97046031
    Validation score: 0.462431
    Iteration 317, loss = 5089.47812246
    Validation score: 0.462700
    Iteration 318, loss = 5087.00132700
    Validation score: 0.462919
    Iteration 319, loss = 5084.56060778
    Validation score: 0.463299
    Iteration 320, loss = 5082.12897922
    Validation score: 0.463436
    Iteration 321, loss = 5079.56508303
    Validation score: 0.463767
    Iteration 322, loss = 5077.08255174
    Validation score: 0.463966
    Iteration 323, loss = 5074.74143466
    Validation score: 0.464319
    Iteration 324, loss = 5072.13796917
    Validation score: 0.464544
    Iteration 325, loss = 5069.79212285
    Validation score: 0.464766
    Iteration 326, loss = 5067.35953934
    Validation score: 0.465023
    Iteration 327, loss = 5064.83868462
    Validation score: 0.465272
    Iteration 328, loss = 5062.48446438
    Validation score: 0.465510
    Iteration 329, loss = 5059.96154676
    Validation score: 0.465727
    Iteration 330, loss = 5057.54857992
    Validation score: 0.465946
    Iteration 331, loss = 5055.24798640
    Validation score: 0.466300
    Iteration 332, loss = 5052.87713268
    Validation score: 0.466453
    Iteration 333, loss = 5050.52523751
    Validation score: 0.466648
    Iteration 334, loss = 5048.15924131
    Validation score: 0.466958
    Iteration 335, loss = 5045.89526188
    Validation score: 0.467104
    Iteration 336, loss = 5043.56131634
    Validation score: 0.467330
    Iteration 337, loss = 5041.15855124
    Validation score: 0.467708
    Iteration 338, loss = 5038.95740458
    Validation score: 0.467874
    Iteration 339, loss = 5036.56765781
    Validation score: 0.468089
    Iteration 340, loss = 5034.12970440
    Validation score: 0.468267
    Iteration 341, loss = 5031.71902307
    Validation score: 0.468592
    Iteration 342, loss = 5029.52776858
    Validation score: 0.468757
    Iteration 343, loss = 5027.29538840
    Validation score: 0.469000
    Iteration 344, loss = 5024.90527375
    Validation score: 0.469175
    Iteration 345, loss = 5022.75888568
    Validation score: 0.469359
    Iteration 346, loss = 5020.47860662
    Validation score: 0.469697
    Iteration 347, loss = 5018.36519115
    Validation score: 0.469867
    Iteration 348, loss = 5016.00854510
    Validation score: 0.470012
    Iteration 349, loss = 5013.64888634
    Validation score: 0.470219
    Iteration 350, loss = 5011.53936333
    Validation score: 0.470443
    Iteration 351, loss = 5009.30190885
    Validation score: 0.470582
    Iteration 352, loss = 5007.03015454
    Validation score: 0.470770
    Iteration 353, loss = 5004.79833706
    Validation score: 0.471041
    Iteration 354, loss = 5002.67548431
    Validation score: 0.471196
    Iteration 355, loss = 5000.51199291
    Validation score: 0.471400
    Iteration 356, loss = 4998.27568619
    Validation score: 0.471570
    Iteration 357, loss = 4996.16957332
    Validation score: 0.471791
    Iteration 358, loss = 4993.95067857
    Validation score: 0.471998
    Iteration 359, loss = 4991.81726045
    Validation score: 0.472210
    Iteration 360, loss = 4989.59815949
    Validation score: 0.472391
    Iteration 361, loss = 4987.39393171
    Validation score: 0.472679
    Iteration 362, loss = 4985.25980011
    Validation score: 0.472715
    Iteration 363, loss = 4983.08136296
    Validation score: 0.473000
    Iteration 364, loss = 4981.08205776
    Validation score: 0.473220
    Iteration 365, loss = 4978.82499523
    Validation score: 0.473400
    Iteration 366, loss = 4976.59705745
    Validation score: 0.473498
    Iteration 367, loss = 4974.72534577
    Validation score: 0.473640
    Iteration 368, loss = 4972.41934762
    Validation score: 0.473916
    Iteration 369, loss = 4970.45958782
    Validation score: 0.474197
    Iteration 370, loss = 4968.38880497
    Validation score: 0.474348
    Iteration 371, loss = 4966.25897594
    Validation score: 0.474555
    Iteration 372, loss = 4964.00160092
    Validation score: 0.474694
    Iteration 373, loss = 4961.88375875
    Validation score: 0.474877
    Iteration 374, loss = 4959.99303387
    Validation score: 0.475139
    Iteration 375, loss = 4957.88629658
    Validation score: 0.475449
    Iteration 376, loss = 4955.88106821
    Validation score: 0.475494
    Iteration 377, loss = 4953.88458875
    Validation score: 0.475716
    Iteration 378, loss = 4951.81040316
    Validation score: 0.475906
    Iteration 379, loss = 4949.68232820
    Validation score: 0.476092
    Iteration 380, loss = 4947.74653371
    Validation score: 0.476210
    Iteration 381, loss = 4945.70170473
    Validation score: 0.476369
    Iteration 382, loss = 4943.83572641
    Validation score: 0.476516
    Iteration 383, loss = 4941.90312843
    Validation score: 0.476596
    Iteration 384, loss = 4939.93232416
    Validation score: 0.476852
    Iteration 385, loss = 4937.98951535
    Validation score: 0.477004
    Iteration 386, loss = 4936.04193968
    Validation score: 0.477270
    Iteration 387, loss = 4934.12197259
    Validation score: 0.477164
    Iteration 388, loss = 4932.13418606
    Validation score: 0.477449
    Iteration 389, loss = 4930.33786991
    Validation score: 0.477629
    Iteration 390, loss = 4928.19640180
    Validation score: 0.477732
    Iteration 391, loss = 4926.35923046
    Validation score: 0.477892
    Iteration 392, loss = 4924.42654576
    Validation score: 0.477913
    Iteration 393, loss = 4922.50205430
    Validation score: 0.478187
    Iteration 394, loss = 4920.55029447
    Validation score: 0.478271
    Iteration 395, loss = 4918.71789453
    Validation score: 0.478476
    Iteration 396, loss = 4916.78550358
    Validation score: 0.478568
    Iteration 397, loss = 4914.91276662
    Validation score: 0.478866
    Iteration 398, loss = 4912.93346596
    Validation score: 0.478816
    Iteration 399, loss = 4911.18326967
    Validation score: 0.479000
    Iteration 400, loss = 4909.24831278
    Validation score: 0.479140
    

    C:\Users\cbarret\AppData\Local\Continuum\anaconda3\lib\site-packages\sklearn\neural_network\multilayer_perceptron.py:564: ConvergenceWarning: Stochastic Optimizer: Maximum iterations (400) reached and the optimization hasn't converged yet.
      % self.max_iter, ConvergenceWarning)
    




    GridSearchCV(cv=3, error_score='raise',
           estimator=Pipeline(memory=None,
         steps=[('imputer', Imputer(axis=0, copy=True, missing_values='NaN', strategy='mean', verbose=0)), ('standardscaler', StandardScaler(copy=True, with_mean=True, with_std=True)), ('mlpregressor', MLPRegressor(activation='relu', alpha=0.0001, batch_size='auto', beta_1=0.9,
           beta_2=0.999, early_sto...
           solver='adam', tol=0.0001, validation_fraction=0.1, verbose=True,
           warm_start=False))]),
           fit_params=None, iid=True, n_jobs=-1,
           param_grid={'mlpregressor__activation': ['logistic', 'tanh'], 'mlpregressor__solver': ['sgd', 'adam'], 'mlpregressor__early_stopping': [True, False], 'mlpregressor__hidden_layer_sizes': [(50,), (100,)], 'mlpregressor__learning_rate_init': [0.001, 0.0001]},
           pre_dispatch='2*n_jobs', refit=True, return_train_score='warn',
           scoring='neg_median_absolute_error', verbose=2)




```python
grid_search.best_score_
```




    -16.269918669250544




```python
grid_search.best_params_
```




    {'mlpregressor__activation': 'tanh',
     'mlpregressor__early_stopping': True,
     'mlpregressor__hidden_layer_sizes': (100,),
     'mlpregressor__learning_rate_init': 0.0001,
     'mlpregressor__solver': 'adam'}




```python
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Imputer
# Multi-layer Perceptron is sensitive to feature scaling, so it is highly recommended to scale your data
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_regression

# Create the pipeline: imputation + MLP regressor
my_pipeline_NN_grid = make_pipeline(Imputer(), StandardScaler(),
                                    MLPRegressor(hidden_layer_sizes = (100,),
                                                 activation = 'tanh',
                                                 early_stopping = True,
                                                 learning_rate_init = 0.0001,
                                                 solver = 'adam',
                                                 max_iter = 500,
                                                 random_state = 42,
                                                 verbose = True)) 

# Fit the model
my_pipeline_NN_grid.fit(train_X, train_y)
```

    Iteration 1, loss = 12365.33617118
    Validation score: -0.937196
    Iteration 2, loss = 12258.92656260
    Validation score: -0.914348
    Iteration 3, loss = 12137.62984836
    Validation score: -0.888118
    Iteration 4, loss = 12001.03568810
    Validation score: -0.858977
    Iteration 5, loss = 11854.14995127
    Validation score: -0.828284
    Iteration 6, loss = 11703.11602306
    Validation score: -0.797159
    Iteration 7, loss = 11551.81977378
    Validation score: -0.766160
    Iteration 8, loss = 11401.97749265
    Validation score: -0.735443
    Iteration 9, loss = 11253.92390169
    Validation score: -0.704980
    Iteration 10, loss = 11107.30596336
    Validation score: -0.674745
    Iteration 11, loss = 10962.24129882
    Validation score: -0.644778
    Iteration 12, loss = 10818.65996750
    Validation score: -0.615148
    Iteration 13, loss = 10677.17168712
    Validation score: -0.585960
    Iteration 14, loss = 10538.34917241
    Validation score: -0.557336
    Iteration 15, loss = 10402.56061390
    Validation score: -0.529516
    Iteration 16, loss = 10270.31448554
    Validation score: -0.502348
    Iteration 17, loss = 10141.97194954
    Validation score: -0.476119
    Iteration 18, loss = 10017.75691617
    Validation score: -0.450782
    Iteration 19, loss = 9897.75851920
    Validation score: -0.426382
    Iteration 20, loss = 9782.02949136
    Validation score: -0.402807
    Iteration 21, loss = 9670.63465322
    Validation score: -0.380221
    Iteration 22, loss = 9563.46361110
    Validation score: -0.358504
    Iteration 23, loss = 9460.47149935
    Validation score: -0.337718
    Iteration 24, loss = 9361.52511966
    Validation score: -0.317717
    Iteration 25, loss = 9266.48372661
    Validation score: -0.298647
    Iteration 26, loss = 9175.23319237
    Validation score: -0.280239
    Iteration 27, loss = 9087.84795319
    Validation score: -0.262653
    Iteration 28, loss = 9003.94042188
    Validation score: -0.245749
    Iteration 29, loss = 8923.30607547
    Validation score: -0.229617
    Iteration 30, loss = 8845.62061515
    Validation score: -0.213970
    Iteration 31, loss = 8770.74216987
    Validation score: -0.198962
    Iteration 32, loss = 8698.48134854
    Validation score: -0.184460
    Iteration 33, loss = 8628.70864004
    Validation score: -0.170450
    Iteration 34, loss = 8561.27880824
    Validation score: -0.156950
    Iteration 35, loss = 8496.07701534
    Validation score: -0.143939
    Iteration 36, loss = 8432.98432886
    Validation score: -0.131286
    Iteration 37, loss = 8371.99344900
    Validation score: -0.119095
    Iteration 38, loss = 8312.87189140
    Validation score: -0.107318
    Iteration 39, loss = 8255.71935524
    Validation score: -0.095926
    Iteration 40, loss = 8200.30983825
    Validation score: -0.084935
    Iteration 41, loss = 8146.56455772
    Validation score: -0.074204
    Iteration 42, loss = 8094.40518687
    Validation score: -0.063865
    Iteration 43, loss = 8043.74345145
    Validation score: -0.053843
    Iteration 44, loss = 7994.55783715
    Validation score: -0.044091
    Iteration 45, loss = 7946.82822585
    Validation score: -0.034709
    Iteration 46, loss = 7900.37118877
    Validation score: -0.025502
    Iteration 47, loss = 7855.21995002
    Validation score: -0.016583
    Iteration 48, loss = 7811.19902415
    Validation score: -0.007979
    Iteration 49, loss = 7768.50047531
    Validation score: 0.000501
    Iteration 50, loss = 7726.90540784
    Validation score: 0.008686
    Iteration 51, loss = 7686.45936824
    Validation score: 0.016592
    Iteration 52, loss = 7647.16981802
    Validation score: 0.024291
    Iteration 53, loss = 7608.90183796
    Validation score: 0.031838
    Iteration 54, loss = 7571.68075989
    Validation score: 0.039109
    Iteration 55, loss = 7535.46143319
    Validation score: 0.046171
    Iteration 56, loss = 7500.19146583
    Validation score: 0.053008
    Iteration 57, loss = 7465.90793467
    Validation score: 0.059803
    Iteration 58, loss = 7432.60915323
    Validation score: 0.066296
    Iteration 59, loss = 7400.20673128
    Validation score: 0.072630
    Iteration 60, loss = 7368.64044899
    Validation score: 0.078704
    Iteration 61, loss = 7338.00453765
    Validation score: 0.084702
    Iteration 62, loss = 7308.30755470
    Validation score: 0.090472
    Iteration 63, loss = 7279.48390714
    Validation score: 0.096069
    Iteration 64, loss = 7251.42775251
    Validation score: 0.101476
    Iteration 65, loss = 7223.99891306
    Validation score: 0.106808
    Iteration 66, loss = 7197.56752633
    Validation score: 0.111891
    Iteration 67, loss = 7171.99590326
    Validation score: 0.116855
    Iteration 68, loss = 7147.01078849
    Validation score: 0.121664
    Iteration 69, loss = 7122.61480382
    Validation score: 0.126344
    Iteration 70, loss = 7098.69556832
    Validation score: 0.130974
    Iteration 71, loss = 7075.23749534
    Validation score: 0.135494
    Iteration 72, loss = 7052.17323353
    Validation score: 0.139925
    Iteration 73, loss = 7029.48515578
    Validation score: 0.144238
    Iteration 74, loss = 7007.03065370
    Validation score: 0.148547
    Iteration 75, loss = 6984.68653942
    Validation score: 0.152779
    Iteration 76, loss = 6962.54261146
    Validation score: 0.157061
    Iteration 77, loss = 6940.51448976
    Validation score: 0.161335
    Iteration 78, loss = 6918.50416273
    Validation score: 0.165512
    Iteration 79, loss = 6896.48330770
    Validation score: 0.169686
    Iteration 80, loss = 6874.47955389
    Validation score: 0.173841
    Iteration 81, loss = 6852.50836475
    Validation score: 0.177957
    Iteration 82, loss = 6830.66874197
    Validation score: 0.182131
    Iteration 83, loss = 6808.92546320
    Validation score: 0.186237
    Iteration 84, loss = 6787.50586283
    Validation score: 0.190287
    Iteration 85, loss = 6766.19510079
    Validation score: 0.194226
    Iteration 86, loss = 6745.11271223
    Validation score: 0.198153
    Iteration 87, loss = 6724.26838805
    Validation score: 0.202033
    Iteration 88, loss = 6703.53346060
    Validation score: 0.205923
    Iteration 89, loss = 6683.28688617
    Validation score: 0.209701
    Iteration 90, loss = 6663.24659260
    Validation score: 0.213424
    Iteration 91, loss = 6643.52667223
    Validation score: 0.217125
    Iteration 92, loss = 6624.13401851
    Validation score: 0.220700
    Iteration 93, loss = 6605.12056243
    Validation score: 0.224234
    Iteration 94, loss = 6586.30349488
    Validation score: 0.227753
    Iteration 95, loss = 6567.81754283
    Validation score: 0.231105
    Iteration 96, loss = 6549.79782186
    Validation score: 0.234498
    Iteration 97, loss = 6531.97619104
    Validation score: 0.237681
    Iteration 98, loss = 6514.52690310
    Validation score: 0.240918
    Iteration 99, loss = 6497.21713448
    Validation score: 0.244114
    Iteration 100, loss = 6480.16091323
    Validation score: 0.247279
    Iteration 101, loss = 6463.31703815
    Validation score: 0.250379
    Iteration 102, loss = 6446.75245099
    Validation score: 0.253424
    Iteration 103, loss = 6430.50103487
    Validation score: 0.256450
    Iteration 104, loss = 6414.45533294
    Validation score: 0.259386
    Iteration 105, loss = 6398.65353067
    Validation score: 0.262305
    Iteration 106, loss = 6382.93668961
    Validation score: 0.265152
    Iteration 107, loss = 6367.63190244
    Validation score: 0.268018
    Iteration 108, loss = 6352.59504195
    Validation score: 0.270741
    Iteration 109, loss = 6337.80766837
    Validation score: 0.273472
    Iteration 110, loss = 6323.18471384
    Validation score: 0.276090
    Iteration 111, loss = 6308.68624683
    Validation score: 0.278722
    Iteration 112, loss = 6294.33035312
    Validation score: 0.281266
    Iteration 113, loss = 6280.19294767
    Validation score: 0.283802
    Iteration 114, loss = 6266.26247251
    Validation score: 0.286307
    Iteration 115, loss = 6252.54878510
    Validation score: 0.288741
    Iteration 116, loss = 6238.98620104
    Validation score: 0.291196
    Iteration 117, loss = 6225.71184972
    Validation score: 0.293481
    Iteration 118, loss = 6212.61608908
    Validation score: 0.295847
    Iteration 119, loss = 6199.58156017
    Validation score: 0.298191
    Iteration 120, loss = 6186.79504091
    Validation score: 0.300466
    Iteration 121, loss = 6174.26129970
    Validation score: 0.302627
    Iteration 122, loss = 6161.85633101
    Validation score: 0.304781
    Iteration 123, loss = 6149.56691081
    Validation score: 0.306989
    Iteration 124, loss = 6137.47492781
    Validation score: 0.309111
    Iteration 125, loss = 6125.60834220
    Validation score: 0.311165
    Iteration 126, loss = 6113.71411155
    Validation score: 0.313228
    Iteration 127, loss = 6102.10316629
    Validation score: 0.315304
    Iteration 128, loss = 6090.52957042
    Validation score: 0.317323
    Iteration 129, loss = 6079.00854136
    Validation score: 0.319315
    Iteration 130, loss = 6067.79037422
    Validation score: 0.321248
    Iteration 131, loss = 6056.60260272
    Validation score: 0.323200
    Iteration 132, loss = 6045.52919107
    Validation score: 0.325126
    Iteration 133, loss = 6034.60401224
    Validation score: 0.327034
    Iteration 134, loss = 6023.83650280
    Validation score: 0.328851
    Iteration 135, loss = 6013.17072157
    Validation score: 0.330672
    Iteration 136, loss = 6002.61053938
    Validation score: 0.332531
    Iteration 137, loss = 5992.28678729
    Validation score: 0.334300
    Iteration 138, loss = 5981.96491218
    Validation score: 0.336040
    Iteration 139, loss = 5971.81014402
    Validation score: 0.337793
    Iteration 140, loss = 5961.85750763
    Validation score: 0.339486
    Iteration 141, loss = 5951.96602553
    Validation score: 0.341134
    Iteration 142, loss = 5942.16557181
    Validation score: 0.342839
    Iteration 143, loss = 5932.53631927
    Validation score: 0.344392
    Iteration 144, loss = 5923.00724991
    Validation score: 0.346040
    Iteration 145, loss = 5913.54966999
    Validation score: 0.347677
    Iteration 146, loss = 5904.17020396
    Validation score: 0.349274
    Iteration 147, loss = 5894.70347098
    Validation score: 0.350683
    Iteration 148, loss = 5885.59242341
    Validation score: 0.352269
    Iteration 149, loss = 5876.49284059
    Validation score: 0.353764
    Iteration 150, loss = 5867.56108113
    Validation score: 0.355263
    Iteration 151, loss = 5858.62778683
    Validation score: 0.356758
    Iteration 152, loss = 5849.82209194
    Validation score: 0.358215
    Iteration 153, loss = 5841.10402451
    Validation score: 0.359670
    Iteration 154, loss = 5832.42139752
    Validation score: 0.361174
    Iteration 155, loss = 5823.75467059
    Validation score: 0.362570
    Iteration 156, loss = 5815.17635526
    Validation score: 0.363980
    Iteration 157, loss = 5806.60385880
    Validation score: 0.365415
    Iteration 158, loss = 5798.13552983
    Validation score: 0.366786
    Iteration 159, loss = 5789.89083318
    Validation score: 0.368178
    Iteration 160, loss = 5781.67885824
    Validation score: 0.369497
    Iteration 161, loss = 5773.61196270
    Validation score: 0.370802
    Iteration 162, loss = 5765.39778373
    Validation score: 0.372094
    Iteration 163, loss = 5757.33863016
    Validation score: 0.373425
    Iteration 164, loss = 5749.25658832
    Validation score: 0.374611
    Iteration 165, loss = 5741.37073635
    Validation score: 0.375929
    Iteration 166, loss = 5733.57941156
    Validation score: 0.377075
    Iteration 167, loss = 5725.81148402
    Validation score: 0.378287
    Iteration 168, loss = 5718.20641974
    Validation score: 0.379544
    Iteration 169, loss = 5710.64365188
    Validation score: 0.380696
    Iteration 170, loss = 5703.21731023
    Validation score: 0.381900
    Iteration 171, loss = 5695.79184861
    Validation score: 0.383026
    Iteration 172, loss = 5688.59160629
    Validation score: 0.384118
    Iteration 173, loss = 5681.39381390
    Validation score: 0.385254
    Iteration 174, loss = 5674.18813247
    Validation score: 0.386347
    Iteration 175, loss = 5667.16609371
    Validation score: 0.387436
    Iteration 176, loss = 5660.22146210
    Validation score: 0.388522
    Iteration 177, loss = 5653.36401399
    Validation score: 0.389555
    Iteration 178, loss = 5646.73318244
    Validation score: 0.390574
    Iteration 179, loss = 5639.87713611
    Validation score: 0.391666
    Iteration 180, loss = 5633.23959550
    Validation score: 0.392671
    Iteration 181, loss = 5626.55561484
    Validation score: 0.393736
    Iteration 182, loss = 5619.94063729
    Validation score: 0.394554
    Iteration 183, loss = 5613.71659900
    Validation score: 0.395496
    Iteration 184, loss = 5606.98879103
    Validation score: 0.396485
    Iteration 185, loss = 5600.80951445
    Validation score: 0.397361
    Iteration 186, loss = 5594.45398151
    Validation score: 0.398369
    Iteration 187, loss = 5588.24044205
    Validation score: 0.399292
    Iteration 188, loss = 5582.08579409
    Validation score: 0.400185
    Iteration 189, loss = 5576.01648851
    Validation score: 0.401009
    Iteration 190, loss = 5569.86114214
    Validation score: 0.401905
    Iteration 191, loss = 5564.06229346
    Validation score: 0.402825
    Iteration 192, loss = 5557.98275165
    Validation score: 0.403640
    Iteration 193, loss = 5552.23355415
    Validation score: 0.404403
    Iteration 194, loss = 5546.49807091
    Validation score: 0.405196
    Iteration 195, loss = 5540.77041597
    Validation score: 0.406002
    Iteration 196, loss = 5535.14189539
    Validation score: 0.406846
    Iteration 197, loss = 5529.50040725
    Validation score: 0.407595
    Iteration 198, loss = 5523.96765099
    Validation score: 0.408359
    Iteration 199, loss = 5518.50362379
    Validation score: 0.409110
    Iteration 200, loss = 5513.01215904
    Validation score: 0.409887
    Iteration 201, loss = 5507.72310493
    Validation score: 0.410538
    Iteration 202, loss = 5502.42512804
    Validation score: 0.411293
    Iteration 203, loss = 5497.12394289
    Validation score: 0.411948
    Iteration 204, loss = 5491.98156348
    Validation score: 0.412670
    Iteration 205, loss = 5486.73260455
    Validation score: 0.413413
    Iteration 206, loss = 5481.59140313
    Validation score: 0.414084
    Iteration 207, loss = 5476.43894566
    Validation score: 0.414815
    Iteration 208, loss = 5471.47649446
    Validation score: 0.415512
    Iteration 209, loss = 5466.45886821
    Validation score: 0.416254
    Iteration 210, loss = 5461.36830099
    Validation score: 0.416876
    Iteration 211, loss = 5456.52239208
    Validation score: 0.417591
    Iteration 212, loss = 5451.55135689
    Validation score: 0.418285
    Iteration 213, loss = 5446.74304519
    Validation score: 0.418977
    Iteration 214, loss = 5441.86689015
    Validation score: 0.419588
    Iteration 215, loss = 5437.18008011
    Validation score: 0.420275
    Iteration 216, loss = 5432.28647872
    Validation score: 0.420917
    Iteration 217, loss = 5427.57965886
    Validation score: 0.421553
    Iteration 218, loss = 5422.97879958
    Validation score: 0.422189
    Iteration 219, loss = 5418.28690320
    Validation score: 0.422784
    Iteration 220, loss = 5413.59247877
    Validation score: 0.423404
    Iteration 221, loss = 5409.12638511
    Validation score: 0.424030
    Iteration 222, loss = 5404.49207221
    Validation score: 0.424654
    Iteration 223, loss = 5400.04177846
    Validation score: 0.425195
    Iteration 224, loss = 5395.61795243
    Validation score: 0.425779
    Iteration 225, loss = 5391.26397131
    Validation score: 0.426403
    Iteration 226, loss = 5386.89169128
    Validation score: 0.427000
    Iteration 227, loss = 5382.43809283
    Validation score: 0.427506
    Iteration 228, loss = 5378.24317138
    Validation score: 0.428137
    Iteration 229, loss = 5373.95258815
    Validation score: 0.428674
    Iteration 230, loss = 5369.83186885
    Validation score: 0.429240
    Iteration 231, loss = 5365.60406736
    Validation score: 0.429749
    Iteration 232, loss = 5361.41335831
    Validation score: 0.430307
    Iteration 233, loss = 5357.25381633
    Validation score: 0.430904
    Iteration 234, loss = 5353.03526612
    Validation score: 0.431446
    Iteration 235, loss = 5348.88245068
    Validation score: 0.431930
    Iteration 236, loss = 5344.93524625
    Validation score: 0.432511
    Iteration 237, loss = 5340.83437266
    Validation score: 0.433017
    Iteration 238, loss = 5336.91522930
    Validation score: 0.433629
    Iteration 239, loss = 5332.95591936
    Validation score: 0.434030
    Iteration 240, loss = 5329.01896623
    Validation score: 0.434604
    Iteration 241, loss = 5324.88487479
    Validation score: 0.434981
    Iteration 242, loss = 5321.18689372
    Validation score: 0.435453
    Iteration 243, loss = 5317.24994130
    Validation score: 0.435929
    Iteration 244, loss = 5313.30734559
    Validation score: 0.436372
    Iteration 245, loss = 5309.50316366
    Validation score: 0.437013
    Iteration 246, loss = 5305.67036024
    Validation score: 0.437391
    Iteration 247, loss = 5301.90736946
    Validation score: 0.437870
    Iteration 248, loss = 5298.11764775
    Validation score: 0.438279
    Iteration 249, loss = 5294.45233936
    Validation score: 0.438782
    Iteration 250, loss = 5290.89879623
    Validation score: 0.439216
    Iteration 251, loss = 5287.09416297
    Validation score: 0.439723
    Iteration 252, loss = 5283.38525207
    Validation score: 0.440172
    Iteration 253, loss = 5279.83981296
    Validation score: 0.440573
    Iteration 254, loss = 5276.33839881
    Validation score: 0.441187
    Iteration 255, loss = 5272.70087877
    Validation score: 0.441478
    Iteration 256, loss = 5269.17801673
    Validation score: 0.441978
    Iteration 257, loss = 5265.79227851
    Validation score: 0.442260
    Iteration 258, loss = 5262.23358686
    Validation score: 0.442788
    Iteration 259, loss = 5258.89909519
    Validation score: 0.443227
    Iteration 260, loss = 5255.42895097
    Validation score: 0.443585
    Iteration 261, loss = 5252.27806829
    Validation score: 0.444017
    Iteration 262, loss = 5248.85174036
    Validation score: 0.444442
    Iteration 263, loss = 5245.65708715
    Validation score: 0.444768
    Iteration 264, loss = 5242.11409062
    Validation score: 0.445261
    Iteration 265, loss = 5238.87249658
    Validation score: 0.445678
    Iteration 266, loss = 5235.59832550
    Validation score: 0.446061
    Iteration 267, loss = 5232.38630355
    Validation score: 0.446501
    Iteration 268, loss = 5229.13583023
    Validation score: 0.447005
    Iteration 269, loss = 5225.87369031
    Validation score: 0.447261
    Iteration 270, loss = 5222.70638268
    Validation score: 0.447687
    Iteration 271, loss = 5219.49001126
    Validation score: 0.448106
    Iteration 272, loss = 5216.18203035
    Validation score: 0.448493
    Iteration 273, loss = 5213.03364798
    Validation score: 0.448847
    Iteration 274, loss = 5209.90951175
    Validation score: 0.449419
    Iteration 275, loss = 5206.70190560
    Validation score: 0.449484
    Iteration 276, loss = 5203.71591283
    Validation score: 0.449901
    Iteration 277, loss = 5200.53676790
    Validation score: 0.450362
    Iteration 278, loss = 5197.56321373
    Validation score: 0.450703
    Iteration 279, loss = 5194.42296936
    Validation score: 0.451062
    Iteration 280, loss = 5191.33540196
    Validation score: 0.451362
    Iteration 281, loss = 5188.33338518
    Validation score: 0.451745
    Iteration 282, loss = 5185.44982017
    Validation score: 0.452159
    Iteration 283, loss = 5182.35191876
    Validation score: 0.452568
    Iteration 284, loss = 5179.47808095
    Validation score: 0.452874
    Iteration 285, loss = 5176.53203324
    Validation score: 0.453238
    Iteration 286, loss = 5173.47161839
    Validation score: 0.453579
    Iteration 287, loss = 5170.61997462
    Validation score: 0.453826
    Iteration 288, loss = 5167.67281453
    Validation score: 0.454247
    Iteration 289, loss = 5164.87658108
    Validation score: 0.454660
    Iteration 290, loss = 5162.02441356
    Validation score: 0.454890
    Iteration 291, loss = 5159.08528624
    Validation score: 0.455370
    Iteration 292, loss = 5156.31272337
    Validation score: 0.455572
    Iteration 293, loss = 5153.20212458
    Validation score: 0.455873
    Iteration 294, loss = 5150.45700491
    Validation score: 0.456237
    Iteration 295, loss = 5147.56883546
    Validation score: 0.456596
    Iteration 296, loss = 5144.73541611
    Validation score: 0.456891
    Iteration 297, loss = 5142.10637014
    Validation score: 0.457223
    Iteration 298, loss = 5139.24782409
    Validation score: 0.457573
    Iteration 299, loss = 5136.60060050
    Validation score: 0.457786
    Iteration 300, loss = 5133.83401079
    Validation score: 0.458275
    Iteration 301, loss = 5131.03793592
    Validation score: 0.458475
    Iteration 302, loss = 5128.50451791
    Validation score: 0.458777
    Iteration 303, loss = 5125.66779116
    Validation score: 0.459047
    Iteration 304, loss = 5123.07696274
    Validation score: 0.459364
    Iteration 305, loss = 5120.44158280
    Validation score: 0.459657
    Iteration 306, loss = 5117.92541588
    Validation score: 0.459730
    Iteration 307, loss = 5115.34407097
    Validation score: 0.460247
    Iteration 308, loss = 5112.55979974
    Validation score: 0.460344
    Iteration 309, loss = 5110.06751305
    Validation score: 0.460601
    Iteration 310, loss = 5107.34366911
    Validation score: 0.460974
    Iteration 311, loss = 5104.81413587
    Validation score: 0.461166
    Iteration 312, loss = 5102.22875702
    Validation score: 0.461590
    Iteration 313, loss = 5099.50027094
    Validation score: 0.461616
    Iteration 314, loss = 5097.04865508
    Validation score: 0.461889
    Iteration 315, loss = 5094.52895422
    Validation score: 0.462250
    Iteration 316, loss = 5091.97046031
    Validation score: 0.462431
    Iteration 317, loss = 5089.47812246
    Validation score: 0.462700
    Iteration 318, loss = 5087.00132700
    Validation score: 0.462919
    Iteration 319, loss = 5084.56060778
    Validation score: 0.463299
    Iteration 320, loss = 5082.12897922
    Validation score: 0.463436
    Iteration 321, loss = 5079.56508303
    Validation score: 0.463767
    Iteration 322, loss = 5077.08255174
    Validation score: 0.463966
    Iteration 323, loss = 5074.74143466
    Validation score: 0.464319
    Iteration 324, loss = 5072.13796917
    Validation score: 0.464544
    Iteration 325, loss = 5069.79212285
    Validation score: 0.464766
    Iteration 326, loss = 5067.35953934
    Validation score: 0.465023
    Iteration 327, loss = 5064.83868462
    Validation score: 0.465272
    Iteration 328, loss = 5062.48446438
    Validation score: 0.465510
    Iteration 329, loss = 5059.96154676
    Validation score: 0.465727
    Iteration 330, loss = 5057.54857992
    Validation score: 0.465946
    Iteration 331, loss = 5055.24798640
    Validation score: 0.466300
    Iteration 332, loss = 5052.87713268
    Validation score: 0.466453
    Iteration 333, loss = 5050.52523751
    Validation score: 0.466648
    Iteration 334, loss = 5048.15924131
    Validation score: 0.466958
    Iteration 335, loss = 5045.89526188
    Validation score: 0.467104
    Iteration 336, loss = 5043.56131634
    Validation score: 0.467330
    Iteration 337, loss = 5041.15855124
    Validation score: 0.467708
    Iteration 338, loss = 5038.95740458
    Validation score: 0.467874
    Iteration 339, loss = 5036.56765781
    Validation score: 0.468089
    Iteration 340, loss = 5034.12970440
    Validation score: 0.468267
    Iteration 341, loss = 5031.71902307
    Validation score: 0.468592
    Iteration 342, loss = 5029.52776858
    Validation score: 0.468757
    Iteration 343, loss = 5027.29538840
    Validation score: 0.469000
    Iteration 344, loss = 5024.90527375
    Validation score: 0.469175
    Iteration 345, loss = 5022.75888568
    Validation score: 0.469359
    Iteration 346, loss = 5020.47860662
    Validation score: 0.469697
    Iteration 347, loss = 5018.36519115
    Validation score: 0.469867
    Iteration 348, loss = 5016.00854510
    Validation score: 0.470012
    Iteration 349, loss = 5013.64888634
    Validation score: 0.470219
    Iteration 350, loss = 5011.53936333
    Validation score: 0.470443
    Iteration 351, loss = 5009.30190885
    Validation score: 0.470582
    Iteration 352, loss = 5007.03015454
    Validation score: 0.470770
    Iteration 353, loss = 5004.79833706
    Validation score: 0.471041
    Iteration 354, loss = 5002.67548431
    Validation score: 0.471196
    Iteration 355, loss = 5000.51199291
    Validation score: 0.471400
    Iteration 356, loss = 4998.27568619
    Validation score: 0.471570
    Iteration 357, loss = 4996.16957332
    Validation score: 0.471791
    Iteration 358, loss = 4993.95067857
    Validation score: 0.471998
    Iteration 359, loss = 4991.81726045
    Validation score: 0.472210
    Iteration 360, loss = 4989.59815949
    Validation score: 0.472391
    Iteration 361, loss = 4987.39393171
    Validation score: 0.472679
    Iteration 362, loss = 4985.25980011
    Validation score: 0.472715
    Iteration 363, loss = 4983.08136296
    Validation score: 0.473000
    Iteration 364, loss = 4981.08205776
    Validation score: 0.473220
    Iteration 365, loss = 4978.82499523
    Validation score: 0.473400
    Iteration 366, loss = 4976.59705745
    Validation score: 0.473498
    Iteration 367, loss = 4974.72534577
    Validation score: 0.473640
    Iteration 368, loss = 4972.41934762
    Validation score: 0.473916
    Iteration 369, loss = 4970.45958782
    Validation score: 0.474197
    Iteration 370, loss = 4968.38880497
    Validation score: 0.474348
    Iteration 371, loss = 4966.25897594
    Validation score: 0.474555
    Iteration 372, loss = 4964.00160092
    Validation score: 0.474694
    Iteration 373, loss = 4961.88375875
    Validation score: 0.474877
    Iteration 374, loss = 4959.99303387
    Validation score: 0.475139
    Iteration 375, loss = 4957.88629658
    Validation score: 0.475449
    Iteration 376, loss = 4955.88106821
    Validation score: 0.475494
    Iteration 377, loss = 4953.88458875
    Validation score: 0.475716
    Iteration 378, loss = 4951.81040316
    Validation score: 0.475906
    Iteration 379, loss = 4949.68232820
    Validation score: 0.476092
    Iteration 380, loss = 4947.74653371
    Validation score: 0.476210
    Iteration 381, loss = 4945.70170473
    Validation score: 0.476369
    Iteration 382, loss = 4943.83572641
    Validation score: 0.476516
    Iteration 383, loss = 4941.90312843
    Validation score: 0.476596
    Iteration 384, loss = 4939.93232416
    Validation score: 0.476852
    Iteration 385, loss = 4937.98951535
    Validation score: 0.477004
    Iteration 386, loss = 4936.04193968
    Validation score: 0.477270
    Iteration 387, loss = 4934.12197259
    Validation score: 0.477164
    Iteration 388, loss = 4932.13418606
    Validation score: 0.477449
    Iteration 389, loss = 4930.33786991
    Validation score: 0.477629
    Iteration 390, loss = 4928.19640180
    Validation score: 0.477732
    Iteration 391, loss = 4926.35923046
    Validation score: 0.477892
    Iteration 392, loss = 4924.42654576
    Validation score: 0.477913
    Iteration 393, loss = 4922.50205430
    Validation score: 0.478187
    Iteration 394, loss = 4920.55029447
    Validation score: 0.478271
    Iteration 395, loss = 4918.71789453
    Validation score: 0.478476
    Iteration 396, loss = 4916.78550358
    Validation score: 0.478568
    Iteration 397, loss = 4914.91276662
    Validation score: 0.478866
    Iteration 398, loss = 4912.93346596
    Validation score: 0.478816
    Iteration 399, loss = 4911.18326967
    Validation score: 0.479000
    Iteration 400, loss = 4909.24831278
    Validation score: 0.479140
    Iteration 401, loss = 4907.64730570
    Validation score: 0.479231
    Iteration 402, loss = 4905.67189247
    Validation score: 0.479430
    Iteration 403, loss = 4903.87083680
    Validation score: 0.479406
    Iteration 404, loss = 4902.31419482
    Validation score: 0.479525
    Iteration 405, loss = 4900.16913995
    Validation score: 0.479669
    Iteration 406, loss = 4898.45317229
    Validation score: 0.479762
    Iteration 407, loss = 4896.66975783
    Validation score: 0.479963
    Iteration 408, loss = 4894.73863571
    Validation score: 0.479917
    Iteration 409, loss = 4892.92589130
    Validation score: 0.480031
    Iteration 410, loss = 4891.17686909
    Validation score: 0.480153
    Iteration 411, loss = 4889.56945079
    Validation score: 0.480220
    Iteration 412, loss = 4887.65777701
    Validation score: 0.480315
    Iteration 413, loss = 4885.93216281
    Validation score: 0.480517
    Iteration 414, loss = 4884.35926389
    Validation score: 0.480411
    Iteration 415, loss = 4882.52453098
    Validation score: 0.480574
    Iteration 416, loss = 4880.62736300
    Validation score: 0.480813
    Iteration 417, loss = 4878.80692867
    Validation score: 0.480817
    Iteration 418, loss = 4877.14081533
    Validation score: 0.480983
    Iteration 419, loss = 4875.54288505
    Validation score: 0.481025
    Iteration 420, loss = 4873.73720773
    Validation score: 0.481132
    Iteration 421, loss = 4872.13826334
    Validation score: 0.481224
    Iteration 422, loss = 4870.46914871
    Validation score: 0.481271
    Iteration 423, loss = 4868.61613425
    Validation score: 0.481391
    Iteration 424, loss = 4866.82826676
    Validation score: 0.481482
    Iteration 425, loss = 4865.34964517
    Validation score: 0.481486
    Iteration 426, loss = 4863.62062761
    Validation score: 0.481760
    Iteration 427, loss = 4861.85919980
    Validation score: 0.481680
    Iteration 428, loss = 4860.14284675
    Validation score: 0.481683
    Iteration 429, loss = 4858.43441359
    Validation score: 0.481858
    Validation score did not improve more than tol=0.000100 for two consecutive epochs. Stopping.
    




    Pipeline(memory=None,
         steps=[('imputer', Imputer(axis=0, copy=True, missing_values='NaN', strategy='mean', verbose=0)), ('standardscaler', StandardScaler(copy=True, with_mean=True, with_std=True)), ('mlpregressor', MLPRegressor(activation='tanh', alpha=0.0001, batch_size='auto', beta_1=0.9,
           beta_2=0.999, early_sto...
           solver='adam', tol=0.0001, validation_fraction=0.1, verbose=True,
           warm_start=False))])




```python
# Accuracy on test data
predictions_grid = my_pipeline_NN_grid.predict(test_X)
%store predictions_grid

from sklearn.metrics import mean_absolute_error
# Write the MAE
print("Mean Absolute Error : " + str(mean_absolute_error(predictions_grid, test_y)))

from sklearn.metrics import median_absolute_error
print("Median Absolute Error test: " + str(round(median_absolute_error(predictions_grid, test_y), 2))) 

from sklearn.metrics import mean_squared_error
from math import sqrt
RMSE = round(sqrt(mean_squared_error(predictions_grid, test_y)), 2)
# Write the RMSE
print("RMSE test: " + str(RMSE)) 
```

    Stored 'predictions_grid' (ndarray)
    Mean Absolute Error : 28.002134513940412
    Median Absolute Error test: 16.01
    RMSE test: 72.02
    

The tuned Neural Network is much better than the one with default hyperparameters. However it is still much less precise than the first two models.

In conclusion, here are the results of the different models I applied:


```python
# Graph x and y axis values
import numpy as np
labels = np.array(['RF','Tuned RF','XGB', 'Tuned XGB', 'MLP', 'Tuned MLP'])
error_val = np.array([14.0, 13.35, 16.15, 13.53, 19.49, 16.01])

# Arrange bars
pos = np.arange(error_val.shape[0])
srt = np.argsort(error_val)

# Plots Mean Absolute Variance bars across functions
import matplotlib.pyplot as plt
plt.figure(figsize=(10,10))
plt.bar(pos, error_val[srt], align = 'center', color='#E35A5C')
plt.xticks(pos, labels[srt])
plt.xlabel('Model')
plt.ylabel('Median Absolute Error in $')
plt.title('Median Absolute Error Model Comparison')
plt.ylim(0,20)
plt.show()
```


![png](plots/output_22_0.png)
