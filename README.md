# Predicting prices for Airbnb rentals

<p style="text-align: justify">Platforms to connect landlords and tentants for temporary housing have become a cheap and good alternative to traditional hotel bookings in the tourism industry.</p>
<p style="text-align: justify">Airbnb's business model relies mainly on a fee charged from the total rental price, so it's in their best interest to optimize the prices in order to maximize their revenue. From the point of view of guests, it is benefitial to them to have fair prices that reflect the market and the value of properties. Likewise, hosts don't want to undercharge for the property nor overcharge, because it can reduce the demand.</p>
<p style="text-align: justify">Thus, the question I'm trying to answer is: <b>Can we predict rental prices from features related to the property ?</b></p>

## Loading the data


```python
import pandas as pd
from sklearn.model_selection import train_test_split
import missingno as msno
from datetime import datetime as dt
```


```python
df = pd.read_csv("./data/train.csv")
```


```python
df.shape
```




    (74111, 29)




```python
df.columns
```




    Index(['id', 'log_price', 'property_type', 'room_type', 'amenities',
           'accommodates', 'bathrooms', 'bed_type', 'cancellation_policy',
           'cleaning_fee', 'city', 'description', 'first_review',
           'host_has_profile_pic', 'host_identity_verified', 'host_response_rate',
           'host_since', 'instant_bookable', 'last_review', 'latitude',
           'longitude', 'name', 'neighbourhood', 'number_of_reviews',
           'review_scores_rating', 'thumbnail_url', 'zipcode', 'bedrooms', 'beds'],
          dtype='object')



Here, we have a dataset of 74,111 rows and 28 features related to the property, such as location, reviews and rooms. The target variable is the log of the price.

The first thing to do is to separate the target variable from the features. The target column is *log_price*, while all the other columns will be part of the feature dataset, except for the id, which I'll drop now as it is not relevant to the problem (it is a unique number for each row).


```python
X_raw = df.iloc[:,2:]
y = df["log_price"]
```

## Cleaning the data

Before we start fitting models into the data, it is useful to explore the dataset to detect anomalies, handle missing values and preprocess some features.

Let's now see a sample of the first 5 rows of the feature set and a brief description of the main statistics:


```python
# pd.set_option("display.max_columns", None)
X_raw.head()
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
      <th>property_type</th>
      <th>room_type</th>
      <th>amenities</th>
      <th>accommodates</th>
      <th>bathrooms</th>
      <th>bed_type</th>
      <th>cancellation_policy</th>
      <th>cleaning_fee</th>
      <th>city</th>
      <th>description</th>
      <th>first_review</th>
      <th>host_has_profile_pic</th>
      <th>host_identity_verified</th>
      <th>host_response_rate</th>
      <th>host_since</th>
      <th>instant_bookable</th>
      <th>last_review</th>
      <th>latitude</th>
      <th>longitude</th>
      <th>name</th>
      <th>neighbourhood</th>
      <th>number_of_reviews</th>
      <th>review_scores_rating</th>
      <th>thumbnail_url</th>
      <th>zipcode</th>
      <th>bedrooms</th>
      <th>beds</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Apartment</td>
      <td>Entire home/apt</td>
      <td>{"Wireless Internet","Air conditioning",Kitche...</td>
      <td>3</td>
      <td>1.0</td>
      <td>Real Bed</td>
      <td>strict</td>
      <td>True</td>
      <td>NYC</td>
      <td>Beautiful, sunlit brownstone 1-bedroom in the ...</td>
      <td>2016-06-18</td>
      <td>t</td>
      <td>t</td>
      <td>NaN</td>
      <td>2012-03-26</td>
      <td>f</td>
      <td>2016-07-18</td>
      <td>40.696524</td>
      <td>-73.991617</td>
      <td>Beautiful brownstone 1-bedroom</td>
      <td>Brooklyn Heights</td>
      <td>2</td>
      <td>100.0</td>
      <td>https://a0.muscache.com/im/pictures/6d7cbbf7-c...</td>
      <td>11201</td>
      <td>1.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Apartment</td>
      <td>Entire home/apt</td>
      <td>{"Wireless Internet","Air conditioning",Kitche...</td>
      <td>7</td>
      <td>1.0</td>
      <td>Real Bed</td>
      <td>strict</td>
      <td>True</td>
      <td>NYC</td>
      <td>Enjoy travelling during your stay in Manhattan...</td>
      <td>2017-08-05</td>
      <td>t</td>
      <td>f</td>
      <td>100%</td>
      <td>2017-06-19</td>
      <td>t</td>
      <td>2017-09-23</td>
      <td>40.766115</td>
      <td>-73.989040</td>
      <td>Superb 3BR Apt Located Near Times Square</td>
      <td>Hell's Kitchen</td>
      <td>6</td>
      <td>93.0</td>
      <td>https://a0.muscache.com/im/pictures/348a55fe-4...</td>
      <td>10019</td>
      <td>3.0</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Apartment</td>
      <td>Entire home/apt</td>
      <td>{TV,"Cable TV","Wireless Internet","Air condit...</td>
      <td>5</td>
      <td>1.0</td>
      <td>Real Bed</td>
      <td>moderate</td>
      <td>True</td>
      <td>NYC</td>
      <td>The Oasis comes complete with a full backyard ...</td>
      <td>2017-04-30</td>
      <td>t</td>
      <td>t</td>
      <td>100%</td>
      <td>2016-10-25</td>
      <td>t</td>
      <td>2017-09-14</td>
      <td>40.808110</td>
      <td>-73.943756</td>
      <td>The Garden Oasis</td>
      <td>Harlem</td>
      <td>10</td>
      <td>92.0</td>
      <td>https://a0.muscache.com/im/pictures/6fae5362-9...</td>
      <td>10027</td>
      <td>1.0</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>House</td>
      <td>Entire home/apt</td>
      <td>{TV,"Cable TV",Internet,"Wireless Internet",Ki...</td>
      <td>4</td>
      <td>1.0</td>
      <td>Real Bed</td>
      <td>flexible</td>
      <td>True</td>
      <td>SF</td>
      <td>This light-filled home-away-from-home is super...</td>
      <td>NaN</td>
      <td>t</td>
      <td>t</td>
      <td>NaN</td>
      <td>2015-04-19</td>
      <td>f</td>
      <td>NaN</td>
      <td>37.772004</td>
      <td>-122.431619</td>
      <td>Beautiful Flat in the Heart of SF!</td>
      <td>Lower Haight</td>
      <td>0</td>
      <td>NaN</td>
      <td>https://a0.muscache.com/im/pictures/72208dad-9...</td>
      <td>94117.0</td>
      <td>2.0</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Apartment</td>
      <td>Entire home/apt</td>
      <td>{TV,Internet,"Wireless Internet","Air conditio...</td>
      <td>2</td>
      <td>1.0</td>
      <td>Real Bed</td>
      <td>moderate</td>
      <td>True</td>
      <td>DC</td>
      <td>Cool, cozy, and comfortable studio located in ...</td>
      <td>2015-05-12</td>
      <td>t</td>
      <td>t</td>
      <td>100%</td>
      <td>2015-03-01</td>
      <td>t</td>
      <td>2017-01-22</td>
      <td>38.925627</td>
      <td>-77.034596</td>
      <td>Great studio in midtown DC</td>
      <td>Columbia Heights</td>
      <td>4</td>
      <td>40.0</td>
      <td>NaN</td>
      <td>20009</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
</div>



We can see some missing values in the data. Let's dive deep into this problem.


```python
msno.matrix(X_raw.sample(250))
```




    <matplotlib.axes._subplots.AxesSubplot at 0x22da68d8548>




![png](output_15_1.png)


<p style="text-align: justify">Some columns have a lot of missing values, so it is not reasonable to remove all those rows from the dataset. Also, by dropping those columns we would lose valuable information to the models. Therefore, the approach I'm going to use is to replace the missing values by the string 'NA' if the variable is categorical, or by 0 if the variable is numerical. Besides replacing by 0 in the numerical case, I'll add boolean features that indicate if the row had a missing value or not. After that, for the columns with few missing values, such as <i>bathrooms</i>, <i>zipcode</i> and <i>beds</i>, I'm just going to drop those rows. Finally, I will also drop the column <i>thumbnail_url</i>, as it doesn't provide useful information, the columns <i>zipcode</i>, as the latitude and longitude already provide a geolocation feature and the text variables <i>description</i> and <i>name</i> in order to simplify the problem and avoid NLP techniques that will increase too much the number of features.</p>


```python
X = X_raw.drop(columns=["thumbnail_url", "description", "name", "zipcode"])
X[["first_review_NA", "host_response_rate_NA", "last_review_NA", "review_scores_rating_NA"]] = X_raw[["first_review", "host_response_rate", "last_review", "review_scores_rating"]].fillna("NA").apply(lambda col: col.apply(lambda x: 1 if x=="NA" else 0))
X[["first_review", "last_review", "review_scores_rating"]] = X_raw[["first_review", "last_review", "review_scores_rating"]].fillna(0)
X.neighbourhood = X_raw.neighbourhood.fillna("NA")
X.host_response_rate = X_raw.host_response_rate.fillna("0%")
print(X.shape[0])
X.dropna(inplace=True)
print(X.shape[0])
```

    74111
    73579
    

By replacing missing values by 0, I'm treating true 0 the same as missing values. This may break the linearity of the model for the linear regression, so that's why I chose to add boolean features too. By removing the missing values, our feature dataset has now around 2,000 less rows.

Now, we need to convert some data types that are not the most appropriate for ML training. I'll convert date strings into number of days until a reference date, I'll convert the <i>amenities</i> column into number of amenities, I'll convert all boolean variables to 0 and 1 and <i>host_response_rate</i> to numeric.


```python
try:
    today = dt.today()
    X.amenities = X.amenities.str.split(",").apply(len)
    X[["first_review", "last_review"]] = X[["first_review", "last_review"]].apply(lambda col: col.apply(lambda x: (today-dt.fromisoformat(x)).days if x!=0 else x))
    X.host_response_rate = X.host_response_rate.str.replace("%", "").astype(int)
    X[["host_has_profile_pic", "host_identity_verified", "instant_bookable"]] = X[["host_has_profile_pic", "host_identity_verified", "instant_bookable"]].apply(lambda col: col.apply(lambda x: 1 if x == "t" else 0))
    X.cleaning_fee = X.cleaning_fee.apply(lambda x: 1 if x==True else 0)
except:
    pass
```

Let's now explore the data a little more:


```python
X.describe()
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
      <th>amenities</th>
      <th>accommodates</th>
      <th>bathrooms</th>
      <th>cleaning_fee</th>
      <th>first_review</th>
      <th>host_has_profile_pic</th>
      <th>host_identity_verified</th>
      <th>host_response_rate</th>
      <th>instant_bookable</th>
      <th>last_review</th>
      <th>latitude</th>
      <th>longitude</th>
      <th>number_of_reviews</th>
      <th>review_scores_rating</th>
      <th>bedrooms</th>
      <th>beds</th>
      <th>first_review_NA</th>
      <th>host_response_rate_NA</th>
      <th>last_review_NA</th>
      <th>review_scores_rating_NA</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>73579.000000</td>
      <td>73579.000000</td>
      <td>73579.000000</td>
      <td>73579.000000</td>
      <td>73579.000000</td>
      <td>73579.000000</td>
      <td>73579.000000</td>
      <td>73579.000000</td>
      <td>73579.000000</td>
      <td>73579.000000</td>
      <td>73579.000000</td>
      <td>73579.000000</td>
      <td>73579.000000</td>
      <td>73579.000000</td>
      <td>73579.000000</td>
      <td>73579.000000</td>
      <td>73579.000000</td>
      <td>73579.000000</td>
      <td>73579.000000</td>
      <td>73579.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>17.635331</td>
      <td>3.160888</td>
      <td>1.236039</td>
      <td>0.734884</td>
      <td>1244.040963</td>
      <td>0.996956</td>
      <td>0.673263</td>
      <td>71.252708</td>
      <td>0.263023</td>
      <td>912.591514</td>
      <td>38.441312</td>
      <td>-92.441681</td>
      <td>20.889425</td>
      <td>72.865437</td>
      <td>1.266802</td>
      <td>1.712975</td>
      <td>0.213920</td>
      <td>0.244920</td>
      <td>0.213498</td>
      <td>0.225472</td>
    </tr>
    <tr>
      <th>std</th>
      <td>6.923991</td>
      <td>2.156297</td>
      <td>0.582947</td>
      <td>0.441398</td>
      <td>781.401863</td>
      <td>0.055092</td>
      <td>0.469023</td>
      <td>42.988835</td>
      <td>0.440278</td>
      <td>517.130914</td>
      <td>3.081897</td>
      <td>21.711043</td>
      <td>37.818332</td>
      <td>39.913863</td>
      <td>0.853585</td>
      <td>1.256094</td>
      <td>0.410074</td>
      <td>0.430043</td>
      <td>0.409779</td>
      <td>0.417896</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>33.338905</td>
      <td>-122.511500</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>13.000000</td>
      <td>2.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>1011.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>964.000000</td>
      <td>34.126898</td>
      <td>-118.342867</td>
      <td>1.000000</td>
      <td>80.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>17.000000</td>
      <td>2.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1341.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>100.000000</td>
      <td>0.000000</td>
      <td>1031.000000</td>
      <td>40.661481</td>
      <td>-76.998494</td>
      <td>6.000000</td>
      <td>94.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>22.000000</td>
      <td>4.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1701.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>100.000000</td>
      <td>1.000000</td>
      <td>1152.000000</td>
      <td>40.746124</td>
      <td>-73.954688</td>
      <td>23.000000</td>
      <td>99.000000</td>
      <td>1.000000</td>
      <td>2.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>86.000000</td>
      <td>16.000000</td>
      <td>8.000000</td>
      <td>1.000000</td>
      <td>4201.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>100.000000</td>
      <td>1.000000</td>
      <td>4136.000000</td>
      <td>42.390437</td>
      <td>-70.985047</td>
      <td>605.000000</td>
      <td>100.000000</td>
      <td>10.000000</td>
      <td>18.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>



There doesn't seem to be any anomalous value for the numeric features. Now, the last thing to do is to convert all categorical variables into one-hot encoding format.


```python
# Coming soon
```

In order to train and evaluate different machine learning models, this dataset must also be split into a training and test datasets. Here, I'm going to use a 80/20 split.


```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

## Baseline model

<p style="text-align: justify">The baseline model for a regression problem is to simply predict without the additional information brought by the features, that is, using only the target variable. The best predictor, which minimizes the RMSE, is the average of the distribution of the target variable, and thus will be used as baseline of comparison with other models.</p>
<p style="text-align: justify">In fact, if we use the R² as the evaluation metric, this baseline is already incorporated in the formula, so all the performances will be measured relative to this baseline.</p>


```python
y.mean()
```




    4.782069108304868



For this problem, the baseline model is to always predict 4.78 as the log of the price.

## First model: Linear Regression

COMING SOON ...

## Second model: Decision Tree

COMING SOON ...

## Third model: Random Forest

COMING SOON ...

## Conclusion

COMING SOON ...
