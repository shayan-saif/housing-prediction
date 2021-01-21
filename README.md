## INFR 3700 Machine Learning
### Final Project
For our final project in INFR 3700 Machine Learning, we explored different regression algorithms to predict the price of a house based on several factors (number of rooms, sq foot, geographical location, etc.)

Authored by Shayan Saif and Reese Daniel

# Machine Learning Final Project Report

November 11, 2020
 INFR 3700
 Reese Daniel &amp; Shayan Saif
 University of Ontario Institute of Technology

## Introduction

The United States of America is a large country with a diverse range of places to live. There are urban and rural environments, big mansions and shoebox apartments, and prices that range dramatically depending on location (location, location!). In this project, we aim to use recent, existing data to train machine learning models to accurately predict how much a given housing unit should cost per month, depending on a variety of factors.

The USA Housing Listings dataset, which is the dataset we elected to use for this project, scrapes data from Craigslist related to apartments, condos, houses, and other housing types available for rent across the United States. The dataset includes around 385,000 observations and 22 columns. Columns include price per month, type of property (apartment, house, townhouse, condo, etc.), square footage, number of beds, number of bathrooms, latitude, longitude, US state, whether cats/dogs are allowed, whether smoking is allowed, if the property is wheelchair accessible, if there is electric vehicle charging, and if the property comes furnished. The data were first scraped from Craigslist in December 2019, and were most recently updated in June 2020.

## Data Graphs &amp; Insights ![](RackMultipart20210121-4-qotoiq_html_4858336d11d3df72.png)

Frequencies of housing types. Apartments constituted the hefty majority of the dataset. For the sake of simplicity and to avoid the need to do one hot encoding, all housing types besides apartments were removed. Thus, the &#39;type&#39; column was removed as well.

![](RackMultipart20210121-4-qotoiq_html_cefcfb5ac86f7060.png)

Histogram of housing prices. Most housing listings in the dataset are between $500 and $4,000, indicating that our model will most likely be much better at predicting prices for apartments that cost only a few thousand dollars per month. To prevent outliers from negatively affecting our models, we removed all apartments with prices greater than $10,000. There were also 1070 rows with a price of $0. We assumed that these are outliers because the listers probably just did not want to share online the price of their units for rent. These rows were removed.

![](RackMultipart20210121-4-qotoiq_html_22ed9d39a41f7fe3.png)

Histogram of housing square feet. Most house listings in the dataset are less than 1500 square feet. 665 rows indicated a square footage of less than 100 sq. ft., so these rows were removed, since they were likely errors.

![](RackMultipart20210121-4-qotoiq_html_7a2fda5a918e37e1.png)

Scatter showing number of beds vs. number of bathrooms. Larger markers indicate a higher occurrence of that bed/bathroom combination. By far, the most common units have 1 bed/1 bathroom, 2 beds/2 bathrooms, or 2 beds/1 bathroom. The range of beds is [0,8] and the range of bathrooms is also [0,8], giving us 64 different combinations.

![](RackMultipart20210121-4-qotoiq_html_452dddb8fdd6b439.png)

When the data are plotted on a world map, we can see that a number of points have incorrect latitudes and longitudes, since this dataset is supposed to only reflect the United States. All points outside of the continental United States were removed. ![](RackMultipart20210121-4-qotoiq_html_b2a08cdee952a29b.png)

If we zoom in on the continental US and color each point by state, we see that the dataset covers close to the entire country, with only a little bit of sparseness in the west. This means that our models should be able to predict pricing accurately irrespective of state.

## Data Preparation

The following columns were removed from the dataset, as they took up valuable space and were not needed for the machine learning algorithms:

- ID (a unique ID for each row)
- URL (the URL for the Craigslist listing)
- Region (the name of the region the listing was in)
- Region\_URL (the URL for the region&#39;s website)
- Image\_URL (the URL for the listing&#39;s image)
- Description (some text describing the property)
- Laundry\_Options (an enum of 5 different types of laundry)
- Parking\_Options (an enum of 7 different types of parking)

1,920 rows were dropped because they contained missing values. 2 housing units were listed as having 1000 beds and 2 other units were listed as having 1100 beds, so these 4 outliers were removed. The state column was removed as this information can be deduced from the longitude and latitude. This left us with the following columns: price; sqfeet; beds; baths; cats\_allowed; dogs\_allowed; smoking\_allowed; wheelchair\_access; electric\_vehicle\_charge; comes\_furnished; latitude; longitude. The data were finally split into 80% training, 20% testing sets. From each set, the price and state columns were removed.

## Machine Learning Algorithms

### Linear Regression

Using the linear regression class from the sklearn.linear\_model library, we fitted the model with the training features and the prices for those features. Afterwards, the algorithm was able to determine the coefficients of the features and the intercept. Furthermore, the model made predictions based on the test data and had a root mean squared error of 511.68.

### Decision Tree Regressor

We imported the Decision Tree Regressor class from the sklearn.tree library. We initialized it with a max\_depth of 4, a minimum leaf sample of 0.1, and random\_state set to 3. Afterwards, we fitted the model with the training dataset and then predicted the test dataset. Afterwards, we extracted the root mean squared error, which compared to the linear regression model. It was 464.74, making it the better algorithm in this case.

### Deep Neural Network

We scaled the training and testing features using the StandardScaler from the preprocessing scikit learn library. This ensured that the algorithm could easily distinguish between the different values it was given. Afterwards, we created the model using the following layers:

![](RackMultipart20210121-4-qotoiq_html_8f2653df1997db55.png)

The model takes in 11 features (sqfeet, beds, baths, etc.) and processes it through three hidden relu layers of size 5, and outputs 1 value. The model was compiled using mean squared error as the loss, adam as the optimizer, and mean squared error as the metric. Finally, the model was run using the training and testing dataset. The results were very poor and the model improved very slightly through 10 epochs. We believe this may be because the optimizer had found a local minimum for the loss and could not be improved further.

### Performance

| Algorithm | Root Mean Squared Error | Data Mean | Accuracy (%) |
| --- | --- | --- | --- |
| Linear Regression | 511.68 | 1151.66 | 55.57% |
| Decision Tree Regressor | 464.74 | 59.65% |
| Deep Neural Network | 516.16 | 44.82% |

### Graphs

Decision Tree Regressor was the most accurate regression model for this dataset. Here are its predictions for different features as the x-axis.

![](RackMultipart20210121-4-qotoiq_html_4d7c084686021314.png) ![](RackMultipart20210121-4-qotoiq_html_e46a151029eb0a66.png)

![](RackMultipart20210121-4-qotoiq_html_fa136b82e45c66ce.png)

![](RackMultipart20210121-4-qotoiq_html_5c88c395e45c4b9f.png)

## Appendix

### Link to Dataset

[https://www.kaggle.com/austinreese/usa-housing-listings](https://www.kaggle.com/austinreese/usa-housing-listings)
