# The Mechanics of Machine Learning

## 1. Introduction

Machine learning uses a **_model_** to capture the relationship between **_feature vectors_** and some **_target variables_** within a **_training data set_**. A feature vector is a set of features or attributes that describe a particular object (e.g. an apartment can be described with: number of bedrooms, number of bathrooms, and location of an apartment (latitude, longitude)). The target is either a scalar value like rent price or a categorical value / class such as “healthy” or “not healthy”. If the target is a numeric value, we're building a **_regressor_**. If the target is a discrete category or class, we're building a **_classifier_**.


Regressors draw curves through the data |  Classifiers draw curves through the space separating classes
:-------------------------:|:-------------------------:
![Regressor](https://i.imgur.com/LG9Rxoc.png)  |  ![Classifier](https://i.imgur.com/iBw0v57.png)

Predictors are usually fitting curves to data and classifiers are drawing decision boundaries in between data points associated with the various categories. In other words, regressors are predicting some continuous numeric value while classifiers are predicting a class. 

Machine learning tasks that involve both feature vectors and target/dependent variables fall into the **_supervised learning category_**. **_Unsupervised learning_** tasks involve just feature vectors without the dependent variable. The most common unsupervised task is called clustering that attempts to cluster similar data points together (similar to the right picture above). The goal of clustering is to discover both the number of categories and assign records to categories. The process of computing model parameters is called **_training the model_** or **_fitting a model_** to the data. If a model is unable to capture the relationship between feature vectors and targets, the model is **_underfitting_**. At the other extreme, a model is **_overfitting_** if it is too specific to the training data and doesn't generalize well (predicts unseen values poorly). To test generality, we either need to be given a **_validation set_** as well as a **_training set_**, or we need to split the provided single data set into a training set and a validation set. The model is exposed only to the training set, reserving the validation set for measuring generality and tuning the model. There is a **_test set_** as well and it is used as a final test of generality; the test set is never used while training or tuning the model. 


## 2. Building a model

**Notebook**: A First Taste of Applied Machine Learning

In this section, we built and used both regressor and classifier models, which are really just two sides of the same coin. Regressors learn the relationship between **features** and **numeric** target variables whereas classifiers learn the relationship between **features** and a set of **target classes** or **categories**.

```.python
# load data into pandas' data frame
dataset = pandas.read_csv(<path to file>)

# extract interesting features
X = dataset[[feature column names of interest]]

# extract target variable
y = dataset[target column name]

# build model
model = ChooseYourModel(hyper-parameters)
model.fit(X,y)
```

We've primarily used **_RandomForestRegressor_** and **_RandomForestClassifier_** in the **_ChooseYourModel_** slot. The **hyper-parameters** of a model represent the key structural or mathematical arguments, such as the number of trees in a random forest.

To make a prediction using model `model` for some test record, call method `predict()`:

```.python
# make predictions
y_predicted = model.predict(test record)
```

For basic testing purposes, we split the data set into 80% training and 20% validation set (the **hold out** method):

```.python
# split data into training 80% and test set 20%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
```

Computing a validation score for any model is as simple as:

```.python
# measure performance
score = model.score(X_test, y_test)
```

Method `score()` returns accuracy (in range 0-1) for classifiers and a common metric called R<sup>2</sup> ("R squared") for regressors.  It measures how well a regressor performs compared to a trivial model that always returns the average of the target (such as apartment price) for any prediction. 1.0 is a perfect score, 0 means the model does no better than predicting the average, and a value < 0 indicates the model is worse than just predicting the average. Of course, we can also compute other metrics that are more meaningful to end-users when necessary, such as the mean absolute error or sum of squared errors.

Most of the work building a model involves data collection, data cleaning, filling in missing values, feature engineering, and proper test set identification. Furthermore, all features fed to a model must be numeric, rather than strings like names or categorical variables like low/medium/high, so we have some data conversions to do. Even with perfect training data, remember that a model can only make predictions based on the training data we provide. Models don't necessarily have the same experience we do, and certainly don't have a human's modeling power.

## 3. Basics of NumPy, Pandas and Matplotlib

**Notebook**: Development Tools

### Pandas

```.python
# import pandas library and give a short alias: pd
import pandas as pd

# read CSV
df = pd.rea0d_csv(PATH)

# save as CSV
pd.to_csv(PATH)

# get first n rows of the data set
df.head(n)
df.head(n).T # transpose first n rows

# get meta information about the data frame
df.info()

# to learn more about data itself
df.describe()

# sort data frame by attribute in descending order
df.sort_values(attribute, ascending=False)

# take column
# return value is of type numpy series
col = df['column_name']

# on numpy series we can execute some useful functions
col.mean(), col.std(), col.max()

# extracting columns
df_new = df[['col1', 'col2', ...]]

# extract rows (2,3,4)
df_new = df.iloc[2:5]

# axis = 1 ~ columns, axis = 0 ~ rows
bedrooms bathrooms latitude longitude price
0	1	1.0	40.7108	-73.9539 2400
1	2	1.0	40.7513	-73.9722 3800
2	2	1.0	40.7575	-73.9625 3495
3	3	1.5	40.7145	-73.9425 3000

df.loc[1:3, 'bathrooms':'longitude']

bathrooms latitude longitude
1	1.0	40.7513	-73.9722
2	1.0	40.7575	-73.9625
3	1.5	40.7145	-73.9425

df.iloc[1:4, 1:4]

bathrooms latitude longitude
1	1.0	40.7513	-73.9722
2	1.0	40.7575	-73.9625
3	1.5	40.7145	-73.9425

# checks is there a null value in the columns
df.isnull()

# checks is there any null value in the columns
df.isnull().any()

# data frame query
df[(df.col1 > val1) & (df.col2 < val2)]
```

### Matplotlib

```.python
# figure and plots
figure, (subplot_first, subplot_second) = plt.subplots(1, 2)

subplot_first.scatter(x, y)
subplotf.set_xlabel('X Label')
subplotf.set_ylabel('Y Label')

subplot_second.hist(data, bins=50, edgecolor='white')
subplot_second.set_xlabel('X Label')
subplot_second.set_ylabel('Y Label');

# to adjust padding between the plots
plt.tight_layout()
```

### NumPy

```.python
# create 1D vector with 5 numbers
a = np.array([1,2,3,4,5])

print(f"type is {type(a)}")
print(f"dtype is {a.dtype}, 64-bit integer")
print(f"ndim is {a.ndim}, 1D array")
print(a)

 type is <class 'numpy.ndarray'>
 dtype is int64, 64-bit integer
 ndim is 1, 1D array
 [1 2 3 4 5]

a = np.random.randint(0, 100, (5,4))

# flatten the matrix and sum it
np.sum(a.flat)

# the flat property is an iterator that is more space efficient than iterating 
# over u.ravel(), which is an actual 1D array of the matrix elements
display(a, a.ravel())
```

## 4. Feature Engineering and Exploratory Data Analysis (Basics)

**Notebooks**: Exploring and Denoising Your Data Set, Categorically Speaking

To train a model, the data set must follow two fundamental rules: all data must be numeric and there can't be any missing values. We must derive numeric features from the nonnumeric features such as strings, dates and categorical variables. The data could have outliers, errors, or contradictory information.

We need to know what the data looks like so our first inspection of the data should yield the column names their datatypes, and whether the target column has numeric values or categories. (If we're creating a regressor, those values must be numeric; if we're classifying, those values must be categories)

* Check if there is missing values in a dataset

```.python
>>> df.isnull().any()
>>> df.isnull().sum()
```

* Get a baseline of the model trained on the "raw" data (no feature engineering, no dealing with outliers and errors). A high score just means that it's possible there is a relationship between features and target and captured by the model. If, however, we can't get a high score, it's an indication that there is no relationship or the model is simply unable to capture it.

* Describe the data set. Check results carefully (min, max for outliers for example). Noise and outliers are potential problems because they can lead to inconsistencies. An inconsistency is a set of similar or identical feature vectors with much different target values.

```.python
>>> df_numeric.describe()
```

* We can either leave noisy or outlier records as-is, delete, or "fix" the records. It's always best to use domain knowledge when identifying outliers, but if we are uncertain about an appropriate range, we can always clip out the bottom and top 1% using a bit of NumPy code.

* Transforming the target variable (using the mathematical log function) into a tighter, more uniform space makes life easier for any model. During training, RFs combine the targets of identical or nearly-identical features by averaging them together, thus, forming the prediction. But, outlier prices wildly skew average targets, so the model's predictions could be very far off. We need to shrink large values a lot and smaller values a little (to fight with ourliers). That magic operation is called the **logarithm** or **log** for short.


Creating a good model is more about **_feature engineering_** than it is about choosing the right model. Feature engineering means improving, acquiring, and even _synthesizing_ features that are strong predictors of your model's target variable. Synthesizing features means deriving new features from existing features or injecting features from other data sources.

We will focus on categorical variables. They take values from a finite set of choices. They can be ordinal and nominal. Ordinal categorical variables have values that can be ordered even if they are not actual numbers. Ordinal variables are the easiest to convert from strings to numbers because we can simply assign a different integer for each possible ordinal value. For example **intereset_level** feature can be encoded as {low: 1, medium: 2, high: 3}.

```.python
df['category_ordinal'] = df['category_ordinal'].map({'low':1,'medium':2,'high':3})
```

On the other hand, we have nominal values for which there is no meaningful order between the category values. The easy way to remember the difference between ordinal and nominal variables is that ordinal variables have order and nominal comes from the word for "name" in Latin (nomen) or French (nom). The first technique to deal with categorical variables is called _label encoding_ and simply converts each category to a numeric value. Another encoding approach is called _frequency encoding_ and it converts categories to the frequencies with which they appear in the training.

```.python
# label encoding
df['category_nominal'] = df['category_nominal'].astype('category_nominal').cat.as_ordered()
df['category_nominal'] = df['category_nominal'].cat.codes + 1

# frequency encoding
counts = df['category_nominal'].value_counts()
df['category_nominal'] = df['category_nominal'].map(counts)
```

Creating features that incorporate information about the target variable is called target encoding and is often used to derive features from categorical variables to great effect. One of the most common target encodings is called **_mean encoding_**, which replaces each category value with the average target value associated with that category.

## 5. Train, Validate, Test

Developing a machine learning model requires three sets of observations: training, validation, and test sets. The model trains just on the training set and model accuracy is evaluated using the validation set during development. After tuning the model on the validation set, we run the test set through the model to get our final measure of model accuracy and generality. If we peek at the test set and run it through an intermediate model rather than our final model, the test set becomes just another validation set. Every change made to a model after testing it on a dataset, tailors the model to that dataset; that dataset is no longer an objective measure of generality. To develop a model in practice, we're usually given a single dataset, rather than separate training, validation, and test sets. That means we need a general procedure for splitting datasets appropriately.

### Splitting time-insensitive datasets

For datasets that do not change significantly over the time period of interest, we want to extract validation and test sets using random sampling of records. This is called the **holdout method**. To get (roughly) 70% of dataframe df into training and 15% into both validation and test sets, we can do this:

```
from sklearn.model_selection import train_test_split

# shuffle data
df = df.sample(frac=1)

df_dev, df_test = train_test_split(df, test_size=0.15)
df_train, df_valid = train_test_split(df_dev, test_size=0.15)
```

After training a model using **_df\_train_**, we'd run **_df\_valid_** data through the model and compute a metric, such as R<sup>2</sup>. Then we'd tune the model so that it's more accurate on **_df\_valid_** data. When we're happy with the model, we'd finally use **_df\_test_** to measure generality.

Another method is called **k-fold cross validation** that splits the dataset into **k** chunks of equal size. We train the model on **k-1** chunks and test it on the other, repeating the procedure **k** times so that we every chunk gets used as a validation set. The overall validation error is the average of the **k** validation errors. 

Here's how to use sklearn for 5-fold cross validation using an RF model:

```
from sklearn.model_selection import cross_val_score

rf = RandomForestRegressor(...)
scores = cross_val_score(rf, X, y, cv=5) # k=5

print(scores.mean())
```

Cross validation and repeated subsampling are excellent techniques for measuring model accuracy, but are unsuitable for time-sensitive datasets.

### Splitting time-sensitive datasets

When observation features or target variables change meaningfully over time, random extraction of validation sets isn't appropriate. Randomly splitting a dataset would yield training and validation sets that overlap in time. That's a problem because it allows the model to train on data from the future and validation metrics would be overly optimistic. Imagine how your model would be used in practice. At some point, you must train a model on the data you have and then deploy it. Any observations subsequently submitted to the model for prediction are necessarily from dates beyond the end of the data used to train the model. Training should always mimic deployment and so our validation set should be from dates beyond the end of the training set.

The process for extracting training, validation, and test sets for time-sensitive data is:

- Sort the records by date, earliest to latest
- Extract the last, say, 15% of the records as **_df\_test_**
- Extract the second to last 15% of the records as **_df\_valid_**
- The remaining 70% of the original data is **_df\_train_**

In an ideal world, datasets would be purely numeric and without missing values. Feature engineering would still be useful, but numericalizing data such as encoding categorical variables, wouldn't be necessary. And we wouldn't have to conjure up missing values. However, real-world datasets are full of categorical variables and riddled with missing values, which introduces synchronization issues between training and validation/test sets.

- If category CAT1 in column COL is encoded as integer value 1 in the training set, the validation and test set must use the same encoding of 1.

- Missing categorical values should be encoded as integer 0 in all sets.

- Missing numeric values in column COL should be filled with the median of just those values from COL in the training set.

- Categorical values in validation or test sets not present in the training set, should be encoded as integer 0; the model has never seen those values, so we encode such values as if they were missing.

We can abstract that list into these important rules for preparing separated training and test sets:

1. Transformations must be applied to features consistently across data subsets.

2. Transformations of validation and test sets can only use data derived from the training set.

To follow those rules, we have to remember all transformations done to the training set for later application to the validation and test sets. In practice, that means tracking the median of all numeric columns, all category to category-to-code mappings, and which categories were one-hot encoded. Special care is required to ensure that one-hot encoded variables use the same name and number of columns in the training and testing sets. It sounds simple enough, but it's easy to screw up the synchronization between training and testing sets. Synchronization bugs usually show up as poor model accuracy, rather than as something obvious like a program exception.
