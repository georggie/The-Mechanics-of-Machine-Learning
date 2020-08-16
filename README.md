# The Mechanics of Machine Learning

## 1. Terminology

Machine learning uses a **_model_** to capture the relationship between **_feature vectors_** and some **_target variables_** within a **_training data set_**. A feature vector is a set of features or attributes that describe a particular object (e.g. an apartment can be described with: number of bedrooms, number of bathrooms, and location of an apartment (latitude, longitude)). The target is either a scalar value like rent price or a categorical value / class such as “healthy” or “not healthy”. If the target is a numeric value, we're building a **_regressor_**. If the target is a discrete category or class, we're building a **_classifier_**.


Regressors draw curves through the data |  Classifiers draw curves through the space separating classes
:-------------------------:|:-------------------------:
![Regressor](https://i.imgur.com/LG9Rxoc.png)  |  ![Classifier](https://i.imgur.com/iBw0v57.png)

Predictors are usually fitting curves to data and classifiers are drawing decision boundaries in between data points associated with the various categories. In other words, regressors are predicting some continuous numeric value while classifiers are predicting a class. 

Machine learning tasks that involve both feature vectors and target/dependent variables fall into the **_supervised learning category_**. **_Unsupervised learning_** tasks involve just feature vectors without the dependent variable. The most common unsupervised task is called clustering that attempts to cluster similar data points together (similar to the right picture above). The goal of clustering is to discover both the number of categories and assign records to categories. The process of computing model parameters is called **_training the model_** or **_fitting a model_** to the data. If a model is unable to capture the relationship between feature vectors and targets, the model is **_underfitting_**. At the other extreme, a model is **_overfitting_** if it is too specific to the training data and doesn't generalize well (predicts unseen values poorly). To test generality, we either need to be given a **_validation set_** as well as a **_training set_**, or we need to split the provided single data set into a training set and a validation set. The model is exposed only to the training set, reserving the validation set for measuring generality and tuning the model. There is a **_test set_** as well and it is used as a final test of generality; the test set is never used while training or tuning the model. 


## 2. A First Taste of Applied Machine Learning

In this section, we built and used both regressor and classifier models, which are really just two sides of the same coin. Regressors learn the relationship between features and **numeric** target variables whereas classifiers learn the relationship between features and a set of **target classes** or **categories**.

```.python
# load data frame
dataset = pd.read_csv(datafile)

# features
X = dataset[[feature column names of interest]]

# target
y = dataset[target column name]

# build model
model = ChooseYourModel(hyper-parameters)
model.fit(X,y)
```

We've primarily used **_RandomForestRegressor_** and **_RandomForestClassifier_** in the _ChooseYourModel_ slot. The **hyper-parameters** of a model represent the key structural or mathematical arguments, such as the number of trees in a random forest.

To make a prediction using model `model` for some test record, call method `predict()`:

```
# make predictions
y_pred = model.predict(test record)
```

For basic testing purposes, we split the data set into 80% training and 20% validation sets (the hold out method):

```.python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
```

Computing a validation score for any model is as simple as:

```.python
# measure performance
score = model.score(X_test, y_test)
```

Method `score()` returns accuracy (in range 0-1) for classifiers and a common metric called R<sup>2</sup> ("R squared") for regressors.  It measures how well a regressor performs compared to a trivial model that always returns the average of the target (such as apartment price) for any prediction. 1.0 is a perfect score, 0 means the model does no better than predicting the average, and a value < 0 indicates the model is worse than just predicting the average. Of course, we can also compute other metrics that are more meaningful to end-users when necessary, such as the mean absolute error we computed for apartment prices.

Most of the work building a model involves data collection, data cleaning, filling in missing values, feature engineering, and proper test set identification. Furthermore, all features fed to a model must be numeric, rather than strings like names or categorical variables like low/medium/high, so we have some data conversions to do. Even with perfect training data, remember that a model can only make predictions based on the training data we provide. Models don't necessarily have the same experience we do, and certainly don't have a human's modeling power.