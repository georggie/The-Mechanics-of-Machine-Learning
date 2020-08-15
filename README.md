# The Mechanics of Machine Learning

## 1. Terminology

Machine learning uses a **_model_** to capture the relationship between **_feature vectors_** and some **_target variables_** within a **_training data set_**. A feature vector is a set of features or attributes that describe a particular object (e.g. an apartment can be described with: number of bedrooms, number of bathrooms, and location of an apartment (latitude, longitude)). The target is either a scalar value like rent price or a categorical value / class such as “healthy” or “not healthy”. If the target is a numeric value, we're building a **_regressor_**. If the target is a discrete category or class, we're building a **_classifier_**.


Regressors draw curves through the data |  Classifiers draw curves through the space separating classes
:-------------------------:|:-------------------------:
![Regressor](https://i.imgur.com/LG9Rxoc.png)  |  ![Classifier](https://i.imgur.com/iBw0v57.png)

Predictors are usually fitting curves to data and classifiers are drawing decision boundaries in between data points associated with the various categories. In other words, regressors are predicting some continuous numeric value while classifiers are predicting a class. 

Machine learning tasks that involve both feature vectors and target/dependent variables fall into the **_supervised learning category_**. **_Unsupervised learning_** tasks involve just feature vectors without the dependent variable. The most common unsupervised task is called clustering that attempts to cluster similar data points together (similar to the right picture above). The goal of clustering is to discover both the number of categories and assign records to categories. The process of computing model parameters is called _training the model_ or _fitting a model_ to the data. If a model is unable to capture the relationship between feature vectors and targets, the model is **_underfitting_**. At the other extreme, a model is **_overfitting_** if it is too specific to the training data and doesn't generalize well (predicts unseen values poorly). To test generality, we either need to be given a **_validation set_** as well as a **_training set_**, or we need to split the provided single data set into a training set and a validation set. The model is exposed only to the training set, reserving the validation set for measuring generality and tuning the model. There is a **_test set_** as well and it is used as a final test of generality; the test set is never used while training or tuning the model. 