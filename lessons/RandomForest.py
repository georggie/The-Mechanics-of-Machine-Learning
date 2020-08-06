import numpy as np
import pandas as pd

class TreeEnsemble():


    def __init__(self, number_of_trees=10, leaf_min_samples=1, sample_size=None):
        """
        TreeEnsemble constructor.

        Parameters:
        -----------
        number_of_trees: Total number of trees in an ensemble.

        leaf_min_samples: The minimum number of data points in the 
        leaf of a tree.

        sample_size: Size of a sample that each individual tree will use in order to grow. 
        If None, the entire data set will be used for growing (training).

        Examples:
        ---------
        >>> model = TreeEnsemble()

        >>> model = TreeEnsemble(number_of_trees=50, leaf_min_samples=10, oob_score=True)
        """
        self.number_of_trees = number_of_trees
        self.leaf_min_samples = leaf_min_samples
        self.sample_size = sample_size

    def learn(self, independent, dependent):
        """
        Trains the forest of decision trees.
        """
        self.independent = independent
        self.dependent = dependent

        # create trees in an ensamble (forest)
        self.trees = [self._grow_tree() for i in range(self.number_of_trees)]

    def predict(self, validation):
        """
        Predicts the values from the validation set.
        """
        return np.mean([tree.predict(validation) for tree in self.trees], axis=0)

    def _grow_tree(self):
        """
        Grows a single tree.
        """        
        return DecisionRegressionTree(self.independent, self.dependent, sample_size=self.sample_size, leaf_min_samples=self.leaf_min_samples)


class DecisionRegressionTree():


    def __init__(self, independent, dependent, sample_size=None, leaf_min_samples=5):
        """
        DecisionRegressionTree constructor.

        Parameters
        ----------
        independent: The data frame of one or more columns 
        used to predict the dependent variable.

        dependent: The value of each point in the independent set 
        and also the variable that model tries to predict.

        leaf_min_samples: The minimum number of data points in the 
        leaf of a tree.

        sample_size: The size of the sample used to grow the tree.

        Return
        ------
        DecisionRegressionTree.

        Examples
        --------
        >>> model = DecisionRegressionTree(independent, dependent)
            <RandomForest.DecisionRegressionTree at 0x7f210c5f7160>
        >>> model.value, model.score, model.best_split_value, model.best_split_column
            (3438.297950310559, 51504254.67458592, 1.0, 'bathrooms')
        >>> model.left_child, model.right_child
            (<RandomForest.DecisionRegressionTree at 0x7f20d46b5850>,
             <RandomForest.DecisionRegressionTree at 0x7f20d46b5f70>)
        """
        self.leaf_min_samples = leaf_min_samples

        if sample_size is None:
            self.independent = independent.reset_index(drop=True)
            self.dependent = dependent.reset_index(drop=True)
        else:
            indexes = np.random.permutation(len(dependent))[:sample_size]
            self.independent = independent.iloc[indexes,:].reset_index(drop=True)
            self.dependent = dependent.iloc[indexes].reset_index(drop=True)
        
        # if the user has passed a series or an array, we will try to convert it to the pandas' data frame 
        if len(self.independent.shape) == 1:
            self.independent = pd.DataFrame(self.independent)
        
        # calculate the format / size of the independent variables
        self.rows, self.columns = self.independent.shape
        
        # set the value to the mean of dependent variable and score to infinity 
        self.value = np.mean(self.dependent)
        self.score = float('inf')

        # mark best split column and split value
        self.best_split_column = None
        self.best_split_value = None
        self.best_split_column_index = None

        # finally, grow the tree
        self._grow()

    def predict(self, validation):
        """
        Predicts the values from the validation set.
        """
        return np.array([self.predict_row(row) for row in validation.values])

    def predict_row(self, row):
        """
        Predicts the value of a row based on the grown tree.
        """
        subtree = self

        if subtree.score == float('inf'):
            return subtree.value
        elif row[subtree.best_split_column_index] <= subtree.best_split_value:
            subtree = subtree.left_child
        else:
            subtree = subtree.right_child

        return subtree.predict_row(row)

    def _grow(self):
        """
        Grows the three starting from the root by finding the best split on the features.
        """
        for column_index in range(self.columns):
            self._find_best_split(column_index)

        # if the score did not improve then we have reached the leaf
        if self.score == float('inf'): 
            return

        index = self.independent[self.best_split_column] <= self.best_split_value
        independent_left, dependent_left = self.independent[index], self.dependent[index]
        independent_right, dependent_right = self.independent[~index], self.dependent[~index]

        self.left_child = DecisionRegressionTree(independent_left, dependent_left, leaf_min_samples=self.leaf_min_samples)
        self.right_child = DecisionRegressionTree(independent_right, dependent_right, leaf_min_samples=self.leaf_min_samples)

    def _find_best_split(self, column_index):
        """
        Finds the best split on the feature such that total RMSE is minimized.

        Parameters
        ----------
        column_index: The index of a column that we will try to split.
        """
        # take feature and target variable
        column = self.independent.values[:, column_index]
        target = self.dependent

        # get index that would sort the feature by value
        # >>> x = np.array([5, 3, 4])
        # >>> index = np.argsort(x) # [1, 2, 0] 
        # >>> x[index] # [3, 4, 5] 
        # sort both feature and target according to index
        index = np.argsort(column)
        x, y = column[index], np.array(target[index])

        # at the beginning, all data points are to the right of the optimal value
        # our goal is to find that optimal value
        # since we sorted the data, we can now find optimal value in one pass through it
        right_count, right_sum, right_sum_squared = self.rows, y.sum(), (y**2).sum()
        left_count, left_sum, left_sum_squared = 0, float(0), float(0)

        for _ in range(self.rows - self.leaf_min_samples):
            left_count, right_count = left_count + 1, right_count - 1
            left_sum, right_sum = left_sum + y[_], right_sum - y[_]
            left_sum_squared, right_sum_squared = left_sum_squared + y[_]**2, right_sum_squared - y[_]**2

            # this line is very important
            # all data with the same feature value should be grouped together or if the leaf does not have enough data
            # then we need to continue
            if x[_] == x[_ + 1] or _ < self.leaf_min_samples - 1:
                continue

            # calculate the standard deviation of values from the left and right side of the currently assumed optimal value 
            # finally, to get the score take a weighted average of standard deviations
            # everything from the left side of currently assumed will predict the mean of all values from the left. 
            # The same holds for the right side. The total error is sum((yi - ymean)^2) for both sides (this is more or less the same as std). 
            # We then take a weighted average of errors on both sides of the currently assumed value in order to get the score.  
            left_std = self._calculate_standard_deviation(left_count, left_sum, left_sum_squared)
            right_std = self._calculate_standard_deviation(right_count, right_sum, right_sum_squared)
            score = left_count*left_std + right_count*right_std

            if score < self.score:
                self.score = score
                self.best_split_column = self.independent.columns[column_index]
                self.best_split_column_index = column_index
                self.best_split_value = x[_]

    def _calculate_standard_deviation(self, n, sum_, sum_squared):
        """
        Calculates the standard deviation of a data set given total number of data points, the sum of data values, and the squared sum of data values.

        Parameters
        ----------
        n: Total number of data points.

        sum_: Total sum of data values.

        sum_squared: Total sum of squared data values.

        Return
        ------
        The standard deviation of a data set.
        """
        return np.sqrt((sum_squared / n) - (sum_ / n)**2)
