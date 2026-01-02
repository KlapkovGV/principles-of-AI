## Dimensionality Reduction
- if the algorithm forces the model to use all variables, including those that are not useful. If you are using a parametric and linear model, these models will theoretically want to include all variables in the model;
- when we want to make the model simpler for any reason. For example, when you want to interpret the mathematical model more easily or when we want to use less computing power.
 
There are two main motivations for reducing the number of input variables in a dataset.

## Scaling Data
- variable scaling is the method used to standardize the ranges of variables in data;
- becuase the range of values for variables can vary significantly, it becomes a necessary step in data processing when using ML algorithms;
- for a model, not every dataset requires scaling. If we do not have variables with different value ranges, we do not neet to perform scaling.

In machine learning, if one variable has a range from 0 to 1 and another has a range from 0 to 1 000 000, the algorithm might incorrectly assume the larger numbers are more important. Scaling brings them to a similar scale (like 0 to 1 or -1 to 1) so the model can treat them fairly.

The two most common methods for scaling data in machine learning are Min Max Scaling (Normalization) and Standardization (Z-score Normalization).

**Min Max Scaling** method rescales the data into a fixed range, typically between 0 and 1. It is highly useful for algorithms that require a bounded range, such as Neural Networks or k-Nearest Neighbors (K-NN).

The formula: x' = (x - min(x)) / (max(x) - min(x)), where
- x is the original value;
- min(x) is the minimum value in the dataset;
- max(x) is the maximum value in the dataset;
- x' is the new value (ranging from 0 to 1).

note: if we have extreme outliers (like one person with a salary of $1 000 000 in a dataset where everyone else earns $50 000), Min Max scaling will "squash" all other data points into a tiny range near 0.

**Standardization (Z-score Scaling)** transforms data so that it has a mean (mu) of 0 and a standard deviation (sigma) of 1. This method is less sensitive to outliers and is often preferred for model that assume a Gaussian (normal) distribution, such as Linear Regression or Support Vector Machines (SVM).

The formula: z = (x - mu) / sigma, where
- mu is the mean of the feature;
- sigma is the standard deviation of the feature;
- z is the standardized score (often called the Z-score).

Comparison table

![comparison table](https://github.com/user-attachments/assets/d11280cf-1771-484a-9a2d-7d64a1c01a0c)

## Handling categorical data

Most machine learning algorithms cannot handle categorical variables as textual data. For example, the gender variable might be entered as "Female" and "Male". These categories need to be represented as numbers, "0" for female and "1" for male.

Before using categorical variables in analysis, we need to code them in a format the algorithm will understand, that is, we must convert those variables into numeric (numerical) values. This is called encoding.

the solution: transforming text labels into numeric values so the model can perform calculation.

Common methods:
- Label Encoding: Assigning a unique number to each category;
- One-Hot Encoding: Creating separate columns for each category with 0s and 1s.

In machine learning, choosing between Label Encoding and One-Hot Encoding is crucial because using the wrong one can accidentally trick model into seeing patterns that aren not actually there.

**Label Encoding**

This method assigns each unique category a whole number (0, 1, 2, etc.). 
- when to use: use this for ordinal data, where the categories have a natural rank or order (e.g., "junior"=0, "mid"=1, "senior"=2);
- the risk: if we use this for non-ordered data (like apple and orange), the model might think orange (1) is greater than or better than apple (0), which is mathematically incorrect.

**One-Hot Encoding**

This method creates a new column for every category and uses "1" to indicate the presence of that category and "0" for the absence. 

- when to use: use this for Nominal Data, where there is no inherent order (e.g., color: red, blue, green);
- the risk: if we have hundreds of categories (like city's name), One-Hot Encoding will create hundreds of new columns, making dataset very large and slow to process.

Which one should ne choose?
- for ordered categories (small, medium, large) recommended method is label encoding;
- for unordered categories (male, female) recommended method is one-hot encoding;
- target variable (the "y" you are predicting) recommended method is label encoding;
- high cardinality (too meny categories) recommended method is binary encoding.

If we apply one-hot encoding to this categorical variable, we get:

![onehot](https://github.com/user-attachments/assets/274edd31-84fc-4feb-bae3-36b7d72d0a7f)

Note: however, notice that the data dimension has increased (new columns was created)! This is a situation we usually want to avoid. Applying this method to variables with a high number of categories can lead to a decline in the performance of our learning algorithm due to the curse of dimensionallity.

**Dummy Encoding**

Let's say we have a categorical variable like the following:
```python
import pandas as pd

data = pd.DataFrame({'City': ['Delhi', 'Mumbai', 
                             'Hyderabad', 'Chennai', 
                             'Bangalore', 'Delhi', 
                             'Hyderabad']})
data
```

![output](https://github.com/user-attachments/assets/ebbc1e6b-14b2-4c55-af15-6669bbf9ca9f)

In this state, the data is still in "text" format and cannot be used by a ml model until it is converted into numbers.

Dummy encoding is a variation of One-Hot encoding where we drop one of the columns (e.g., if you have n categories, you only keep n-1 columns) to avoid a situation where variables are highly correlated with each other.

## Handling Missing Observations

Data we work with for learning purposes usually comes in the form of a dataset with pre-defined attributes.

In datasets you may encounter in real life, the values of some attributes may be missing. This generally happens when the dataset is handmade and the person working on it forgets to fill in some values, or when they never measured them at all. Of course, there can be many other different reasins for the occurrence of missing observations.

The approaches that can be used regarding missing observations depend on the type of problem. There is no single "best" way to deal with missing observations. It may be best to try several techniques, build several models, and choose the one that works.

Once we have completed building the model and started the process of obtaining predictions, if our observation is not complete, we must use the same data imputation technique that we used to complete the training data to fill in the existing missing observation or observations.

## Methods to Hadle Missing Observations

**Using constant values**

Replace the missing observation with a constant outside the fixed range of values, such as -999, -1, etc. Another technique is to replace the missing value with a value outside the range of values that an attribute can take. For example, if the normal range is [0, 1], we can set the missing value to 2 or -1.

The idea here is for learning algorithm to learn what the best course of action is when the attribute has a value significantly different from the normal values. 

Alternatively, you can reolace the missing value with a value in the middle of the range. For example, if the range of an attribute is [-1, 1], we can set the missing value to 0. The idea here is that a value in the middle of the range will not significantly affect the predictions.

![missing values](https://github.com/user-attachments/assets/d144317c-1fa7-4aa6-997a-6df74de50abf)

**Using statistical measures**

Replace with mean and median values: this simple imputation method treats each variable (column) separately and relies on ignoring any mutual relationship with other variables.
- mean is suitable for continuous data that does not contain outliers;
- median is suitable for continuous data that contains outliers;
- cateforical attributes is for these, we can choose to fill missing values with the most common value.

Note that mean, median, and mode assignment reduces any correlation between the imputed variable(s). This is because we assume there is no relationship between the imputed variable and other measured variables. Therefore, this imputation method has some attractive features for univariate analysis but becomes problematic for multivariate analysis.

**Explaining how different machine learning algorithms react to missing data**

Let the algorithm handle the missing data: 
- Some algorithms (e.g., XGBoost) can take missing values into account and learn the best imputation values based on reducing training loss;
- some algorithms have a option to ignore missing observations (e.g., LightGBM - use_missing=False).

However, other algorithms will "panic" complain about the missing values, and produce an error (e.g., LinearRegression in Scikit-Learn). In this case, we will need to process the missing observations and clean them before feeding them into the algorithm. 

![missing values1](https://github.com/user-attachments/assets/d37c3302-25ac-4ea4-a5fb-43d4351b96ce)

**Technique for handling missing data using the K-Nearest Neighbors (KNN) algorithm**

By using the K-Nearest neighbors algorithm, we first find the K closest neighbors to the observation with missing data. Then, we impute the missing observations based on the non-missing observations in those neighbors, using the mode (for categorical variables) and/or the mean (for continuous variables).

The way how this works

KNN imputation looks at similar rows to fill the gap:
1. The algorithm looks at other attributes (columns) to find rows that are most similar to the one with the missing value;
2. It pick the K(e.g., 5) most similar rows;
3. It takes the average (mean) or most common value (mode) from these 5 neighbors to fill our missing spot.

## Data Splitting

A fundamental step in evaluating how well machine learning model actually works

The ultimate goal of any ml model is to learn from the examples we have in a way that can generalize to new examples the model has not yet seen.

The only way to know how well a model will generalize to new examples is to test the model on new examples.

We need to split our existing data into two groups: the training set and the test set.

We train our model using the training set and test it using the test data. The error rate on new examples is called the generalization error, and by evaluating the model on the test set, we obtain an estimate of this error. This value indicates how well the model will perform on examples it has never seen before.

**Practical guidelines for data splitting and common ratios** 

It is very common to reserve 80% of the data for training and 20% for testing. Sometimes we might also see this ratio as 75% - 25%. This depends on the size of the dataset. If we have a dataset with 10 million examples, setting aside 1% means the test set will contain 100 000 examples, which will likely be more than enough to get a good estimate of the generalization error.

However, when we have very little data, splitting it into training and test sets can leave us with a vey small test set. Let's say we only have 100 examples; if we do a simple 80% - 20% split, we will have 20 examples in our test set. This is not enough. In such a test set, we could get almost any performance purely by chance.

Note: the underlying distributions of both the training set and the test set must be similar.

train_test_split function ...

## Training Model

- training a model is the process of applying a machine learning algorithm to the training data;
- in machine learning, when we train a model, we adjust the model's parameters and hyperparamaters to increase its performance in solving a specific task;
- the training of a machine learning model is an iterative process. After the final model is obtained, predictions are made.

key concepts explained
- parameter are internal to the model and learned directly from the data;
- hyperparemeters are external settings you choose before training to control how the learning process behaves;
- iterative process. Models rarely get it right on the first try. The algorithm looks at the data, makes a guess, checks the error, and adjust itself multiple times until it performs well;
- generalization. The goal of the training is not just to memorize this data, but to learn patterns that work on the test set.

**More technical definition of what a parameter is**

A parameter is a configuration variable that is internal to the model and whose value is estimated directly from the data. It is generally not set manually by the researcher and generally saved as part of the learned model. 

Machine learning models are essentially mathematical functions that represent the relationships between different aspects of the data. In a simple linear regression model like Y = b_0 + b_1 * X, the values of b_0 and b_1 are parameters, and they are estimated from the data defined by the variables Y and X. 

**Hyperparameters**

A hyperparameter, also referred to a tuning parameter:
- is external to the model;
- cannot be estimated from the data;
- is generally specified by the researcher;
- is generally tuned using heuristic methods;
- is generally tuned for a specific modeling problem based on a certain prediction;
- is generally used in a process to help estimate model parameters.

  ![sum](https://github.com/user-attachments/assets/f8440f02-6499-4629-ab86-ad0e79ad7800)

Summary of above study path - machine learning pipeline
1. Preprocessing: Dimensionality Reduction, Scaling, and Encoding;
2. Cleaning: Handling missing Attributes through various imputation methods;
3. Splitting: Separating data into training and test sets to check generalization;
4. Training: An iterative process where the model learns (capture patterns) internal Parameters based on the Hyperparameters we set.

**Concrete example of a hyperparameter using a Dicision Tree algorithm**

What should the maximum allowed depth be for my Decision Tree algorithm?
- Depth 0: The root node of the tree (e.g., petal length =< 2.45);
- Depth 1: The first level of braches where data is split again;
- Depth 2: The subsequent level of the tree.

The maximum depth is a hyperparameter because we must decide how depp the tree is allowed to grow before the training starts.
- if the tree is too shallow, it may be too simple to learn the data;
- if it is too deep, it may overfit.

![decision tree](https://github.com/user-attachments/assets/db9eaeac-e01d-43e8-bf1f-140f5fafe5c6)

**How many neurons should we put in a neural network layer?**

![neuralnetwork](https://github.com/user-attachments/assets/43a44a23-e45e-48a3-b245-0d2a34df8339)

The number of neurons in each layer and the number of hidden layers themselves are hyperparameters. As the researcher, you must specify these external configurations before training starts. The model will then use these settings to estimate its internal parameters.

**What should we set our learning rate to for the gradient descent algorithm?**

The learning rate is a hyperparameter because it is an external setting you choose before training to control how quickly the model updates its internal parameters.

![learningrate](https://github.com/user-attachments/assets/df253992-886c-49e7-b87b-ffabff759ef9)

The three scenarios shown:
1. too low:
   - a small learning rate requires many updates before reaching the minimum point. The training process will be very slow and may get stuck before finding the optimal solution.
2. just rigth:
   - the optimal learning rate swiftly reaches the minimum point. Efficient training that finds the best parameters in a reasonable amount of time.
3. too high:
   - too large of a learning rate causes drastic update which lead to divergent behaviors. The model overshoots the minimum and bounces back and forh, failing to ever settle on an answer.
  
![learningrate1](https://github.com/user-attachments/assets/fb55b64c-1c8f-4d2e-88a3-ab33be3e6002)

## Hyperparameter Tuning and Model Selection 

Evaluating a model is easy. We simply use a test dataset. However, let's assume we are hesitating between two types of models. How can we decide between them? One ption is to train both models with the training data you have and compare how well they can generalize.

When we have two different algorithms (for example, a Dicision Tree vs. a Neural Network), we need a fair way to see which one performs better on real-world data. As mentioned earlier, the test set acts as the final exam. If model A gets 90% accuracy on the test set and model B gets 85%, model A is generelly considered the better choice for generalization.

**However, how do we choose the best hyperparameters for a model?**

In this situation, we set aside a portion of our existing training data as a validation set; we then evaluate different candidate models on this dataset and choose the best one.

Understanding the three-way split
- training set is used to help the model estimate its internal paremeters (like weight);
- validation set is used specifically to tune hyperparameters (like learning rate or tree depth);
- test set is used only once at the end to estimate the final generalization error.

In python, we can perform the training/test set split using the train_test_split function within the model_selection module of the Scikit-Learn library.

If the validation set remains too small, model evaluations mill not be accurate. Consequently, we might select an inadequate model. Conversely, if we choose a very large validation set, then the ramaining training set will be much smaller than the full training set.

## k-Fold Cross-Validation

