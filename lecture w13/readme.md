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

