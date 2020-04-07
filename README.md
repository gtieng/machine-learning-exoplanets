# Using Machine Learning to Classify Exoplanets with 40 Features
While data cleaning and visualization prove to be valuable core skills in the world of data analysis, beyond them are even more powerful techniques and tools that unlock the potential of what data can be. Machine learning gives us the power to create statistical models to make predictions as well as classifications. In the following project, we will be using various tools in the `scikit-learn` library to analyze a dataframe consisting of 40 columns in order to classify masses in space as potential exoplanets with close to 90% accuracy.

## Cleaning the Data
As in manual data analysis, machine learning requires that the dataset with which you are working to be clean to reduce errors and inefficiencies. In the following code, we'll import our `exoplanets.csv` file and drop all columns and rows with null values to reduce potential loss of accuracy.

```
import pandas as pd
df = pd.read_csv("exoplanet_data.csv")

df = df.dropna(axis="columns", how="all")
df = df.dropna()
df.head()
```
![](https://github.com/gtieng/machine-learning-exoplanets/blob/master/1_dataframe.png)

## Creating Training and Testing Data
In this dataset, we will have 40 features from which our machine learning models can classify space masses as a `CANDIDATE` for an exoplanet, a `CONFIRMED` exoplanet, or a `FALSE POSITIVE` for an exoplanet, all of which are contained in the `"koi_disposition"` column. 

In the following code, we will need to separate the descriptive features (`X`) from the predicted value (`y`), and then separate them once again into training (for machine learning) and testing (for model validation) sets of data. If not specified, the default are set to 75% training/25% testing.

```
from sklearn.model_selection import train_test_split

y = df["koi_disposition"]
X = df.drop("koi_disposition", axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
```

## Authors
**Gerard Tieng** - Data Analyst and Social Media Marketer \ 
[http://www.twitter.com/gerardtieng](http://www.twitter.com/gerardtieng) \
[http://www.linkedin.com/in/gerardtieng](http://www.linkedin.com/in/gerardtieng)
