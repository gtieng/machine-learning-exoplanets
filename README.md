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

## Pre-Processing Using Scalers
For best optimization, it is best for feature values to be normalized when working with machine learning algorithms. If you glance at our dataset, you'll notice that feature values can range from small negative values to larger, place digits. In the following step, we'll pre-process our data before fitting into the model. The `MinMaxScaler()` is one such tool that transforms all data to land within the range of 0 and 1.

```
from sklearn.preprocessing import MinMaxScaler

X_scaler = MinMaxScaler().fit(X_train)
X_train_scaled = X_scaler.transform(X_train)
X_test_scaled = X_scaler.transform(X_test)
```
## Training the First Model
Now that our data has been split and pre-processed, we can now fit it into a model. For the first instance, we'll select Logistic Regression. This is one of the more simpler classification models and should give us a good idea about the feasability to fit a model to our data. Here is the code below:

```
from sklearn.linear_model import LogisticRegression

model1 = LogisticRegression()
model1.fit(X_train_scaled, y_train)

print(model1.score(X_train_scaled, y_train))
0.842456608811749

print(model1.score(X_test_scaled, y_test))
0.8352402745995423
```

## Hyperparameter Tuning the First Model
With 83-84% accuracy, we can say that our model is fairly accurate but there's still room for improvement. Taking into consideration how many parameters there are to adjust in a standard Logistic Regression model (see below), we can play around with a few.

```
LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
                   intercept_scaling=1, l1_ratio=None, max_iter=100,
                   multi_class='warn', n_jobs=None, penalty='l2',
                   random_state=None, solver='warn', tol=0.0001, verbose=0,
                   warm_start=False)
```

Using a tool like `GridSearchCV`, we are able to run our model multiple times, mixing-and-matching parameters until we settle on set that produces the best results. In this particular case, parameters are passed as a dictionary. We'll adjust `C` and `penalty` to see what happens.

```
from sklearn.model_selection import GridSearchCV

parameters1 = {"C": [100, 10, 1, 0.1, 0.01],
             "penalty": ["l1", "l2"]}

grid1 = GridSearchCV(model1, parameters1)
grid1.fit(X_train_scaled, y_train)
```

Here, we can print the winning combination which gave us an additional 3% improvement.
```
print(grid1.best_params_)
{'C': 10, 'penalty': 'l1'}

print(grid1.best_score_)
0.8735456799542247
```


## Authors
**Gerard Tieng** - Data Analyst and Social Media Marketer \ 
[http://www.twitter.com/gerardtieng](http://www.twitter.com/gerardtieng) \
[http://www.linkedin.com/in/gerardtieng](http://www.linkedin.com/in/gerardtieng)
