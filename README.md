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

## Authors
**Gerard Tieng** - Data Analyst and Social Media Marketer \ 
[http://www.twitter.com/gerardtieng](http://www.twitter.com/gerardtieng) \
[http://www.linkedin.com/in/gerardtieng](http://www.linkedin.com/in/gerardtieng)
