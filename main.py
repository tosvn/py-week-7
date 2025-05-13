import pandas as pd

# Load the dataset (using Iris dataset from sklearn as an example)
from sklearn.datasets import load_iris

# Load dataset
data = load_iris()
df = pd.DataFrame(data=data.data, columns=data.feature_names)

# Display the first few rows of the dataset
print(df.head())

# Explore the structure of the dataset
print(df.info())

# Check for missing values
print(df.isnull().sum())

# Clean the dataset (for simplicity, fill missing values with the mean of the column)
df.fillna(df.mean(), inplace=True)

# Check if missing values have been handled
print(df.isnull().sum())

# Compute basic statistics of numerical columns
print(df.describe())

# For groupings, let's assume we have a categorical column (e.g., target from the Iris dataset)
# Adding target values to the DataFrame for demonstration
df['species'] = data.target

# Perform grouping by 'species' and compute the mean of the numerical columns
grouped = df.groupby('species').mean()
print(grouped)

# Any interesting findings can be noted here
# Example: The mean sepal length varies slightly between species

import matplotlib.pyplot as plt
import seaborn as sns

# 1. Line chart showing trends over time (for demonstration, we'll create a time-series-like plot)
# For the sake of demonstration, let's simulate a time series of average petal lengths for each species
# In real use, this would be a time-based dataset.

plt.figure(figsize=(10, 6))
sns.lineplot(x=df['species'], y=df['petal length (cm)'])
plt.title('Trends of Petal Length by Species')
plt.xlabel('Species')
plt.ylabel('Petal Length (cm)')
plt.show()

# 2. Bar chart comparing numerical values across categories (e.g., average petal length per species)
plt.figure(figsize=(10, 6))
sns.barplot(x='species', y='petal length (cm)', data=df)
plt.title('Average Petal Length per Species')
plt.xlabel('Species')
plt.ylabel('Average Petal Length (cm)')
plt.show()

# 3. Histogram of a numerical column (e.g., distribution of sepal length)
plt.figure(figsize=(10, 6))
sns.histplot(df['sepal length (cm)'], kde=True, bins=15)
plt.title('Distribution of Sepal Length')
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Frequency')
plt.show()

# 4. Scatter plot to visualize the relationship between two numerical columns (e.g., sepal length vs. petal length)
plt.figure(figsize=(10, 6))
sns.scatterplot(x='sepal length (cm)', y='petal length (cm)', data=df)
plt.title('Sepal Length vs Petal Length')
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Petal Length (cm)')
plt.show()

try:
    # Try loading a CSV dataset if not using Iris
    df = pd.read_csv('your_dataset.csv')
except FileNotFoundError:
    print("Error: The dataset file was not found.")
except pd.errors.EmptyDataError:
    print("Error: The dataset file is empty.")
except Exception as e:
    print(f"An unexpected error occurred: {e}")

# Conclusion & Findings
# The mean values of numerical features (like petal length) may vary slightly across species.

# Some species tend to have longer or shorter petals or sepals.

# Based on the visualizations, you may notice clusters in the scatter plot, which could indicate relationships between the features.
