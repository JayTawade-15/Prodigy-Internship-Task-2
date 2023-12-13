import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the Titanic dataset
titanic_data = pd.read_csv('titanic.csv')

# Display the first few rows of the dataset
print(titanic_data.head())

# Check for missing values
print(titanic_data.isnull().sum())

# Handle missing values
# For simplicity, you can fill missing numerical values with the mean and missing categorical values with the mode.
titanic_data['Age'].fillna(titanic_data['Age'].mean(), inplace=True)
titanic_data['Embarked'].fillna(titanic_data['Embarked'].mode()[0], inplace=True)

# Explore relationships between variables
# Example: Relationship between Pclass and Survived using a bar plot
sns.barplot(x='Pclass', y='Survived', data=titanic_data)
plt.title('Survival Rate by Pclass')
plt.show()

# Example: Relationship between Age and Survived using a box plot
sns.boxplot(x='Survived', y='Age', data=titanic_data)
plt.title('Age Distribution by Survival')
plt.show()

# Example: Pair plot to visualize relationships between numerical variables
sns.pairplot(titanic_data[['Survived', 'Pclass', 'Age', 'SibSp', 'Parch', 'Fare']])
plt.show()

# Example: Correlation heatmap to identify correlations between numerical variables
correlation_matrix = titanic_data[['Survived', 'Pclass', 'Age', 'SibSp', 'Parch', 'Fare']].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap')
plt.show()
