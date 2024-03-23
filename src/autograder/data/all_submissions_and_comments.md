# Student 01
```python
# # Generating Synthetic data for Exploring Customer Chrun 

# ## Introduction
# Customer churn, the loss of customers, poses a substantial challenge to businesses, leading to revenue loss and impeding growth. Addressing this issue proactively requires the development of predictive models capable of identifying customers at risk of churning at a local store. 
# 
# In this notebook,  I am trying to build synthetic data for predicting customer churn. The dataset used for modeling consists of two key input features: monthly spend and the duration of active shopping months. By leveraging these features, I seek to classify customers as either likely to churn(1) or likely to remain engaged with the business(0).

# ## Step 1: Import the libraries we will use in this notebook

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# set this to ensure the results are repeatable. 
np.random.seed(1)

# ## Step 2: Generate Synthetic Data

# Define number of samples
n_samples = 100

# Generate shopping activity in months
shopping_activity_months = np.random.normal(0,60, size=n_samples) 
    
# Generate monthly spend with a normal distribution
mean_normal = 0
std_normal = 10000
monthly_spend = np.random.normal(mean_normal, std_normal, size=n_samples)

# Ensure spend and shopping_activity_months cannot be negative
monthly_spend = np.abs(monthly_spend)
shopping_activity_months = np.abs(shopping_activity_months)

# Generate churn labels based on the condition
# If monthly spend or tenure is below a specific threshold, churn is 1; otherwise, churn is 0
churn = np.where((monthly_spend < 5000) | (shopping_activity_months < 5), 1, 0)

# Create DataFrame
df = pd.DataFrame({'Shopping Activity (months)': shopping_activity_months,
                   'Monthly Spend': monthly_spend,
                   'Churn': churn})
df.head()


# ## Step 3: Explore the given data

# Plot the scatter plot
plt.figure(figsize=(8, 6))
plt.scatter(df['Shopping Activity (months)'], df['Monthly Spend'], c=df['Churn'], cmap='cool', edgecolor='black')
plt.title('Shopping Activity vs. Monthly Spend (Colored by Churn)')
plt.xlabel('Shopping Activity (months)')
plt.ylabel('Monthly Spend')
plt.colorbar(label='Churn')  # Add colorbar to show churn status
plt.grid(True)
plt.show()

# ## Step 4: Save the Data

df.to_csv(r'/Users/poorna/Downloads/customer_churn.csv', index=False)

# In this notebook, I aim to construct a classification model using Support Vector Classifier (SVC) with linear, radial basis function (RBF), and polynomial kernels. 
# 
# Additionally, I will perform hyperparameter tuning to enhance the predictive performance in predicting customer churn. The dataset utilized in this analysis comprises synthetic data, featuring two primary input features: monthly spend and the duration of active shopping months. 
# By utilizing these features, the objective is to categorize customers into two groups: likely to churn (labeled as 1) or likely to remain engaged with the business (labeled as 0).

# # Model Training and Testing on Customer Chrun dataset

# Our investigation will encompass the following models:
# - Support Vector Machine with Linear kernel (SVC Linear)
# - Support Vector Machine with Polynomial kernel (SVC Poly)
# - Support Vector Machine with Radial Basis Function kernel (SVC RBF)

# ## 1. Import the libraries we will use in this notebook

import pandas as pd
from sklearn.svm import SVC
from matplotlib import pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV

np.random.seed(1)

# ## 2. Load data

# Load data which is cleaned and preprocessed

df = pd.read_csv(r'/Users/poorna/Downloads/customer_churn.csv') 
df.head()

# ### Perform Train-Test split

# Use sklearn to split df into a training set and a test set
X = df[['Shopping Activity (months)', 'Monthly Spend']]
y = df['Churn']


# test_size=0.3: 30% of the data will be used for testing, 70% for training
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

# ## 3. Model the data

# First, let's create a dataframe to load the model performance metrics into.

performance = pd.DataFrame({"model": [], "Accuracy": [], "Precision": [], "Recall": [], "F1": [], "Parameters": []})

# ### 3.1 Fit a SVM classification model using linear kernal

# defining parameter range 
param_grid = {'C': [0.01, 0.1, 0.5, 1, 5, 10, 50, 100],  
              'kernel': ['linear']}

# Using f1 for scoring
grid = GridSearchCV(SVC(), param_grid, scoring='f1', refit = True, verbose = 3, n_jobs=-1) 
  
# fitting the model for grid search 
_ = grid.fit(X_train, y_train)

# print best parameter after tuning 
print(grid.best_params_) 
  
# print how our model looks after hyper-parameter tuning 
print(grid.best_estimator_)

y_pred = grid.predict(X_test) 

recall = recall_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

performance = pd.concat([performance, pd.DataFrame({"model": ["SVM Linear"], "Accuracy": [accuracy], "Precision": [precision], "Recall": [recall], "F1": [f1], "Parameters": [grid.best_params_]})])


performance

# ## SVM Linear - Interpretation of Performance

# Accuracy: The model achieved an accuracy of 93.33%. This indicates that 93.33% of the predictions made by the model were correct.
# 
# Precision: The precision of the model is 87.5%. Precision represents the proportion of true positive predictions out of all positive predictions made by the model. In this case, it means that when the model predicts a customer will churn, it is correct 87.5% of the time.
# 
# Recall: The recall of the model is 100%. Recall, also known as sensitivity, represents the proportion of actual positive cases that were correctly identified by the model. A recall of 100% indicates that the model successfully identified all customers who actually churned.
# 
# F1 Score: The F1 score, which is the harmonic mean of precision and recall, is 93.33%. It provides a balance between precision and recall, with higher values indicating better performance.
# 
# Parameters: The parameters of the model are {'C': 0.1, 'kernel': 'linear'}. 'C' is the regularization parameter, which controls the trade-off between maximizing the margin and minimizing the classification error. A smaller value of C indicates stronger regularization.

# ### 3.2 Fit a SVM classification model using rbf kernal

# defining parameter range 
param_grid = {'C': [0.1, 1, 10, 100],  
              'gamma': [1, 0.1, 0.01, 0.001, 0.0001], 
              'kernel': ['rbf']}
  
grid = GridSearchCV(SVC(), param_grid, scoring='f1', refit = True, verbose = 3, n_jobs=-1) 
  
# fitting the model for grid search 
_ = grid.fit(X_train, y_train)

# print best parameter after tuning 
print(grid.best_params_) 
  
# print how our model looks after hyper-parameter tuning 
print(grid.best_estimator_)

y_pred = grid.predict(X_test) 

recall = recall_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

performance = pd.concat([performance, pd.DataFrame({"model": ["SVM rbf"], "Accuracy": [accuracy], "Precision": [precision], "Recall": [recall], "F1": [f1], "Parameters": [grid.best_params_]})])

performance

# ## SVM RBF - Interpretation of Performance

# Accuracy: The model achieved an accuracy of 86.67%. This indicates that 86.67% of the predictions made by the model were correct.
# 
# Precision: The precision of the model is 81.25%. Precision represents the proportion of true positive predictions out of all positive predictions made by the model. In this case, it means that when the model predicts a customer will churn, it is correct 81.25% of the time.
# 
# Recall: The recall of the model is 92.86%. Recall, also known as sensitivity, represents the proportion of actual positive cases that were correctly identified by the model. A recall of 92.86% indicates that the model successfully identified 92.86% of customers who actually churned.
# 
# F1 Score: The F1 score, which is the harmonic mean of precision and recall, is 86.67%. It provides a balance between precision and recall, with higher values indicating better performance.
# 
# Parameters: The parameters of the model are {'C': 0.01, 'coef0': 5, 'kernel': 'poly'}. 'C' is the regularization parameter, controlling the trade-off between maximizing the margin and minimizing the classification error. A smaller value of C indicates stronger regularization.

# ### 3.3 Fit a SVM classification model using polynomial kernal

# defining parameter range 
param_grid = {'C': [0.01, 0.1, 0.5, 1, 5, 10, 50, 100],  
              'coef0': [0.01, 0.1, 0.5, 1, 5, 10, 50, 100],
              'kernel': ['poly']}
  
grid = GridSearchCV(SVC(), param_grid, scoring='f1', refit = True, verbose = 3, n_jobs=-1) 
  
# fitting the model for grid search 
_ = grid.fit(X_train, y_train)

# print best parameter after tuning 
print(grid.best_params_) 
  
# print how our model looks after hyper-parameter tuning 
print(grid.best_estimator_)

y_pred = grid.predict(X_test) 

recall = recall_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)


performance = pd.concat([performance, pd.DataFrame({"model": ["SVM Poly"], "Accuracy": [accuracy], "Precision": [precision], "Recall": [recall], "F1": [f1], "Parameters": [grid.best_params_]})])

performance

# ## SVM Poly - Interpretation of Performance

# Accuracy: The model achieved an accuracy of 80.00%. This indicates that 80.00% of the predictions made by the model were correct.
# 
# Precision: The precision of the model is 90.00%. Precision represents the proportion of true positive predictions out of all positive predictions made by the model. In this case, it means that when the model predicts a customer will churn, it is correct 90.00% of the time.
# 
# Recall: The recall of the model is 64.29%. Recall, also known as sensitivity, represents the proportion of actual positive cases that were correctly identified by the model. A recall of 64.29% indicates that the model successfully identified 64.29% of customers who actually churned.
# 
# F1 Score: The F1 score, which is the harmonic mean of precision and recall, is 75.00%. It provides a balance between precision and recall, with higher values indicating better performance.
# 
# Parameters: The parameters of the model are {'C': 10, 'gamma': 0.0001, 'kernel': 'rbf'}. 'C' is the regularization parameter, controlling the trade-off between maximizing the margin and minimizing the classification error. A smaller value of C indicates stronger regularization. 'Gamma' is the parameter of the RBF kernel, which defines the influence of a single training example. A smaller value of gamma indicates a larger similarity radius.

# ## 4.0 Discussion

performance.sort_values(by="F1", ascending=False)

# I have considered F1 as my choice of comparision metric because both Precision and recall capture different aspects of model performance. Precision measures the accuracy of positive predictions, while recall measures the ability to capture all positive instances. F1 score combines these two metrics into a single value, providing a balanced assessment of the model's ability to make accurate predictions while capturing as many true positives as possible. This is especially important in churn prediction, where both false positives (misidentifying loyal customers as churners) and false negatives (missing actual churners) have significant implications for business outcomes.

# Based on the F1 scores, which provide a balanced assessment of precision and recall, the SVM Linear model emerges as the most effective in predicting customer churn. The high F1 score of the SVM Linear model indicates a strong ability to make accurate predictions while effectively capturing true positive instances of churn. This suggests that the SVM Linear model strikes a good balance between minimizing false positives (misidentifying loyal customers as churners) and false negatives (missing actual churners), making it well-suited for practical applications in customer churn prediction.

# Overall, while all three models demonstrate the potential to predict customer churn, the SVM Linear model stands out as the most effective choice based on its higher F1 score. 


```
**Feedback:**
Data Generation (30%) - 0.45/0.45
• Creativity and relevance of the chosen scenario - 0.15/0.15
• Correct implementation of data generation techniques - 0.15/0.15
• Clarity and structure of the data (including labeling and documentation) - 0.15/0.15

Model Implementation and Tuning (40%) - 0.6/0.6
• Correct implementation of SVC/SVR models with the three specified kernels - 0.2/0.2
• Effective use of hyper-parameter tuning, with a clear rationale for chosen parameters - 0.2/0.2
• Documentation of model performance metrics, with comparisons to justify the best-performing model - 0.2/0.2

Analysis and Discussion (30%)- 0.45/0.45
• Clarity and depth of the introduction and explanation of the chosen problem and dataset - 0.225/0.225
• Critical analysis of model results, including insights into performance variations across different kernels - 0.225/0.225

---

# Student 02
```python
# #### The data we are generating is related to the customer profiling for an online platform. By analyzing 'Items Purchased' and 'Time Spent online,' the company aims to identify high-value customers. A high-value customer, denoted by 'Value Customer' as 1, signifies individuals with substantial purchases or extended online engagement. This classification aids the business in tailoring strategies to retain and enhance relationships with high-value clientele, potentially optimizing marketing efforts and improving overall customer satisfaction and loyalty.

# #### Importing libraries, setting random seed, and preparing for Support Vector Machine (SVM) classification with evaluation metrics, data splitting, hyperparameter tuning, and feature scaling.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, f1_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler

np.random.seed(1)

# #### Generating a sample DataFrame with 30 rows, 'Items Purchased' ranging from 0 to 10, and 'Time Spent online' ranging from 1 to 30.

sample_size = 30
Item_min_val = 0
Item_max_val = 10
Time_min_val = 1
Time_max_val = 30

df = pd.DataFrame({'Items Purchased': np.linspace(Item_min_val,Item_max_val,sample_size).astype(int),'Time Spent online': np.linspace(Time_min_val,Time_max_val,sample_size)})
df

df['Value Customer'] = ((df['Items Purchased'] > 5) |  (df['Time Spent online'] > 10)).astype(int)
df

# #### Adding random noise to 'Time Spent online' column in DataFrame.

df['Time Spent online'] = df['Time Spent online'] + np.random.uniform(-30,30, sample_size)

# #### Scatter plot visualizing the relationship between 'Time Spent Online' and 'Items Purchased' with color-coded points based on 'Value Customer' status.

plt.figure(figsize=(10, 6))
plt.scatter(df['Time Spent online'] , df['Items Purchased'], c=df['Value Customer'], cmap='cool', edgecolors='k', alpha=0.8, s=100)
plt.colorbar(label='Value Customer')
plt.xlabel('Time Spent Online')
plt.ylabel('Items Purchased')
plt.title('Items Purchased vs Time Spent Online')
plt.grid(True)
plt.show()

# #### Saving DataFrame 'df' to a CSV file named 'value_customer.csv' without including the index.

df.to_csv('value_customer.csv', index=False)

# ### Train and Split Data

# Use sklearn to split df into a training set and a test set

X = df.iloc[:,:2]
y = df['Value Customer']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)


# ### Modeling the Data

performance = pd.DataFrame({"model": [], "Accuracy": [], "Precision": [], "Recall": [], "F1": [], "F2": [], "Parameters": []})

# generate a fbeta 2 scorer
from sklearn.metrics import make_scorer
from sklearn.metrics import fbeta_score
f2_scorer = make_scorer(fbeta_score, beta=2)

# ### 3.1 Fit a SVM classification model using linear kernal

# defining parameter range 
param_grid = {'C': [0.01, 0.1, 0.5, 1, 5, 10, 50, 100],  
              'kernel': ['linear']}
  

grid = GridSearchCV(SVC(), param_grid, scoring=f2_scorer, refit = True, verbose = 3, n_jobs=-1) 
  
# fitting the model for grid search 
_ = grid.fit(X_train, y_train)

# #### Print best parameters and the tuned SVM model, then predict on test set and calculate performance metrics for evaluation.

# print best parameter after tuning 
print(grid.best_params_) 
  
# print how our model looks after hyper-parameter tuning 
print(grid.best_estimator_)

y_pred = grid.predict(X_test) 

recall = recall_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
f2 = fbeta_score(y_test, y_pred, beta=2)

performance = pd.concat([performance, pd.DataFrame({"model": ["SVM Linear"], "Accuracy": [accuracy], "Precision": [precision], "Recall": [recall], "F1": [f1], "F2": [f2], "Parameters": [grid.best_params_]})])


# ### 3.2 Fit a SVM classification model using rbf kernal

# defining parameter range 
param_grid = {'C': [0.1, 1, 10, 100],  
              'gamma': [1, 0.1, 0.01, 0.001, 0.0001], 
              'kernel': ['rbf']}
  
grid = GridSearchCV(SVC(), param_grid, scoring=f2_scorer, refit = True, verbose = 3, n_jobs=-1) 
  
_ = grid.fit(X_train, y_train)

# #### Printing best parameters and the tuned model, then evaluating and storing performance metrics for an SVM (RBF kernel) after hyperparameter tuning.

# print best parameter after tuning 
print(grid.best_params_) 
  
# print how our model looks after hyper-parameter tuning 
print(grid.best_estimator_)

y_pred = grid.predict(X_test) 

recall = recall_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
f2 = fbeta_score(y_test, y_pred, beta=2)

performance = pd.concat([performance, pd.DataFrame({"model": ["SVM rbf"], "Accuracy": [accuracy], "Precision": [precision], "Recall": [recall], "F1": [f1], "F2": [f2],"Parameters": [grid.best_params_]})])

# ### 3.3 Fit a SVM classification model using polynomial kernal

# defining parameter range 
param_grid = {'C': [0.01, 0.1, 0.5, 1, 5, 10, 50, 100],  
              'coef0': [0.01, 0.1, 0.5, 1, 5, 10, 50, 100],
              'kernel': ['poly']}
  
grid = GridSearchCV(SVC(), param_grid, scoring=f2_scorer, refit = True, verbose = 3, n_jobs=-1) 
  
# fitting the model for grid search 
_ = grid.fit(X_train, y_train)

# #### Print best parameters and the tuned model, then evaluate its performance metrics and append results to a DataFrame.

# print best parameter after tuning 
print(grid.best_params_) 
  
# print how our model looks after hyper-parameter tuning 
print(grid.best_estimator_)

y_pred = grid.predict(X_test) 

recall = recall_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
f2 = fbeta_score(y_test, y_pred, beta=2)

performance = pd.concat([performance, pd.DataFrame({"model": ["SVM Poly"], "Accuracy": [accuracy], "Precision": [precision], "Recall": [recall], "F1": [f1], "F2": [f2], "Parameters": [grid.best_params_]})])

# ## 4.0 Summary

# #### Sorting the 'performance' DataFrame by the column "F2" in descending order.

performance.sort_values(by="F2", ascending=False)

# #### Three Support Vector Machine (SVM) models were evaluated for performance, and the results show that the SVM Poly model performs best overall, with an accuracy of 88.89% and well-balanced precision, recall, and F1 score. With the specified parameter settings {'C': 0.01, 'coef0': 50}, it makes use of a polynomial kernel.


```
**Feedback:**
Data Generation (30%) - 0.4/0.45
• Creativity and relevance of the chosen scenario - 0.15/0.15
• Correct implementation of data generation techniques - 0.15/0.15
• Clarity and structure of the data (including labeling and documentation) - 0.1/0.15

Model Implementation and Tuning (40%) - 0.6/0.6
• Correct implementation of SVC/SVR models with the three specified kernels - 0.2/0.2
• Effective use of hyper-parameter tuning, with a clear rationale for chosen parameters - 0.2/0.2
• Documentation of model performance metrics, with comparisons to justify the best-performing model - 0.2/0.2

Analysis and Discussion (30%)- 0.45/0.45
• Clarity and depth of the introduction and explanation of the chosen problem and dataset - 0.225/0.225
• Critical analysis of model results, including insights into performance variations across different kernels - 0.225/0.225

Need to create a dedicated notebook for data generation as this keeps your code more organized. So, 0.05 marks were deducted

---

# Student 03
```python
# ### Introduction
# 
# Here for this assignment I'm going to use population density and GDP Per capita as features to predict the Carbon emissions which is a major environmental problem and almost every company is trying to minimize their carbon emissions to achieve sustainability.
# 
# Population Density: Population density refers to the number of people living per unit area,Population density is a significant factor influencing carbon emissions. Areas with higher population densities might have more transportation, industrial activities, and energy consumption, leading to higher carbon emissions.
# Unit: Persons per square kilometer.
# 
# GDP per Capita: GDP per capita can reflect the economic activity and development level of a region. Generally, higher GDP per capita might correlate with higher industrialization, energy consumption, and potentially higher carbon emissions.GDP per capita is a measure of the economic output of a country or region per person. It is calculated by dividing the gross domestic product (GDP) by the total population.
# Unit: Currency (usually in USD for international comparisons).
# 
# Carbon Emissions: Carbon emissions represent the release of carbon dioxide (CO2) and other greenhouse gases into the atmosphere, primarily as a result of human activities such as burning fossil fuels, deforestation, and industrial processes.
# Unit: Metric tons of CO2 equivalent (or any other unit used for measuring greenhouse gas emissions).

# First step would be importing required packages. Let's do that.

# ### 1. Import packages 

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

np.random.seed(1) # To ensure the results are repeatable.

# ### 2. Define our hidden relationship Model

# First, let's set a sample size.

num_samples = 2000

b0 = 100 # y intercept
b1 = 0.5 # slope for population density
b2 = 0.02 # slope for GDP Per capita

# ### 3. Create input and output data using the model

# Generate synthetic data
population_density = np.random.uniform(50, 500, num_samples)  # range for population density (persons/km^2)
gdp_per_capita = np.random.uniform(5000, 50000, num_samples)  # range for GDP per capita (in USD)
carbon_emissions = b0 + b1 * population_density + b2 * gdp_per_capita

%matplotlib widget 

fig = plt.figure()
ax = fig.add_subplot(projection='3d')

ax.scatter(population_density, gdp_per_capita, carbon_emissions)

ax.set_xlabel('population_density')
ax.set_ylabel('gdp_per_capita')
ax.set_zlabel('carbon_emissions')
plt.tight_layout()
plt.show()

# ### 4. Adding noise to the data

# Since this is a synthesized data, to make it look more like random and real data I'm going to add some noise which will hide the linear relationship of the model. Let's get into it

e_mean = 0
e_stdev = 40
e = np.round(np.random.normal(e_mean, e_stdev, num_samples), 2) # round to two decimal places
carbon_emissions = carbon_emissions + e

# Now, let's look at the data again to visualize our changes after adding noise.

# this is a notebook 'MAGIC' that will allow for creation of interactive plot
%matplotlib widget 

fig = plt.figure()
ax = fig.add_subplot(projection='3d')

ax.scatter(population_density,gdp_per_capita, carbon_emissions)

ax.set_xlabel('population_density')
ax.set_ylabel('gdp_per_capita')
ax.set_zlabel('carbon_emissions')
plt.tight_layout()
plt.show()

# ### 5. Create a Dataframe.
# 
# Since we want to examine SVC/SVR performance with 3 different types of kernels with this data. We want to store this data in a dataframe and store it in a csv file to use it for further analysis.

# Create DataFrame
df = pd.DataFrame({
    'population_density': population_density,
    'gdp_per_capita': gdp_per_capita,
    'carbon_emissions': carbon_emissions
})

df.head(5)

# ### 6. Store the dataframe in a csv file

df.to_csv('carbon_emissions_dataset.csv', index=False)

# Since we are done with the data generation it's time to load that into our jupyter environment and examine it's performance after hyper parameter tuning with 3 different kernels. Let's look in to it. In this case first step would be importing packages.

# ### 1. Import required packages.

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error,r2_score

np.random.seed(1) #set the seed to ensure same results on repitition

# ### 2. Load the data.

# Next step would be to load the data into our jupyter notebook.

df = pd.read_csv(r"C:\Users\Admin\Downloads\ISM6251\carbon_emissions_dataset.csv")
df.head(5)

# ### 3. Train - test Split.

# Now that we had a look into our dataset. Let's perform train and test splitfor model training and hyperparameter tuning. Here I'm using a 80-20 split ratio since we have a small dataset of 2000 observations. So that we'll be having major part of it to train our model.

# Split features and target variable

X = df[['population_density', 'gdp_per_capita']]
y = df['carbon_emissions']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ### 4. Standardization

# Models like SVM, KNN, Decision trees are extremely sensitive to scaling and stanradization is essential to enusre all columns are in the same range. Here in this dataset we have GDP and population density where gdp is very high in units compared to population density and population_density so I'm standardizing to ensure our results are accurate and dependable. I'm performing standaridization after train test split to prevent any data leakage as we will be exploring all the distribution of our features.

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ### 5. Hyperparameter tuning - GridSearchCV

# Define parameter grids for SVR with different kernels
param_grid_linear = {'C': [0.1, 1, 10, 100]}
param_grid_rbf = {'C': [0.1, 1, 10, 100], 'gamma': [0.1, 0.01, 0.001]}
param_grid_poly = {'C': [0.1, 1, 10, 100], 'degree': [2, 3, 4]}

# Perform grid search cross-validation for each SVR model
svr_linear = GridSearchCV(SVR(kernel='linear'), param_grid_linear, cv=5)
svr_rbf = GridSearchCV(SVR(kernel='rbf'), param_grid_rbf, cv=5)
svr_poly = GridSearchCV(SVR(kernel='poly'), param_grid_poly, cv=5)

# ### 5.1 Training the models

# Fit SVR models
svr_poly.fit(X_train_scaled, y_train)

svr_rbf.fit(X_train_scaled, y_train)

svr_linear.fit(X_train_scaled, y_train)

# ### 5.2. Testing the model

# Predictions
y_pred_linear = svr_linear.predict(X_test_scaled)
mse_linear = mean_squared_error(y_test, y_pred_linear)

y_pred_rbf = svr_rbf.predict(X_test_scaled)
mse_rbf = mean_squared_error(y_test, y_pred_rbf)

y_pred_poly = svr_poly.predict(X_test_scaled)
mse_poly = mean_squared_error(y_test, y_pred_poly)

print("Mean square error (Linear Kernel):", mse_linear)
print("Mean square error (RBF Kernel):", mse_rbf)
print("Mean square error (Polynomial Kernel):", mse_poly)

# Calculate R-squared for each SVR model
r2_linear = r2_score(y_test, y_pred_linear)
r2_rbf = r2_score(y_test, y_pred_rbf)
r2_poly = r2_score(y_test, y_pred_poly)

print("R-squared (Linear Kernel):", r2_linear)
print("R-squared (RBF Kernel):", r2_rbf)
print("R-squared (Polynomial Kernel):", r2_poly)

# ### 6. Discussion

# The lower the MSE, the better the model's performance. In this case, both the Linear Kernel and RBF Kernel SVR models have relatively similar MSE values, indicating comparable predictive performance. However, the Polynomial Kernel SVR model has significantly higher MSE compared to the other two, suggesting poorer predictive performance. This suggests that, for this particular dataset, the linear and RBF kernels are more suitable for capturing the underlying relationships between the features (population density, GDP per capita) and the target variable (carbon emissions). The polynomial kernel, on the other hand, might be overfitting the data or struggling to capture the underlying patterns effectively, leading to higher prediction errors. After comparing the R-squared values 
# 
# Linear Kernel: R-squared value of 0.9797. This indicates that approximately 97.97% of the variance in carbon emissions can be explained by the features (population density and GDP per capita) when using the linear kernel SVR model.
# RBF Kernel: R-squared value of 0.9799. Similarly, the RBF kernel SVR model also performs very well, with approximately 97.99% of the variance in carbon emissions explained by the features.
# Polynomial Kernel: R-squared value of 0.8292. The polynomial kernel SVR model has a lower R-squared value compared to the linear and RBF kernels, indicating that it explains less of the variance in carbon emissions. However, an R-squared value of 0.8292 still suggests a reasonable fit to the data.
# 
# Based on both the MSE and R-squared values, it seems that the linear and RBF kernel SVR models perform similarly well and outperform the polynomial kernel SVR model in terms of predictive accuracy and explanatory power for this particular dataset. This suggests that the linear and RBF kernels are more appropriate for capturing the underlying relationships between the features and the target variable. In conclusion, for this dataset, both the linear and RBF kernel SVR models provide highly accurate predictions of carbon emissions based on population density and GDP per capita, while the polynomial kernel SVR model shows slightly lower performance.


```
**Feedback:**
Data Generation (30%) - 0.45/0.45
• Creativity and relevance of the chosen scenario - 0.15/0.15
• Correct implementation of data generation techniques - 0.15/0.15
• Clarity and structure of the data (including labeling and documentation) - 0.15/0.15

Model Implementation and Tuning (40%) - 0.6/0.6
• Correct implementation of SVC/SVR models with the three specified kernels - 0.2/0.2
• Effective use of hyper-parameter tuning, with a clear rationale for chosen parameters - 0.2/0.2
• Documentation of model performance metrics, with comparisons to justify the best-performing model - 0.2/0.2

Analysis and Discussion (30%)- 0.35/0.45
• Clarity and depth of the introduction and explanation of the chosen problem and dataset - 0.125/0.225
• Critical analysis of model results, including insights into performance variations across different kernels - 0.225/0.225

Need to mention whether you choose regression or classification task in introduction or data generation notebook. So, 0.1 marks were deducted.

---

# Student 04
```python
# # Performance Analysis of SVC with Different Kernels: Linear, RBF, Polynomial

# ## Introduction
# 
# In this notebook, we generated synthetic data and then conducted a comparative analysis of Support Vector Classification (SVC) using three different kernels: Linear, RBF, and Polynomial. We assessed their respective performances to understand their efficacy in classifying the synthetic data.
# 
# The synthetic dataset we will be generating is Height and Weight as input variables and create a binary output variable indicating whether a person is an athlete or not.

# - Lets assume that, we are an Athlete Management agency. As an Athlete Management agency, our primary goal is to ensure that all potential athletes receive our agency's advertisement. While it's acceptable to send advertisements to non-athletes, it's crucial to avoid missing any potential athletes.
# 
# - We will use the SVM class to fit a model to the data. We will use GridSearchCV to find the best hyper-parameters for the model - and we will test rbf, linear and polynomial kernels. 
# 
# - **The scoring metric we will use is custom beta score, with a beta of 2** (which means we are more interested in recall than precision). 
# 
# - The reason for choosing this metric is that false negatives (missing potential athletes) are more costly than false positives (sending advertisements to non-athletes). Therefore, we prioritize recall to ensure that we capture as many athletes as possible. However, we cannot ignore precision entirely as we aim to maintain a reasonable level of accuracy in targeting potential athletes and avoid unnecessary advertisement costs.

# ## 1. Import the libraries that we expect to use

import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
from sklearn.metrics import fbeta_score

np.random.seed(1)

# ## 2. Data Generation

# Generating random values for Height and Weight. Assuming mean height of 175 cm with standard deviation of 20 cm and mean weight of 75 kg with standard deviation of 20 kg.

height = np.round(np.random.normal(loc=175, scale=20, size=500),2)
weight = np.round(np.random.normal(loc=75, scale=20, size=500),2)

# Generating binary output variable indicating whether the person is an athlete or not. Let's assume athletes are those with height greater than 180 cm and weight less than 80 kg

is_athlete = ((height > 180) & (weight < 80)).astype(int)

# Adding some randomness or noise to the data by flipping the labels

# Introducing randomness by flipping some of the labels randomly
# Let's flip 10% of the labels randomly
num_flips = int(0.1 * 500)
flip_indices = np.random.choice(500, num_flips, replace=False)
is_athlete[flip_indices] = 1 - is_athlete[flip_indices]

df = pd.DataFrame({'height': height, 'weight': weight, 'is_athlete': is_athlete})

df.head()

df['is_athlete'].value_counts()

# Use sklearn to split df into a training set and a test set
X = df[['height','weight']]
y = df['is_athlete']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# ## 3. Model the data

# First, let's create a dataframe to load the model performance metrics into.

performance = pd.DataFrame({"model": [], "Accuracy": [], "Precision": [], "Recall": [], "F1": [], "F2": [], "Parameters": []})

# create a fbeta 2 scorer
f2_scorer = make_scorer(fbeta_score, beta=2)

# ### 3.1 Fit a SVM classification model using linear kernal

# defining parameter range 
param_grid = {'C': [0.01, 0.1, 0.5, 1, 5, 10, 50, 100],  
              'kernel': ['linear']}
  
grid = GridSearchCV(SVC(), param_grid, scoring=f2_scorer, refit = True, verbose = 3, n_jobs=-1) 
  
# fitting the model for grid search 
_ = grid.fit(X_train, y_train)

# print best parameter after tuning 
print(grid.best_params_) 
  
# print how our model looks after hyper-parameter tuning 
print(grid.best_estimator_)

y_pred = grid.predict(X_test) 

recall = recall_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
f2 = fbeta_score(y_test, y_pred, beta=2)

performance = pd.concat([performance, pd.DataFrame({"model": ["SVM Linear"], "Accuracy": [accuracy], "Precision": [precision], "Recall": [recall], "F1": [f1], "F2": [f2], "Parameters": [grid.best_params_]})])


performance

# ### 3.2 Fit a SVM classification model using rbf kernal

# defining parameter range 
param_grid = {'C': [0.1, 1, 10, 100],  
              'gamma': [1, 0.1, 0.01, 0.001, 0.0001], 
              'kernel': ['rbf']}
  
grid = GridSearchCV(SVC(), param_grid, scoring=f2_scorer, refit = True, verbose = 3, n_jobs=-1) 
  
# fitting the model for grid search 
_ = grid.fit(X_train, y_train)

# print best parameter after tuning 
print(grid.best_params_) 
  
# print how our model looks after hyper-parameter tuning 
print(grid.best_estimator_)

y_pred = grid.predict(X_test) 

recall = recall_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
f2 = fbeta_score(y_test, y_pred, beta=2)

performance = pd.concat([performance, pd.DataFrame({"model": ["SVM rbf"], "Accuracy": [accuracy], "Precision": [precision], "Recall": [recall], "F1": [f1], "F2": [f2],"Parameters": [grid.best_params_]})])

performance

# ### 3.3 Fit a SVM classification model using polynomial kernal

# defining parameter range 
param_grid = {'C': [0.01, 0.1, 0.5, 1, 5, 10, 50, 100],  
              'coef0': [0.01, 0.1, 0.5, 1, 5, 10, 50, 100],
              'kernel': ['poly']}
  
grid = GridSearchCV(SVC(), param_grid, scoring=f2_scorer, refit = True, verbose = 3, n_jobs=-1) 
  
# fitting the model for grid search 
_ = grid.fit(X_train, y_train)

# print best parameter after tuning 
print(grid.best_params_) 
  
# print how our model looks after hyper-parameter tuning 
print(grid.best_estimator_)

y_pred = grid.predict(X_test) 

recall = recall_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
f2 = fbeta_score(y_test, y_pred, beta=2)

performance = pd.concat([performance, pd.DataFrame({"model": ["SVM Poly"], "Accuracy": [accuracy], "Precision": [precision], "Recall": [recall], "F1": [f1], "F2": [f2], "Parameters": [grid.best_params_]})])

performance

# ## 4.0 Summary

performance.sort_values(by="F2", ascending=False)

# - The SVM with the RBF kernel outperformed the other models across most metrics, indicating its effectiveness in capturing both precision and recall.
# - The polynomial kernel SVM also performed reasonably well but slightly lower than the RBF kernel SVM.
# - The linear kernel SVM achieved the lowest scores in terms of precision, recall, and F1/F2 scores, suggesting that it might not be as effective in capturing the complexities of the data compared to the non-linear kernels (RBF and polynomial).
# - Considering the priority of minimizing false negatives (missed potential athletes) and balancing this with precision to avoid sending advertisements to a large number of non-athletes, the RBF kernel SVM appears to be the most suitable model for this classification task, followed by the polynomial kernel SVM.


```
**Feedback:**
Data Generation (30%) - 0.4/0.45
• Creativity and relevance of the chosen scenario - 0.15/0.15
• Correct implementation of data generation techniques - 0.15/0.15
• Clarity and structure of the data (including labeling and documentation) - 0.1/0.15

Model Implementation and Tuning (40%) - 0.6/0.6
• Correct implementation of SVC/SVR models with the three specified kernels - 0.2/0.2
• Effective use of hyper-parameter tuning, with a clear rationale for chosen parameters - 0.2/0.2
• Documentation of model performance metrics, with comparisons to justify the best-performing model - 0.2/0.2

Analysis and Discussion (30%)- 0.45/0.45
• Clarity and depth of the introduction and explanation of the chosen problem and dataset - 0.225/0.225
• Critical analysis of model results, including insights into performance variations across different kernels - 0.225/0.225

Need to create a dedicated notebook for data generation as this keeps your code more organized. So, 0.05 marks were deducted

---

# Student 05
```python
# 
# I have created Credit card classfication problem which has two input variables "Credit Score","Income level" and
# label(whether to issue credit) as output label.
# This Problem can further be developed by adding in more input variables (ex:Working Sector,Assets owned)
# This model after complete developmemt can be Credit card companies for shortening their credit card issuing time.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
np.random.seed(1)


# Generate random data
credit_score = np.random.randint(300, 850, 100)
income_level = np.random.randint(20000, 150000, 100)
labels = np.where((credit_score >= 600) & (income_level >= 40000), 1, 0)

# Create DataFrame
data = {
    'Credit Score': credit_score,
    'Income Level': income_level,
    'Approval Status': labels
}

df = pd.DataFrame(data)

# Display the first few rows of the DataFrame
print(df.head())


df.to_csv('credit.csv', index=False)

from sklearn.svm import SVC
from matplotlib import pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV



X = df[['Credit Score', 'Income Level']]  # Features (credit score and income level)
y = df['Approval Status']

# Splitting the Data Set into Train and Test Data Sets

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)


X_train

performance = pd.DataFrame({"model": [], "Accuracy": [], "Precision": [], "Recall": [], "F1": [], "F2": [], "Parameters": []})

# create a fbeta 2 scorer
from sklearn.metrics import make_scorer
from sklearn.metrics import fbeta_score
f2_scorer = make_scorer(fbeta_score, beta=2)


# Building a SVC model with linear Kernel

# defining parameter range 
param_grid = {'C': [0.01, 0.1, 0.5, 1, 5, 10, 50, 100],  
              'kernel': ['linear']}
  
#grid = GridSearchCV(SVC(), param_grid, scoring='f1', refit = True, verbose = 3, n_jobs=-1) 

grid = GridSearchCV(SVC(), param_grid, scoring=f2_scorer, refit = True, verbose = 3, n_jobs=-1) 
  
# fitting the model for grid search 
_ = grid.fit(X_train, y_train)

# print best parameter after tuning 
print(grid.best_params_) 
  
# print how our model looks after hyper-parameter tuning 
print(grid.best_estimator_)

y_pred = grid.predict(X_test) 

recall = recall_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
f2 = fbeta_score(y_test, y_pred, beta=2)

performance = pd.concat([performance, pd.DataFrame({"model": ["SVM Linear"], "Accuracy": [accuracy], "Precision": [precision], "Recall": [recall], "F1": [f1], "F2": [f2], "Parameters": [grid.best_params_]})])


performance

# ### Linear SVM:
# Accuracy: 90%
# Precision: 80%
# Recall: 100%
# F1-score: 88.89%
# AUC: 95.24%
# Hyperparameters: {'C': 0.01, 'kernel': 'linear'}
# 
# ### Analysis:
# 
# At 90%, the linear SVM's accuracy is comparatively high, meaning that most cases are properly classified. But it's important to remember that accuracy by itself could not give the whole story, particularly when dealing with unbalanced datasets.
# When the precision is 80%, it means that 80% of the occurrences that the model predicted as positive are indeed positive.
# The recall, or sensitivity, of the model is 100%, indicating that it successfully captures all positive cases.
# The harmonic mean of memory and precision, or F1-score, is 88.89%, suggesting that recall and precision are well-balanced.
# With an AUC (Area Under the ROC Curve) of 95.24%, the model appears to function effectively at various thresholds.
# One of the selected hyperparameters is the regularization parameter (C), which is 0.01 and indicates a relatively low regularization strength

# Building a SVC model with rbf Kernel

# defining parameter range 
param_grid = {'C': [0.1, 1, 10, 100],  
              'gamma': [1, 0.1, 0.01, 0.001, 0.0001], 
              'kernel': ['rbf']}
  
grid = GridSearchCV(SVC(), param_grid, scoring=f2_scorer, refit = True, verbose = 3, n_jobs=-1) 
  
# fitting the model for grid search 
_ = grid.fit(X_train, y_train)

# print best parameter after tuning 
print(grid.best_params_) 
  
# print how our model looks after hyper-parameter tuning 
print(grid.best_estimator_)

y_pred = grid.predict(X_test) 

recall = recall_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=1)
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
f2 = fbeta_score(y_test, y_pred, beta=2)

performance = pd.concat([performance, pd.DataFrame({"model": ["SVM rbf"], "Accuracy": [accuracy], "Precision": [precision], "Recall": [recall], "F1": [f1], "F2": [f2],"Parameters": [grid.best_params_]})])

# ### RBF SVM:
# 
# Accuracy: 60%
# Precision: 100%
# Recall: 0%
# F1-score: 0%
# AUC: 0%
# 
# ### Analysis:
# 
# The accuracy of the RBF SVM is just 60%, a significant decrease from the other SVM models.
# With 100% precision, the model is always right when it predicts positive events. But this metric by itself doesn't tell us enough about how well the model works.
# With a recall of 0%, the model is completely unable to identify positives because it is unable to identify any positive occurrences.
# When the model's F1-score and AUC are both 0%, it indicates that it performs extremely poorly across the board.
# The regularization parameter (C) of 1 and the gamma value of 0.0001 are the selected hyperparameters. For the model to perform better, these hyperparameters may need to be adjusted further.
# In conclusion, the RBF SVM model exhibits subpar performance in terms of given the supplied hyperparameters. It fails to effectively classify positive instances, indicating that adjustments to the hyperparameters or possibly a different kernel choice might be necessary for better performance.

# Building a SVC model with poly Kernel

# defining parameter range 
param_grid = {'C': [0.01, 0.1, 0.5, 1, 5, 10, 50, 100],  
              'coef0': [0.01, 0.1, 0.5, 1, 5, 10, 50, 100],
              'kernel': ['poly']}
  
grid = GridSearchCV(SVC(), param_grid, scoring=f2_scorer, refit = True, verbose = 3, n_jobs=-1) 
  
# fitting the model for grid search 
_ = grid.fit(X_train, y_train)

# print best parameter after tuning 
print(grid.best_params_) 
  
# print how our model looks after hyper-parameter tuning 
print(grid.best_estimator_)

y_pred = grid.predict(X_test) 

recall = recall_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
f2 = fbeta_score(y_test, y_pred, beta=2)

performance = pd.concat([performance, pd.DataFrame({"model": ["SVM Poly"], "Accuracy": [accuracy], "Precision": [precision], "Recall": [recall], "F1": [f1], "F2": [f2], "Parameters": [grid.best_params_]})])

performance.sort_values(by="F2", ascending=False)

# ### Polynomial SVM:
# 
# Accuracy: 90%
# Precision: 84.62%
# Recall: 91.67%
# F1-score: 88.00%
# AUC: 90.16%
# 
# ### Analysis:
# 
# 90% accuracy is attained by the polynomial SVM, which is similar to the linear SVM.
# The model appears to offer a decent balance between recall and precision, with recall coming in at 91.67% and precision at 84.62%.
# With an F1-score of 88.00%, overall performance is good.
# With an AUC of 90.16%, the model appears to outperform the linear SVM, albeit marginally, across a range of criteria.
# A regularization parameter (C) of 0.5 and a coefficient (coef0) of 100 are the selected hyperparameters. The model performs well, which can be attributed to these hyperparameters appearing to be sensible selections.
# In conclusion, the polynomial SVM model exhibits a solid balance between precision and recall, performing well across a variety of evaluation metrics when equipped with the specified hyperparameters. The selected hyperparameters appear to be useful, which contributes to the model's overall good performance.
# 




```
**Feedback:**
Data Generation (30%) - 0.4/0.45
• Creativity and relevance of the chosen scenario - 0.15/0.15
• Correct implementation of data generation techniques - 0.15/0.15
• Clarity and structure of the data (including labeling and documentation) - 0.1/0.15

Model Implementation and Tuning (40%) - 0.6/0.6
• Correct implementation of SVC/SVR models with the three specified kernels - 0.2/0.2
• Effective use of hyper-parameter tuning, with a clear rationale for chosen parameters - 0.2/0.2
• Documentation of model performance metrics, with comparisons to justify the best-performing model - 0.2/0.2

Analysis and Discussion (30%)- 0.45/0.45
• Clarity and depth of the introduction and explanation of the chosen problem and dataset - 0.225/0.225
• Critical analysis of model results, including insights into performance variations across different kernels - 0.225/0.225

Need to create a dedicated notebook for data generation as this keeps your code more organized. So, 0.05 marks were deducted

---

# Student 06
```python
# # Vehicle Insurance Claim Prediction Using Support Vector Machine (SVM)

# ### Author details:
# 
# 
# #### Date: 02/26/2024

# In this project, our goal is to predict if someone will claim insurance for their vehicle. We'll use different types of Support Vector Machine (SVM) models with different approaches to understand the data better. We'll mainly focus on how good these models are at predicting claims, paying attention to how well they spot real claims, even if they sometimes make mistakes. We'll use metrics like Accuracy, Precision, Recall, F1 Score, and F2 Score to measure their performance, giving extra importance to Recall.

# ### Import Libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC, SVR
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, f1_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler

np.random.seed(1)


# This code snippet demonstrates the implementation of a Support Vector Machine (SVM) classifier using Scikit-Learn. It includes necessary imports for data manipulation, visualization, model training, evaluation metrics, and preprocessing.

# ### Load and Preprocess Data

# Load the dataset
df = pd.read_csv('/Users/manoj/Desktop/Data_folder/vehicle_insurance_claim_prediction.csv')

# Separate features and target variable
X = df[['AgeOfVehicle', 'NumberofAccidents']]
y = df['ClaimFiled']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# The code snippet provided encompasses loading the dataset, separating features and the target variable, splitting the data into training and test sets, and standardizing the features for improved model performance.
# 
# We're splitting our data into two parts, one for training the model and the other for testing its performance. We use test_size=0.3 to reserve 30% of the data for testing. We also set random_state=1 to ensure that each time we run the code, the data split remains the same, making our results consistent and reproducible.

# ### Model the Data

# #### Initialize Performance Tracking

performance = pd.DataFrame({"model": [], "Accuracy": [], "Precision": [], "Recall": [], "F1": [], "F2": [], "Parameters": []})
#This code initializes a DataFrame named performance to store performance metrics for different models. 

# create a fbeta 2 scorer
from sklearn.metrics import make_scorer
from sklearn.metrics import fbeta_score
f2_scorer = make_scorer(fbeta_score, beta=2)

# This code snippet demonstrates how to create a custom scorer for F-beta 2 score, a metric useful for evaluating binary classification models with a focus on recall. The make_scorer function from sklearn.metrics is utilized to create the scorer, with the beta value set to 2.

# ### a. Fit a SVM classification model using linear kernal

# defining parameter range 
param_grid = {'C': [0.01, 0.1, 0.5, 1, 5, 10, 50, 100],  
              'kernel': ['linear']}
  
#grid = GridSearchCV(SVC(), param_grid, scoring='f1', refit = True, verbose = 3, n_jobs=-1) 

grid = GridSearchCV(SVC(), param_grid, scoring=f2_scorer, refit = True, verbose = 3, n_jobs=-1) 
  
# fitting the model for grid search 
_ = grid.fit(X_train, y_train)

# In this code snippet, a grid search is performed to find the optimal hyperparameters for a Support Vector Machine (SVM) model. The parameter grid consists of different values for the regularization parameter C and the kernel function kernel, with 'linear' specified as the kernel. The f2_scorer scoring function is utilized for evaluation during the grid search, aiming to optimize for a specific F2 score. The model is trained using the training data (X_train and y_train) with parallel processing (n_jobs=-1). The best parameters are selected and used to refit the model (refit=True). The process is executed with verbosity level set to 3 for detailed output during the search.

# print best parameter after tuning 
print(grid.best_params_) 
  
# print how our model looks after hyper-parameter tuning 
print(grid.best_estimator_)

y_pred = grid.predict(X_test) 

# Calculate metrics
recall = recall_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
f2 = fbeta_score(y_test, y_pred, beta=2)

# Update performance dataframe
performance = pd.concat([performance, pd.DataFrame({"model": ["SVM Linear"], "Accuracy": [accuracy], "Precision": [precision], "Recall": [recall], "F1": [f1], "F2": [f2], "Parameters": [grid.best_params_]})])


# After tuning the hyperparameters, we print the best parameters and the updated model. Then, we make predictions on the test data and calculate various performance metrics such as accuracy, precision, recall, F1 score, and F2 score. These metrics help us evaluate how well our Support Vector Machine (SVM) model performs. Finally, we update a datframe containing the performance metrics for comparison with other models.

# ### b. Fit a SVM classification model using rbf kernal

# defining parameter range 
param_grid = {'C': [0.1, 1, 10, 100],  
              'gamma': [1, 0.1, 0.01, 0.001, 0.0001], 
              'kernel': ['rbf']}
  
grid = GridSearchCV(SVC(), param_grid, scoring=f2_scorer, refit = True, verbose = 3, n_jobs=-1) 
  
# fitting the model for grid search 
_ = grid.fit(X_train, y_train)

# We're trying out different settings for our Support Vector Machine model to see which ones work best. We're testing various combinations of 'C' and 'gamma' using the 'rbf' kernel and a scoring method called 'f2_scorer'. Then, we'll train the best model using all available CPU cores for efficiency.

# print best parameter after tuning 
print(grid.best_params_) 
  
# print how our model looks after hyper-parameter tuning 
print(grid.best_estimator_)

y_pred = grid.predict(X_test) 

# Calculate metrics
recall = recall_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
f2 = fbeta_score(y_test, y_pred, beta=2)

# Update performance dataframe
performance = pd.concat([performance, pd.DataFrame({"model": ["SVM rbf"], "Accuracy": [accuracy], "Precision": [precision], "Recall": [recall], "F1": [f1], "F2": [f2],"Parameters": [grid.best_params_]})])

# After tuning the hyperparameters, we print the best parameter and the model's updated configuration. Then, we make predictions on test data and calculate metrics like accuracy, precision, recall, F1, and F2 scores. These metrics assess the performance of our SVM model with the RBF kernel. Finally, we update a dataframe with the model's performance metrics and best parameters.

# ### c. Fit a SVM classification model using polynomial kernal

# defining parameter range 
param_grid = {'C': [0.01, 0.1, 0.5, 1, 5, 10, 50, 100],  
              'coef0': [0.01, 0.1, 0.5, 1, 5, 10, 50, 100],
              'kernel': ['poly']}
  
grid = GridSearchCV(SVC(), param_grid, scoring=f2_scorer, refit = True, verbose = 3, n_jobs=-1) 
  
# fitting the model for grid search 
_ = grid.fit(X_train, y_train)

# We're using a grid search to find the best parameters for a support vector machine (SVM) model. We've defined a range of values for 'C' and 'coef0' parameters and are testing them with a polynomial kernel. The GridSearchCV method helps us systematically find the optimal combination of parameters by evaluating each model's performance using a special scoring function. This process ensures we select the SVM model with the most suitable parameters for our problem.

# print best parameter after tuning 
print(grid.best_params_) 
  
# print how our model looks after hyper-parameter tuning 
print(grid.best_estimator_)

y_pred = grid.predict(X_test) 

# Calculate metrics
recall = recall_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
f2 = fbeta_score(y_test, y_pred, beta=2)

# Update performance dataframe
performance = pd.concat([performance, pd.DataFrame({"model": ["SVM Poly"], "Accuracy": [accuracy], "Precision": [precision], "Recall": [recall], "F1": [f1], "F2": [f2], "Parameters": [grid.best_params_]})])

# We find the best model settings, test the model with new data, and measure how well it performs using various metrics like accuracy, precision, and recall. Then, we update a table with these results and the best settings.

# ### Sorting Performance Data by F2 Column

performance.sort_values(by="F2", ascending=False)

# This code snippet tells the computer to arrange the rows of the performance data based on the values in the "F2" column, with the highest values appearing first. It's like organizing a list from biggest to smallest according to a specific criterion.

# ### Anyalsis

# We looked at different models to predict if someone will file a claim for vehicle insurance. We focused on a score called F2, which helps us figure out how good the models are at spotting cases where people actually file claims.
# 
# Results:
# 
# The 'SVM Poly' model was pretty good. It got an F2 score of 68.68%. That means it did a decent job of finding real claim cases without making too many mistakes.
# The 'SVM Linear' model was a bit better, with an F2 score of 62.94%. It was really good at not getting tricked by false claims, but it sometimes missed out on spotting real ones.
# The 'SVM rbf' model was similar to the linear one, also getting an F2 score of 62.85%.
# 
# What We Found:
# 
# The 'SVM Poly' model was best at spotting real claim cases, even though it sometimes made mistakes by guessing too many claims.
# The 'SVM Linear' model was great at avoiding false alarms, but it missed out on some real claims.
# The 'SVM rbf' model balanced things out, being okay at both spotting real claims and avoiding false alarms.
# 
# Conclusion: 
# If we want a model that's pretty good at both spotting real claims and avoiding false alarms, 'SVM Poly' is the way to go. But if we care more about avoiding false alarms, we might pick 'SVM Linear' or 'SVM rbf' depending on how much we're willing to trade off between spotting real claims and avoiding mistakes.

# # Vehicle Insurance Claim Prediction Dataset Generation.

# ### Author details:
# 
# 
# 
# #### Date: 02/26/2024

# This document outlines the procedure for creating a synthetic dataset. The dataset is designed to simulate the prediction of vehicle insurance claims based on the age of the vehicle and the number of accidents it has been involved in. This exercise aims to showcase the application of binary classification models. The Python programming language, along with libraries such as pandas, numpy, and matplotlib, will be utilized for dataset generation, introducing randomness to reflect real-world data, data visualization, and saving the dataset for future analysis.

# ### Import Libraries
# 
# First, we import the necessary libraries for data manipulation, numerical operations, and plotting.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ### Set Random Seed
# 
# Setting a random seed ensures the reproducibility of our results.

np.random.seed(1)


# ### Generate Sample Data
# 
# We define our sample size and generate random distributions for the features: Age of Vehicle and Number of Accidents.

# Define sample size
sample_size = 1000

# Generate 'AgeOfVehicle' and 'NumberofAccidents' with random distributions
age_of_vehicle = np.random.randint(0, 20, sample_size)  # Vehicles aged between 0 to 20 years
number_of_accidents = np.random.randint(0, 5, sample_size)  # 0 to 4 accidents

# Create DataFrame
df_vehicle = pd.DataFrame({
    'AgeOfVehicle': age_of_vehicle,
    'NumberofAccidents': number_of_accidents
})

# ### Generating Claim Filed Based on Simplistic Assumptions

# Generating ClaimFiled based on simplistic assumptions
df_vehicle['ClaimFiled'] = np.random.binomial(1, p=(number_of_accidents + age_of_vehicle) / (20 + 4))  # Simplistic probability

# This code generates a binary column ClaimFiled in a DataFrame df_vehicle, where the value indicates whether a claim has been filed or not. The probability of filing a claim is calculated based on simplistic assumptions related to the number of accidents and the age of the vehicle.

### Displaying DataFrame Columns

#This code snippet shows how to get and display the names of the columns in a DataFrame called df_vehicle.

print(df_vehicle.columns)


# ### Data Visualization

# Plotting

fig = plt.figure()
ax = fig.add_subplot()
colors = np.array(["blue", "red"])
ax.scatter(df_vehicle['AgeOfVehicle'], df_vehicle['NumberofAccidents'], c=colors[df_vehicle['ClaimFiled']], alpha=0.5)
ax.set_xlabel('Age of Vehicle')
ax.set_ylabel('Number of Accidents')
plt.show()


# In this plot, each point represents a vehicle, with its position determined by its age and the number of accidents it has experienced. The color of each point indicates whether a claim was filed (blue for no claim filed, red for claim filed), offering further insight into the data.

# ### Save to CSV
# 
# Finally, we save our dataset to a CSV file for further analysis or to serve as input for machine learning models.

# Save to CSV
df_vehicle.to_csv('/Users/manoj/Desktop/Data_folder/vehicle_insurance_claim_prediction.csv', index=False)

# Display the first few rows of the DataFrame
print(df_vehicle.head())




```
**Feedback:**
Data Generation (30%) - 0.45/0.45
• Creativity and relevance of the chosen scenario - 0.15/0.15
• Correct implementation of data generation techniques - 0.15/0.15
• Clarity and structure of the data (including labeling and documentation) - 0.15/0.15

Model Implementation and Tuning (40%) - 0.6/0.6
• Correct implementation of SVC/SVR models with the three specified kernels - 0.2/0.2
• Effective use of hyper-parameter tuning, with a clear rationale for chosen parameters - 0.2/0.2
• Documentation of model performance metrics, with comparisons to justify the best-performing model - 0.2/0.2

Analysis and Discussion (30%)- 0.45/0.45
• Clarity and depth of the introduction and explanation of the chosen problem and dataset - 0.225/0.225
• Critical analysis of model results, including insights into performance variations across different kernels - 0.225/0.225

---

# Student 07
```python
# 
# 
# ## WE03-SVM

# ### Introduction
# In this analysis, we aimed to evaluate the performance of Support Vector Machine (SVM) classifiers with different kernels (linear, RBF, and polynomial) on a synthetic dataset generated to mimic a grading system based on marks obtained in two subjects. The dataset consists of randomly generated marks for two subjects for 1000 samples, along with the corresponding grades A,B,C,D and less than that its F which is fail.

#import libraries

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score, fbeta_score

# Step 1: Generate synthetic data

np.random.seed(0)  # For reproducibility

# Generate random marks for two subjects (0-100)

num_samples = 1000
marks_subject1 = np.random.randint(0, 101, num_samples)
marks_subject2 = np.random.randint(0, 101, num_samples)

# Define grading system based on marks obtained

def get_grade(mark):
    if mark >= 90:
        return 'A'
    elif 80 <= mark < 90:
        return 'B'
    elif 70 <= mark < 80:
        return 'C'
    elif 60 <= mark < 70:
        return 'D'
    else:
        return 'F'  # Fail

# Assign grades based on marks obtained

grades = [get_grade((m1 + m2) / 2) for m1, m2 in zip(marks_subject1, marks_subject2)]

# Create DataFrame for the synthetic dataset

data = pd.DataFrame({'Subject1': marks_subject1, 'Subject2': marks_subject2, 'Grade': grades})

# Step 2: Split data into features and target variable

X = data[['Subject1', 'Subject2']]
y = data['Grade']

# After generating the data, we split it into training and testing sets, performed hyper-parameter tuning for the SVM classifiers.

# Step 3: Split data into training and testing sets

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8, random_state=42)

# Step 4: Perform hyper-parameter tuning for SVC

models = {
    'SVC_linear': GridSearchCV(SVC(kernel='linear'), {'C': [0.1, 1, 10]}, cv=5),
    'SVC_rbf': GridSearchCV(SVC(kernel='rbf'), {'C': [0.1, 1, 10], 'gamma': [0.1, 1, 10]}, cv=5),
    'SVC_poly': GridSearchCV(SVC(kernel='poly'), {'C': [0.1, 1, 10], 'degree': [2, 3, 4]}, cv=5)
}

# Fit models and print best hyper-parameters

for name, model in models.items():
    model.fit(X_train, y_train)
    print(f"{name} Best Hyperparameters: {model.best_params_}")

# Evaluating the models' performance using accuracy, precision, recall, F1-score, and F2-score metrics.


# Evaluate model performance

for name, model in models.items():
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    f2 = fbeta_score(y_test, y_pred, average='weighted', beta=2)
    print(f"{name} Metrics:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-score: {f1:.4f}")
    print(f"F2-score: {f2:.4f}")
    print()


# SVC Linear Kernel: The linear kernel SVM achieved the highest overall performance among the three kernels, with an accuracy, precision, recall, F1-score, and F2-score all above 0.98. This indicates that the linear SVM effectively separated the classes in the dataset and made accurate predictions for the grades.
# 
# SVC RBF Kernel: The RBF kernel SVM showed decent performance, although it was noticeably lower compared to the linear kernel. While it still achieved good accuracy, precision, and recall, the F1-score and F2-score were lower, indicating a slightly lower balance between precision and recall. This suggests that the RBF kernel may not be the best choice for this particular dataset.
# 
# SVC Polynomial Kernel: The polynomial kernel SVM also demonstrated strong performance, with accuracy, precision, recall, F1-score, and F2-score all above 0.97. While slightly lower than the linear kernel, it outperformed the RBF kernel across all metrics, indicating that it may be a better choice than the RBF kernel for this dataset.

# Plotting the graph based on Accuracy for linear, rbf and poly metrics


# Extract model names and their accuracies
model_names = list(models.keys())
accuracies = [accuracy_score(y_test, model.predict(X_test)) for model in models.values()]

# Plot the line graph
plt.figure(figsize=(10, 6))
plt.plot(model_names, accuracies, marker='o', linestyle='-')
plt.title('Accuracy of Different SVM Kernels')
plt.xlabel('SVM Kernel')
plt.ylabel('Accuracy')
plt.ylim(0, 1)  # Limit y-axis to range [0, 1]
plt.grid(True)
plt.show()



# Extract model names and their accuracies

model_names = list(models.keys())
accuracies = [accuracy_score(y_test, model.predict(X_test)) for model in models.values()]

# Plot the pie chart

plt.figure(figsize=(8, 8))
plt.pie(accuracies, labels=model_names, autopct='%1.1f%%', startangle=140)
plt.title('Accuracy of Different SVM Kernels')
plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.show()


# ## Conclusion 

# Based on the performance metrics and considering the balance between precision and recall, the SVC with a linear kernel appears to be the best fit for this grading system dataset. It achieved the highest accuracy and balanced performance across all evaluation metrics, indicating its effectiveness in predicting grades based on subject marks.


```
**Feedback:**
Data Generation (30%) - 0.4/0.45
• Creativity and relevance of the chosen scenario - 0.15/0.15
• Correct implementation of data generation techniques - 0.15/0.15
• Clarity and structure of the data (including labeling and documentation) - 0.1/0.15

Model Implementation and Tuning (40%) - 0.6/0.6
• Correct implementation of SVC/SVR models with the three specified kernels - 0.2/0.2
• Effective use of hyper-parameter tuning, with a clear rationale for chosen parameters - 0.2/0.2
• Documentation of model performance metrics, with comparisons to justify the best-performing model - 0.2/0.2

Analysis and Discussion (30%)- 0.45/0.45
• Clarity and depth of the introduction and explanation of the chosen problem and dataset - 0.225/0.225
• Critical analysis of model results, including insights into performance variations across different kernels - 0.225/0.225

Need to create a dedicated notebook for data generation as this keeps your code more organized. So, 0.05 marks were deducted

---

# Student 08
```python
# # W03 SVM

# ## Introduction
# 
# I have created a sample data of binary distribution which has two features and one target. Features are temparature and humidity. Based on these two features we can determine if its gonna rain or not which is target making it a binary classification problem. We have 5000 observations based on which we can create a model(SVM) which can be used to predict if its gonna rain or not. We will use GridSearchCV to find the best hyper-parameters for the model - and we will test rbf, linear and polynomial kernels. The scoring metric we will use is custom beta score, with a beta of 2 (which means we are more interested in recall than precision). We will use the `SVM` class to fit a model to the data and then plot the decision boundary.

# ## 1. Setup

import pandas as pd
from sklearn.svm import SVC
from matplotlib import pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV

np.random.seed(1)

# ## 2. Load and Exploring Data

df = pd.read_csv("C:/Users/ravik/Downloads/weather_prediction_large_dataset.csv")
df.head(3)

# Using the info function we can check the data type and count

df.info()

# Using the describe function we can observe different statistics

df.describe()

# Split the dataset into features and target variable
X = df[['Temperature', 'Humidity']]  # Features
y = df['WillRain']  # Target variable

# Here X are the feautres where features are Temperature and Humidity, y is the target which is binary willRain

# Splitting the data into test and train and scaling the data

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
from sklearn.preprocessing import StandardScaler

# Assuming X_train and X_test are your features for training and test sets respectively

# Initialize the StandardScaler
scaler = StandardScaler()

# Fit on the training data
scaler.fit(X_train)

# Transform the training data
X_train = scaler.transform(X_train)

# Transform the test data
X_test = scaler.transform(X_test)

# ## 3. Model the data

# Creating a performance variable to store all 3 model's score to sort and analyze 

performance = pd.DataFrame({"model": [], "Accuracy": [], "Precision": [], "Recall": [], "F1": [], "F2": [], "Parameters": []})

# create a fbeta 2 scorer
from sklearn.metrics import make_scorer
from sklearn.metrics import fbeta_score
f2_scorer = make_scorer(fbeta_score, beta=2)

# ### 3.1 Fit a SVM classification model using linear kernal

# defining parameter range 
param_grid = {'C': [0.01, 0.1, 0.5, 1, 5, 10, 50, 100],  
              'kernel': ['linear']}
  
#grid = GridSearchCV(SVC(), param_grid, scoring='f1', refit = True, verbose = 3, n_jobs=-1) 

grid = GridSearchCV(SVC(), param_grid, scoring=f2_scorer, refit = True, verbose = 3, n_jobs=-1) 
  
# fitting the model for grid search 
_ = grid.fit(X_train, y_train)

# print best parameter after tuning 
print(grid.best_params_) 
  
# print how our model looks after hyper-parameter tuning 
print(grid.best_estimator_)

y_pred = grid.predict(X_test) 

recall = recall_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
f2 = fbeta_score(y_test, y_pred, beta=2)

performance = pd.concat([performance, pd.DataFrame({"model": ["SVM Linear"], "Accuracy": [accuracy], "Precision": [precision], "Recall": [recall], "F1": [f1], "F2": [f2], "Parameters": [grid.best_params_]})])

performance

# ### 3.2 Fit a SVM classification model using rbf kernal

# defining parameter range 
param_grid = {'C': [0.1, 1, 10, 100],  
              'gamma': [1, 0.1, 0.01, 0.001, 0.0001], 
              'kernel': ['rbf']}
  
grid = GridSearchCV(SVC(), param_grid, scoring=f2_scorer, refit = True, verbose = 3, n_jobs=-1) 
  
# fitting the model for grid search 
_ = grid.fit(X_train, y_train)

# print best parameter after tuning 
print(grid.best_params_) 
  
# print how our model looks after hyper-parameter tuning 
print(grid.best_estimator_)

y_pred = grid.predict(X_test) 

recall = recall_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
f2 = fbeta_score(y_test, y_pred, beta=2)

performance = pd.concat([performance, pd.DataFrame({"model": ["SVM rbf"], "Accuracy": [accuracy], "Precision": [precision], "Recall": [recall], "F1": [f1], "F2": [f2],"Parameters": [grid.best_params_]})])

# ### 3.3 Fit a SVM classification model using polynomial kernal

# defining parameter range 
param_grid = {'C': [0.01, 0.1, 0.5, 1, 5, 10, 50, 100],  
              'coef0': [0.01, 0.1, 0.5, 1, 5, 10, 50, 100],
              'kernel': ['poly']}
  
grid = GridSearchCV(SVC(), param_grid, scoring=f2_scorer, refit = True, verbose = 3, n_jobs=-1) 
  
# fitting the model for grid search 
_ = grid.fit(X_train, y_train)

# print best parameter after tuning 
print(grid.best_params_) 
  
# print how our model looks after hyper-parameter tuning 
print(grid.best_estimator_)

y_pred = grid.predict(X_test) 

recall = recall_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
f2 = fbeta_score(y_test, y_pred, beta=2)

performance = pd.concat([performance, pd.DataFrame({"model": ["SVM Poly"], "Accuracy": [accuracy], "Precision": [precision], "Recall": [recall], "F1": [f1], "F2": [f2], "Parameters": [grid.best_params_]})])

# ## 4.0 Summary

performance.sort_values(by="F2", ascending=False)

# 
# ### SVM with RBF Kernel
# - **Accuracy**: 1.000 indicates that the model perfectly classified all the test data.
# - **Precision**: 1.000 suggests that every instance predicted as positive by the model was indeed positive.
# - **Recall**: 1.000 means the model identified all positive instances correctly.
# - **F1 Score**: 1.000 shows a perfect balance between precision and recall.
# - **F2 Score**: 1.000 indicates a preference for recall in the calculation, but since recall is perfect, the score is also perfect.
# - **Parameters**: Used `C=1` and `gamma=1`, which suggests a certain level of regularization and scale of the Gaussian kernel respectively.
# 
# ### SVM with Polynomial Kernel
# - **Accuracy**: 0.998 demonstrates almost perfect classification, with very few misclassifications.
# - **Precision**: 0.996454 indicates that almost all instances predicted as positive are indeed positive, with a very small margin of error.
# - **Recall**: 1.000 means the model captured all positive instances without missing any.
# - **F1 Score**: 0.998224 is nearly perfect, indicating a strong balance between precision and recall.
# - **F2 Score**: 0.999289 suggests a slight emphasis on recall in the score's calculation, which remains near perfect.
# - **Parameters**: `C=100` and `coef0=10` with a polynomial kernel indicate a high degree of model complexity and adjustment for the independent term in the kernel function.
# 
# ### SVM with Linear Kernel
# - **Accuracy**: 0.857 shows that the model has a good level of correct classifications, though not as high as the other models.
# - **Precision**: 0.877477 suggests a high probability that predicted positive instances are actually positive.
# - **Recall**: 0.866548 indicates that the model identified a majority of the positive instances.
# - **F1 Score**: 0.871979 reflects a balance between precision and recall, though not as high as the other models.
# - **F2 Score**: 0.868712 shows a slight emphasis on recall, indicating good performance but not as outstanding as the other models.
# - **Parameters**: `C=0.01` with a linear kernel suggests the model uses a high degree of regularization.
# 
# ### Summary
# - The **SVM with RBF kernel** shows perfect performance across all metrics, potentially indicating overfitting to the training data or a very clear margin of separation in the data.
# - The **SVM with Polynomial kernel** also exhibits excellent performance, nearly matching the RBF kernel, indicating its effectiveness for the given data with slightly complex patterns.
# - The **SVM with Linear kernel** demonstrates good but not exceptional performance, suggesting that the data may not be linearly separable, which limits the effectiveness of this simpler model.
# - The differences in performance metrics underscore the importance of choosing the right kernel and parameters for SVM models based on the specific characteristics of the data.
# 
# In the "Will Rain" scenario, using the F2 score instead of the F1 score means we're more concerned about missing a prediction of rain than we are about mistakenly saying it will rain when it won't. Basically, it's more important to catch all the times it might rain, even if that means sometimes we say it will rain and it doesn't. This way, we're erring on the side of caution, ensuring we're prepared for rain that might come, rather than being caught off guard by rain we didn't predict.
# 
# These results highlight the versatility and power of SVMs in handling various data distributions and the critical role of parameter tuning in achieving optimal model performance.

import numpy as np
import pandas as pd

# Generate random data for 1000 samples
np.random.seed(1)  # For reproducibility
temperatures = np.random.randint(15, 35, size=5000)  # Temperatures between 15 and 35 degrees Celsius
humidities = np.random.randint(40, 100, size=5000)  # Humidity between 40% and 100%

# Define a simple rule for WillRain: more likely to rain if humidity > 75% or temperature < 20
will_rain = [1 if humidity > 75 or temperature < 20 else 0 for temperature, humidity in zip(temperatures, humidities)]

# Create a DataFrame
df_large = pd.DataFrame({
    'Temperature': temperatures,
    'Humidity': humidities,
    'WillRain': will_rain
})

# Save the DataFrame to a CSV file
filename_large = 'C:/Users/ravik/Downloads/weather_prediction_large_dataset.csv'
df_large.to_csv(filename_large, index=False)

filename_large

# The scenario  revolves around a binary classification task, where the objective is to predict rainfall events based on daily weather conditions. Specifically, the model aims to determine the likelihood of rain (denoted as 1 for rain, and 0 for no rain) by analyzing two key meteorological features: temperature and humidity. This task aligns with typical binary classification problems in machine learning, where the goal is to categorize data points into two distinct groups.
# 
# In this case, the underlying hypothesis for predicting rain is based on specific thresholds for temperature and humidity. The criteria for a rainfall event are defined as follows: rain is expected when the temperature is below 20 degrees Celsius and the humidity exceeds 75%. Conversely, conditions not meeting these criteria are predicted to result in no rainfall.
# 
#  By applying machine learning models such as Support Vector Machines (SVM) with different kernels (linear, RBF, and polynomial), we explore the capacity of these algorithms to capture the relationship between the input features and the binary outcome of rain or no rain. Through this process, we aim to evaluate the effectiveness of various model configurations and hyperparameters in predicting rainfall, thereby gaining insights into the predictive power and limitations of machine learning models in weather forecasting.

import pandas as pd
df_large = pd.read_csv("C:/Users/ravik/Downloads/weather_prediction_large_dataset.csv")
df_large.head(3)

df_large.info()

df_large.describe()

import matplotlib.pyplot as plt

# Scatter plot
plt.figure(figsize=(10, 6))
plt.scatter(df_large['Temperature'], df_large['Humidity'], c=df_large['WillRain'], cmap='coolwarm', alpha=0.5)
plt.colorbar(label='Will Rain')
plt.xlabel('Temperature (°C)')
plt.ylabel('Humidity (%)')
plt.title('Temperature vs Humidity')
plt.grid(True)
plt.show()

# - The horizontal line at the bottom shows how warm or cold it is, from kind of cool (15°C) to pretty warm (over 32°C).
# - The vertical line on the side shows how moist the air is, from not too damp (40%) to really wet (100%).
# - The colors of the dots tell us if it's likely to rain or not. Light blue dots mean there's not much chance of rain, and dark red dots mean it's very likely to rain.
# 
# Looking at the dots, we can see:
# - When it's cooler (under 20°C), the dots tend to be dark red, meaning there's a good chance it'll rain.
# - When the air is very damp (over 75% humidity), the dots are also dark red, so it's likely to rain then, too, even if it's warmer.




```
**Feedback:**
Data Generation (30%) - 0.45/0.45
• Creativity and relevance of the chosen scenario - 0.15/0.15
• Correct implementation of data generation techniques - 0.15/0.15
• Clarity and structure of the data (including labeling and documentation) - 0.15/0.15

Model Implementation and Tuning (40%) - 0.6/0.6
• Correct implementation of SVC/SVR models with the three specified kernels - 0.2/0.2
• Effective use of hyper-parameter tuning, with a clear rationale for chosen parameters - 0.2/0.2
• Documentation of model performance metrics, with comparisons to justify the best-performing model - 0.2/0.2

Analysis and Discussion (30%)- 0.45/0.45
• Clarity and depth of the introduction and explanation of the chosen problem and dataset - 0.225/0.225
• Critical analysis of model results, including insights into performance variations across different kernels - 0.225/0.225

---

# Student 09
```python
# ## <b> Assignment 3: Support Vector Mechanism </b>

# A dataset is created with Independent variables as Runs scored and Wickets took by a player in his career in the Domestic Matches to get selected in the international team.
# 
# <b> Independent Variables :</b> <br>
# <b>Runs:</b>  This indicates the number of runs scored by a player in all the cricket matches he played.<br>
# <b>Wickets:</b>  This tells about the number of wickets he took in his career.
# 
# <b> Dependent Variable:</b> <br>
# <b> Selected :</b>  This variable tells whether the player is selected or not for the International Cricket team.

# ## 1. Importing all the necessary libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ## 2. Defining the Sample Size, Minimum and Maximum values in each Feature

# > Minimum and Maximum values of each features i.e., Runs and Wickets are defined here. Also, sample size is specified here

runs_min = 1000
runs_max = 10000
wickets_min = 50
wickets_max = 400
sample_size= 700

# > <b> Random Seed </b> is used to get the same sequence of numbers everytime. Using a random seed will ensure that the model's training and evaluation processes are consistent across different runs.  

np.random.seed(100)

# > Selecting 700 random numbers from the above specified Minimum and Maximums.

runs = np.random.randint(runs_min,runs_max,size=sample_size)
wickets = np.random.randint(wickets_min,wickets_max,size=sample_size)

# > <b> Creating a Dataframe from the array of values generated above and storing in cric_df dataframe.</b>

cric_df = pd.DataFrame({'Runs':runs,'Wickets':wickets})
cric_df

# > Now we have to select the player who performed well in both the categories.The player who scored more than 5000 runs and took more than 250 wickets will get selected for the International team.

a= cric_df['Runs']>4000
b = cric_df['Wickets']>150

cric_df['Selected'] = ((a) & (b)).astype(int)

cric_df

# > Checking the <b>data types</b> of each variable in the Dataframe

cric_df.info()

# > Calculate the overall number of <b>missing values</b> in each column 

cric_df.isna().sum()

# > <b> Calculating the Number of 0's and 1's in the Selected Column </b>

cric_df['Selected'].value_counts()

# > <b> Storing all the Independent Variables in the X and target variables in the y </b> 

X= cric_df[['Runs','Wickets']]
y= cric_df['Selected']

# > <b> Splitting the Dataset with test size of 20% </b> 

fom sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=1)

# > <b> Standardizing the features.</b> <br> This will ensure that all features are centered around zero and have the same variance.

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_sc = pd.DataFrame(scaler.fit_transform(X_train),columns=['Runs','Wickets'])
X_test_sc = pd.DataFrame(scaler.transform(X_test),columns=['Runs','Wickets'])
X_train_sc.head()

from sklearn.metrics import make_scorer
from sklearn.metrics import fbeta_score

# ## Linear SVM

# > <b> Finding the best hyperparameter configuration (C value) for an SVM classifier using a linear kernel, based on maximizing the F2 score across different values of C. </b>

from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
f2_scorer = make_scorer(fbeta_score, beta=2)
param = {'C':[0.01,0.1,1,5,10,50,100],'kernel':['linear']}
lr_grid = GridSearchCV(estimator=SVC(),param_grid=param,scoring=f2_scorer,verbose=3,n_jobs=-1)
model = lr_grid.fit(X_train_sc,y_train)

# >  <b> Identifying the Parameters that led to the best model performance during the grid search like C and kernel in the SVM model. </b>

lr_grid.best_params_

from sklearn.metrics import accuracy_score,f1_score,precision_score,recall_score
y_pred = model.predict(X_test_sc)
lr_accuracy = accuracy_score(y_test,y_pred)
lr_recall = recall_score(y_test,y_pred)
lr_precision = precision_score(y_test,y_pred)
lr_f1 = f1_score(y_test,y_pred)
df_scores = pd.DataFrame()

# ## Polynomial SVM

# > <b> Finding the best hyperparameter configuration (C value) for an SVM classifier using a poly kernel, based on maximizing the F2 score across different values of C.</b>

param = {'C':[0.01,0.1,1,5,10,50,100],'coef0':[0.01,0.1,1,5,10,50,100],'kernel':['poly']}
pl_grid = GridSearchCV(estimator=SVC(),param_grid=param,scoring=f2_scorer,verbose=3,n_jobs=-1)
model = pl_grid.fit(X_train_sc,y_train)

# >  <b> Identifying the Parameters that led to the best model performance during the grid search like C and kernel in the SVM model. </b>

pl_grid.best_params_

y_pred = model.predict(X_test_sc)
pl_accuracy = accuracy_score(y_test,y_pred)
pl_recall = recall_score(y_test,y_pred)
pl_precision = precision_score(y_test,y_pred)
pl_f1 = f1_score(y_test,y_pred)

# ## RBF SVM

# > <b> Finding the best hyperparameter configuration (C value) for an SVM classifier using a poly kernel, based on maximizing the F2 score across different values of C.</b>

param = {'C':[0.01,0.1,1,5,10,50,100],'gamma':[1,0.1,0.001,0.0001],'kernel':['rbf']}
rb_grid = GridSearchCV(estimator=SVC(),param_grid=param,scoring=f2_scorer,verbose=3,n_jobs=-1)
model = rb_grid.fit(X_train_sc,y_train)

# >  <b> Identifying the Parameters that led to the best model performance during the grid search like C and kernel in the SVM model. </b>

rb_grid.best_params_

y_pred = model.predict(X_test_sc)
rb_accuracy = accuracy_score(y_test,y_pred)
rb_recall = recall_score(y_test,y_pred)
rb_precision = precision_score(y_test,y_pred)
rb_f1 = f1_score(y_test,y_pred)

# #### Creating the Lists and Dictionary to take a summary of the Values obtained by running each type of Model

model = ['Linear SVM','Polynomial SVM', 'RBF SVM']
accuracy = [lr_accuracy,pl_accuracy,rb_accuracy]
recall = [lr_recall, pl_recall,rb_recall]
precision = [lr_precision, pl_precision,rb_precision]
f1 = [lr_f1, pl_f1,rb_f1]
bestparameters = [lr_grid.best_params_,pl_grid.best_params_,rb_grid.best_params_]

summary_dict = {
    'Model': model,
    'Accuracy Score' : accuracy,
    'Recall Score' : recall,
    'Precision Score' : precision,
    'F1 Score' : f1,
    'Best Parameters' : bestparameters
}

# #### Creating a Dataframe to show the accuracy, precision, recall, F1 Score and the Best Parameters in each Model

summary = pd.DataFrame(summary_dict)

summary

# <b> Accuracy is used to measure the overall correctness of the model. Recall Score is used to find the actual positive cases that were correctly identified by the model. Precision score is used to identify positive identifications that were actually correct. F1 Score is the harmonic mean of precision and recall, providing a balance between the two. </b>

# <b> The best model among the three Models considered is Polynomial SVM and the RBF SVM. Both the Models have the highest accuracy score (0.978571), recall score (0.951613), precision score (1.000000), and F1 score (0.975207) compared to the Linear SVM. These metrics suggest that both the Polynomial and RBF SVM models are highly effective in correctly classifying the data, with a perfect precision score indicating no false positives in the predictions. </b>


```
**Feedback:**
Data Generation (30%) - 0.4/0.45
• Creativity and relevance of the chosen scenario - 0.15/0.15
• Correct implementation of data generation techniques - 0.15/0.15
• Clarity and structure of the data (including labeling and documentation) - 0.1/0.15

Model Implementation and Tuning (40%) - 0.6/0.6
• Correct implementation of SVC/SVR models with the three specified kernels - 0.2/0.2
• Effective use of hyper-parameter tuning, with a clear rationale for chosen parameters - 0.2/0.2
• Documentation of model performance metrics, with comparisons to justify the best-performing model - 0.2/0.2

Analysis and Discussion (30%)- 0.35/0.45
• Clarity and depth of the introduction and explanation of the chosen problem and dataset - 0.125/0.225
• Critical analysis of model results, including insights into performance variations across different kernels - 0.225/0.225

Need to create a dedicated notebook for data generation as this keeps your code more organized and should mention whether you choose classification or regression task in introduction or data generation notebook. So, 0.15 marks were deducted.

---

# Student 10
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(1)

sample_size = 25
min_val = 1
max_val = 10

df = pd.DataFrame({'kgs_smoked': np.linspace(min_val,max_val,sample_size)})
df

df['cancer'] = (df['kgs_smoked'] > (max_val/min_val)/2).astype(int)

df['kgs_smoked'] = df['kgs_smoked'] + np.random.uniform(-10,10, sample_size)
df

fig = plt.figure()
ax = fig.add_subplot()
colors = np.array(["blue", "red"])
ax.scatter(df['kgs_smoked'], df['cancer'], c=colors[np.ravel(df['cancer'])])
ax.set_xlabel('kgs smoked')
ax.set_ylabel('positive cancer disgnosis')
plt.show()


df.to_csv('./data/cancer.csv', index=False)

# # SVM Demonstration
# 
# In this tutorial we will demonstrate how to use the `SVM` class in `scikit-learn` to perform logistic regression on a dataset. 
# 
# The synthetic dataset we will use is the cancer dataset that is produced by the data_gen notebook. 
# 
# This is a simple dataset that predicts if someone has cancer based on the number of kilograms of tobacco they have smoked in total.
# 
# This dataset, therefore, has only one feature and a binary target variable (1 is they have cancer, 0 if they don't).
# 
# We will use the `SVM` class to fit a model to the data and then plot the decision boundary.
# 
# We will also use the `SVM` class to predict the probability of a person having cancer based on the number of kilograms of tobacco they have smoked.
# 
# We will use GridSearchCV to find the best hyper-parameters for the model - and we will test rbf, linear and polynomial kernels. The scoring metric we will use is custom beta score, with a beta of 2 (which means we are more interested in recall than precision).
# 
# The reason for this metric is that their is a difference between the cost of a false positive and a false negative in this case. A false negative is much more costly than a false positive, as it means someone with cancer is not being treated. But, we cannot fully ignore precision, as we don't want to be treating people who don't have cancer.

# ## 1. Setup

# Import modules

import pandas as pd
from sklearn.svm import SVC
from matplotlib import pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV

np.random.seed(1)

# ## 2. Load data

# Load data (it's already cleaned and preprocessed)

# Uncomment the following snippet of code to debug problems with finding the .csv file path
# This snippet of code will exit the program and print the current working directory.
#import os
#print(os.getcwd())

df = pd.read_csv('./data/cancer.csv') # let's use the same data as we did in the logistic regression example
df.head(3)

# Use sklearn to split df into a training set and a test set

X = df[['kgs_smoked']]
y = df['cancer']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)


# ## 3. Model the data

# First, let's create a dataframe to load the model performance metrics into.

performance = pd.DataFrame({"model": [], "Accuracy": [], "Precision": [], "Recall": [], "F1": [], "F2": [], "Parameters": []})

# create a fbeta 2 scorer
from sklearn.metrics import make_scorer
from sklearn.metrics import fbeta_score
f2_scorer = make_scorer(fbeta_score, beta=2)


# ### 3.1 Fit a SVM classification model using linear kernal

# defining parameter range 
param_grid = {'C': [0.01, 0.1, 0.5, 1, 5, 10, 50, 100],  
              'kernel': ['linear']}
  
#grid = GridSearchCV(SVC(), param_grid, scoring='f1', refit = True, verbose = 3, n_jobs=-1) 

grid = GridSearchCV(SVC(), param_grid, scoring=f2_scorer, refit = True, verbose = 3, n_jobs=-1) 
  
# fitting the model for grid search 
_ = grid.fit(X_train, y_train)

# print best parameter after tuning 
print(grid.best_params_) 
  
# print how our model looks after hyper-parameter tuning 
print(grid.best_estimator_)

y_pred = grid.predict(X_test) 

recall = recall_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
f2 = fbeta_score(y_test, y_pred, beta=2)

performance = pd.concat([performance, pd.DataFrame({"model": ["SVM Linear"], "Accuracy": [accuracy], "Precision": [precision], "Recall": [recall], "F1": [f1], "F2": [f2], "Parameters": [grid.best_params_]})])


performance

# ### 3.2 Fit a SVM classification model using rbf kernal

# defining parameter range 
param_grid = {'C': [0.1, 1, 10, 100],  
              'gamma': [1, 0.1, 0.01, 0.001, 0.0001], 
              'kernel': ['rbf']}
  
grid = GridSearchCV(SVC(), param_grid, scoring=f2_scorer, refit = True, verbose = 3, n_jobs=-1) 
  
# fitting the model for grid search 
_ = grid.fit(X_train, y_train)

# print best parameter after tuning 
print(grid.best_params_) 
  
# print how our model looks after hyper-parameter tuning 
print(grid.best_estimator_)

y_pred = grid.predict(X_test) 

recall = recall_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
f2 = fbeta_score(y_test, y_pred, beta=2)

performance = pd.concat([performance, pd.DataFrame({"model": ["SVM rbf"], "Accuracy": [accuracy], "Precision": [precision], "Recall": [recall], "F1": [f1], "F2": [f2],"Parameters": [grid.best_params_]})])

# ### 3.3 Fit a SVM classification model using polynomial kernal

# defining parameter range 
param_grid = {'C': [0.01, 0.1, 0.5, 1, 5, 10, 50, 100],  
              'coef0': [0.01, 0.1, 0.5, 1, 5, 10, 50, 100],
              'kernel': ['poly']}
  
grid = GridSearchCV(SVC(), param_grid, scoring=f2_scorer, refit = True, verbose = 3, n_jobs=-1) 
  
# fitting the model for grid search 
_ = grid.fit(X_train, y_train)

# print best parameter after tuning 
print(grid.best_params_) 
  
# print how our model looks after hyper-parameter tuning 
print(grid.best_estimator_)

y_pred = grid.predict(X_test) 

recall = recall_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
f2 = fbeta_score(y_test, y_pred, beta=2)

performance = pd.concat([performance, pd.DataFrame({"model": ["SVM Poly"], "Accuracy": [accuracy], "Precision": [precision], "Recall": [recall], "F1": [f1], "F2": [f2], "Parameters": [grid.best_params_]})])

# ## 4.0 Summary

# From out results, we can see that the linear kernel and rbf models perform the best. For the SVC model with a linear kernel, the best C value is 0.1. For the SVC model with a rbf kernel, the best C value is 0.1 and the best gamma value is 0.1. The polynomial kernel model did not perform as well as the other two models and therefore we will not use it.

performance.sort_values(by="F2", ascending=False)


```
**Feedback:**
Data Generation (30%) - 0.45/0.45
• Creativity and relevance of the chosen scenario - 0.15/0.15
• Correct implementation of data generation techniques - 0.15/0.15
• Clarity and structure of the data (including labeling and documentation) - 0.15/0.15

Model Implementation and Tuning (40%) - 0.6/0.6
• Correct implementation of SVC/SVR models with the three specified kernels - 0.2/0.2
• Effective use of hyper-parameter tuning, with a clear rationale for chosen parameters - 0.2/0.2
• Documentation of model performance metrics, with comparisons to justify the best-performing model - 0.2/0.2

Analysis and Discussion (30%)- 0.35/0.45
• Clarity and depth of the introduction and explanation of the chosen problem and dataset - 0.125/0.225
• Critical analysis of model results, including insights into performance variations across different kernels - 0.225/0.225

Need to mention whether you choose regression or classification task in introduction or data generation notebook. So, 0.1 marks were deducted.

---

# Student 11
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(1)

sample_size = 20
weight_min_val = 28
weight_max_val = 130
height_min_val=1.2
height_max_val=1.5

df = pd.DataFrame({'weight': np.linspace(weight_min_val,weight_max_val,sample_size),'height':np.linspace(height_min_val,height_max_val,sample_size)})
df

bmi=df['weight']/df['height']**2
max_bmi = bmi.max()
min_bmi = bmi.min()
df['healthy_weight'] = (bmi<(max_bmi+min_bmi)/2 ).astype(int)
df

df.to_csv('bmi.csv', index=False)

# The dataset contains information on individuals' weight and height, and it computes their Body Mass Index (BMI) while also categorizing whether their BMI suggests a healthy weight status.

df=pd.read_csv('bmi.csv')
df.head(10)

from sklearn.svm import SVC
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, f1_score


X=df.iloc[:,:2]
y=df[['healthy_weight']]

# creating training and test data

X_train, X_test, y_train, y_test=train_test_split(X,y,test_size=0.3,random_state=43)

np.ravel(y_train)


metrics = pd.DataFrame({"model": [], "Accuracy": [], "Precision": [], "Recall": [], "F1": [], "F2": [], "Parameters": []})


# creating a fbeta 2 scorer
from sklearn.metrics import make_scorer
from sklearn.metrics import fbeta_score
f2_scorer = make_scorer(fbeta_score, beta=2)

# ### Fitting a SVM classification model using linear kernal

# defining parameter range 
param_grid = {'C': [0.01, 0.1, 0.5, 1, 5, 10, 50, 100],  
              'kernel': ['linear']}
grid = GridSearchCV(SVC(), param_grid, scoring=f2_scorer, refit = True, verbose = 3, n_jobs=-1) 
  
# fitting the model for grid search 
_ = grid.fit(X_train, np.ravel(y_train))

print(grid.best_params_) 
  
# model after hyper-parameter tuning 
print(grid.best_estimator_)

y_pred = grid.predict(X_test) 

recall = recall_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
f2 = fbeta_score(y_test, y_pred, beta=2)

metrics = pd.concat([metrics, pd.DataFrame({"model": ["SVM Linear"], "Accuracy": [accuracy], "Precision": [precision], "Recall": [recall], "F1": [f1], "F2": [f2], "Parameters": [grid.best_params_]})])


metrics

# The best parameters obtained for the SVM model with a linear kernel are {'C': 0.01, 'kernel': 'linear'}. These parameters suggest a low regularization strength (C=0.01) and the use of a linear kernel for decision boundary calculation.the SVM model with a linear kernel, optimized with the chosen parameters, demonstrates high recall and accuracy, making it effective in correctly identifying positive instances, which could be crucial in scenarios where capturing all positives is essential, such as medical diagnoses or fraud detection.

# ###  Fitting a SVM classification model using rbf kernal

# defining parameter range 
param_grid = {'C': [0.1, 1, 10, 100],  
              'gamma': [1, 0.1, 0.01, 0.001, 0.0001], 
              'kernel': ['rbf']}
  
grid = GridSearchCV(SVC(), param_grid, scoring=f2_scorer, refit = True, verbose = 3, n_jobs=-1) 
  
# fitting the model for grid search 
_ = grid.fit(X_train, np.ravel(y_train))

print(grid.best_params_) 
  
# model after hyper-parameter tuning 
print(grid.best_estimator_)

y_pred = grid.predict(X_test) 

recall = recall_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
f2 = fbeta_score(y_test, y_pred, beta=2)

metrics = pd.concat([metrics, pd.DataFrame({"model": ["SVM rbf"], "Accuracy": [accuracy], "Precision": [precision], "Recall": [recall], "F1": [f1], "F2": [f2],"Parameters": [grid.best_params_]})])

metrics

# The best parameters for this SVM model with an RBF kernel are {'C': 1, 'gamma': 0.001, 'kernel': 'rbf'}. These parameters imply a moderate regularization strength (C=1) and a gamma value of 0.001, which controls the influence of individual training samples on the decision boundary.
# 
# In summary, the SVM model with an RBF kernel, optimized with the chosen parameters, demonstrates outstanding performance across all metrics

# ###  Fitting a SVM classification model using polynomial kernal

# defining parameter range 
param_grid = {'C': [0.01, 0.1, 0.5, 1, 5, 10, 50, 100],  
              'coef0': [0.01, 0.1, 0.5, 1, 5, 10, 50, 100],
              'kernel': ['poly']}
  
grid = GridSearchCV(SVC(), param_grid, scoring=f2_scorer, refit = True, verbose = 3, n_jobs=-1) 
  
# fitting the model for grid search 
_ = grid.fit(X_train, np.ravel(y_train))

print(grid.best_params_) 
  
#model after hyper-parameter tuning 
print(grid.best_estimator_)

y_pred = grid.predict(X_test) 

recall = recall_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
f2 = fbeta_score(y_test, y_pred, beta=2)

metrics = pd.concat([metrics, pd.DataFrame({"model": ["SVM Poly"], "Accuracy": [accuracy], "Precision": [precision], "Recall": [recall], "F1": [f1], "F2": [f2], "Parameters": [grid.best_params_]})])

metrics


# The best parameters for this SVM model with a polynomial kernel are {'C': 0.01, 'coef0': 5, 'kernel': 'poly'}. These parameters suggest a low regularization strength (C=0.01), a coefficient value of 5 for the polynomial kernel function, and the use of a polynomial kernel for decision boundary calculation.
# 
# In summary, the SVM model with a polynomial kernel, optimized with the chosen parameters, demonstrates exceptional performance across all metrics

metrics.sort_values(['Precision'],ascending=False)




```
**Feedback:**
Data Generation (30%) - 0.4/0.45
• Creativity and relevance of the chosen scenario - 0.15/0.15
• Correct implementation of data generation techniques - 0.15/0.15
• Clarity and structure of the data (including labeling and documentation) - 0.1/0.15

Model Implementation and Tuning (40%) - 0.6/0.6
• Correct implementation of SVC/SVR models with the three specified kernels - 0.2/0.2
• Effective use of hyper-parameter tuning, with a clear rationale for chosen parameters - 0.2/0.2
• Documentation of model performance metrics, with comparisons to justify the best-performing model - 0.2/0.2

Analysis and Discussion (30%)- 0.275/0.45
• Clarity and depth of the introduction and explanation of the chosen problem and dataset - 0.05/0.225
• Critical analysis of model results, including insights into performance variations across different kernels - 0.225/0.225

Need to create a dedicated notebook for data generation as this keeps your code more organized, no introduction section is provided and need to mention whether you choose regression or classification task in introduction or data generation notebook. So, 0.225 marks were deducted.

---

# Student 12
```python
# # Credit card approval using Support Vector Mechanism

# A synthetic data has been created based on credit card approval.
# 
# The data has 2 input variables 'income','rent'. Where 'income' is the annual income of a person and 'rent' is the rent paid by the person. And based on these 2 variables credit card approval is decided.
# 
# By using SVM, credit card approval is been predicted by using 'income' and 'rent' variables.

# # Importing the Libraries

import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score,f1_score,precision_score,recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV

sample_size=100
rent_min_val = 500
rent_max_val = 3000
income_min_val = 10000
income_max_val = 50000

np.random.seed(234)
df = pd.DataFrame({'income':np.random.randint(income_min_val,income_max_val,size=sample_size),
                   'rent':np.random.randint(rent_min_val,rent_max_val,size=sample_size)})
df.head()

# Approving credit card if Annual income is greater than 15000 USD and rent paid is greater than 1800 USD

#Approving credit card if Income is greater than $15000 and rent greater than $1800
df['approved'] = ((df['income']>15000) & (df['rent']>1800)).astype(int)
df.approved.value_counts()

# # Splitting the data for training and testing

X = df[['income','rent']]
y = df['approved']

X.head()

y.head()

x_train,x_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=234)

# The range of 'Income' variable is 10000-50000 and the range of variable 'rent' is 500-3000. So scaling must be done to get both the variables into same scale. And scaled data after spliting the data so that test data will be fresh to the model.

scaler = StandardScaler()
x_train_sc = pd.DataFrame(scaler.fit_transform(x_train),columns=['income','rent'])
x_test_sc = pd.DataFrame(scaler.transform(x_test),columns=['income','rent'])
x_train_sc.head()

# Here there is a imbalance in target variable so accuracy score wouldnt be a perfect scoring metric to evaluate the model performance. As in this case the false negative is more costly than false positive as in case of false negative the credit card company would loose potential customer. So f2 score has been took as scoring metric.

from sklearn.metrics import make_scorer
from sklearn.metrics import fbeta_score
f2_scorer = make_scorer(fbeta_score, beta=2)

# # Linear SVM

#linear model
param = {'C':[0.01,0.1,1,5,10,50,100],
             'kernel':['linear']}
grid = GridSearchCV(estimator=SVC(),param_grid=param,scoring=f2_scorer,verbose=3,n_jobs=-1)
model = grid.fit(x_train_sc,y_train)

grid.best_params_

y_pred = model.predict(x_test_sc)
accuracy = accuracy_score(y_test,y_pred)
recall = recall_score(y_test,y_pred)
precision = precision_score(y_test,y_pred)
f1 = f1_score(y_test,y_pred)
df_scores = pd.DataFrame()

df_scores =pd.concat([df_scores,pd.DataFrame({'model':['Linear SVM'],'accuracy_score':[accuracy],'recall_score':[recall],'precision':[precision],'f1_score':[f1],'parameters':[grid.best_params_]})])
df_scores

# # Polynomial SVM

#polynomial
param = {'C':[0.01,0.1,1,5,10,50,100],
         'coef0':[0.01,0.1,1,5,10,50,100],
             'kernel':['poly']}
grid = GridSearchCV(estimator=SVC(),param_grid=param,scoring=f2_scorer,verbose=3,n_jobs=-1)
model = grid.fit(x_train_sc,y_train)

grid.best_params_

y_pred = model.predict(x_test_sc)
accuracy = accuracy_score(y_test,y_pred)
recall = recall_score(y_test,y_pred)
precision = precision_score(y_test,y_pred)
f1 = f1_score(y_test,y_pred)

df_scores =pd.concat([df_scores,pd.DataFrame({'model':['poly SVM'],'accuracy_score':[accuracy],'recall_score':[recall],'precision':[precision],'f1_score':[f1],'parameters':[grid.best_params_]})])
df_scores

# # rbf SVM

#rbf
param = {'C':[0.01,0.1,1,5,10,50,100],
         'gamma':[1,0.1,0.001,0.0001],
             'kernel':['rbf']}
grid = GridSearchCV(estimator=SVC(),param_grid=param,scoring=f2_scorer,verbose=3,n_jobs=-1)
model = grid.fit(x_train_sc,y_train)

grid.best_params_

y_pred = model.predict(x_test_sc)
accuracy = accuracy_score(y_test,y_pred)
recall = recall_score(y_test,y_pred)
precision = precision_score(y_test,y_pred)
f1 = f1_score(y_test,y_pred)

df_scores =pd.concat([df_scores,pd.DataFrame({'model':['rbf SVM'],'accuracy_score':[accuracy],'recall_score':[recall],'precision':[precision],'f1_score':[f1],'parameters':[grid.best_params_]})])
df_scores

# From the above results we can see all the models are performing well. But Polynomial kernal mode is giving  1 recall score indicating overfitting. rbf kernal mode model and Linear kernal mode are performing well. Linear SVM at C=1 and rbf SVM at C=1,gamma=1 are performing well in predicting the approval of credit cards.


```
**Feedback:**
Data Generation (30%) - 0.4/0.45
• Creativity and relevance of the chosen scenario - 0.15/0.15
• Correct implementation of data generation techniques - 0.15/0.15
• Clarity and structure of the data (including labeling and documentation) - 0.1/0.15

Model Implementation and Tuning (40%) - 0.6/0.6
• Correct implementation of SVC/SVR models with the three specified kernels - 0.2/0.2
• Effective use of hyper-parameter tuning, with a clear rationale for chosen parameters - 0.2/0.2
• Documentation of model performance metrics, with comparisons to justify the best-performing model - 0.2/0.2

Analysis and Discussion (30%)- 0.35/0.45
• Clarity and depth of the introduction and explanation of the chosen problem and dataset - 0.125/0.225
• Critical analysis of model results, including insights into performance variations across different kernels - 0.225/0.225

Need to create a dedicated notebook for data generation as this keeps your code more organized and should mention whether you choose classification or regression task in introduction or data generation notebook. So, 0.15 marks were deducted.

---

# Student 13
```python
# # SVM Model Fitting

# ### Introduction
# 
# In WE03_data_gen Notebook, we generated synthetic data to simulate blood pressure and cholesterol levels in patients for predicting hypertension risk. Now, in this Notebook, we delve into model fitting, where we train and evaluate Support Vector Machine (SVM) models.
# 
# With a dataset prepared in WE03_data_gen Notebook, we are poised to explore the effectiveness of SVM models in predicting hypertension risk. Leveraging different kernels - linear, radial basis function (RBF), and polynomial - we aim to optimize model performance through hyper-parameter tuning.
# 
# In this notebook, we focus on fine-tuning the SVM models to achieve optimal predictive accuracy. By experimenting with various kernel functions and tuning parameters, we seek to identify the best-performing model for hypertension risk prediction.
# 
# Let's proceed with training and evaluating SVM models to gain insights into their predictive capabilities and contribute to advancements in healthcare analytics.

# ## 1. Setup

# Import modules

import pandas as pd
from matplotlib import pyplot as plt
import numpy as np

from sklearn.model_selection import train_test_split

from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, f1_score
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import make_scorer
from sklearn.metrics import fbeta_score

np.random.seed(1)

# ## 2. Load data

# Load data (it's already cleaned and preprocessed)

df = pd.read_csv('hypertension.csv') 
df.head()

# Use sklearn to split df into a training set and a test set

X = df[['BloodPressure', 'Cholesterol']]
y = df['Hypertension']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


# ## 3. Model the data

# First, let's create a dataframe to load the model performance metrics into.

performance = pd.DataFrame({"model": [], "Accuracy": [], "Precision": [], "Recall": [], "F1": [], "F2": [], "Parameters": []})

# create a fbeta 2 scorer
f2_scorer = make_scorer(fbeta_score, beta=2)


# ### 3.1 Fit a SVM classification model using linear kernal

# defining parameter range 
param_grid = {'C': [0.01, 0.1, 0.5, 1, 5, 10, 50, 100],  
              'kernel': ['linear']}
  
#grid = GridSearchCV(SVC(), param_grid, scoring='f1', refit = True, verbose = 3, n_jobs=-1) 

grid = GridSearchCV(SVC(), param_grid, scoring=f2_scorer, refit = True, verbose = 3, n_jobs=-1) 
  
# fitting the model for grid search 
_ = grid.fit(X_train, y_train)

# print best parameter after tuning 
print(grid.best_params_) 
  
# print how our model looks after hyper-parameter tuning 
print(grid.best_estimator_)

y_pred = grid.predict(X_test) 

recall = recall_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
f2 = fbeta_score(y_test, y_pred, beta=2)

performance = pd.concat([performance, pd.DataFrame({"model": ["SVM Linear"], "Accuracy": [accuracy], "Precision": [precision], "Recall": [recall], "F1": [f1], "F2": [f2], "Parameters": [grid.best_params_]})])


performance

# ### 3.2 Fit a SVM classification model using rbf kernal

# defining parameter range 
param_grid = {'C': [0.1, 1, 10, 100],  
              'gamma': [1, 0.1, 0.01, 0.001, 0.0001], 
              'kernel': ['rbf']}
  
grid = GridSearchCV(SVC(), param_grid, scoring=f2_scorer, refit = True, verbose = 3, n_jobs=-1) 
  
# fitting the model for grid search 
_ = grid.fit(X_train, y_train)

# print best parameter after tuning 
print(grid.best_params_) 
  
# print how our model looks after hyper-parameter tuning 
print(grid.best_estimator_)

y_pred = grid.predict(X_test) 

recall = recall_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
f2 = fbeta_score(y_test, y_pred, beta=2)

performance = pd.concat([performance, pd.DataFrame({"model": ["SVM rbf"], "Accuracy": [accuracy], "Precision": [precision], "Recall": [recall], "F1": [f1], "F2": [f2],"Parameters": [grid.best_params_]})])

performance

# ### 3.3 Fit a SVM classification model using polynomial kernal

# defining parameter range 
param_grid = {'C': [0.01, 0.1, 0.5, 1, 5, 10, 50, 100],  
              'coef0': [0.01, 0.1, 0.5, 1, 5, 10, 50, 100],
              'kernel': ['poly']}
  
grid = GridSearchCV(SVC(), param_grid, scoring=f2_scorer, refit = True, verbose = 3, n_jobs=-1) 
  
# fitting the model for grid search 
_ = grid.fit(X_train, y_train)

# print best parameter after tuning 
print(grid.best_params_) 
  
# print how our model looks after hyper-parameter tuning 
print(grid.best_estimator_)

y_pred = grid.predict(X_test) 

recall = recall_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
f2 = fbeta_score(y_test, y_pred, beta=2)

performance = pd.concat([performance, pd.DataFrame({"model": ["SVM Poly"], "Accuracy": [accuracy], "Precision": [precision], "Recall": [recall], "F1": [f1], "F2": [f2], "Parameters": [grid.best_params_]})])

performance

# ## 4.0 Summary

performance.sort_values(by="F2", ascending=False)

# From our analysis, we can see that the SVM models with polynomial, rbf, and linear kernels exhibit varying performance metrics.
# 
# - The SVM model with a polynomial kernel achieved perfect accuracy, precision, recall, F1 score, and F2 score, indicating flawless classification performance. It was trained with a high C value of 100 and a coefficient of 50.
# 
# - The SVM model with an rbf kernel also demonstrated excellent performance, achieving a high accuracy of 96%. While its precision and recall were slightly lower than the polynomial model, it still maintained a high F1 and F2 score. This model was trained with a C value of 0.1 and a gamma value of 0.001.
# 
# - The SVM model with a linear kernel, although slightly less accurate than the other two models, still performed well with an accuracy of 91%. It exhibited high precision, recall, and F1 score, indicating good overall classification performance. This model was trained with a C value of 50.
# 
# In summary, all three SVM models show promising results in classifying hypertension risk based on blood pressure and cholesterol levels. The polynomial kernel model achieved perfect performance, while the rbf and linear kernel models also performed admirably with high accuracy and balanced precision and recall. Further validation and testing are recommended to assess the models' generalizability and robustness.



# # Exploring Hypertension Prediction Using SVM:
#     
# ## Introduction
# 
# In this healthcare analytics project, our focus is on predicting hypertension risk in patients, a prevalent medical condition associated with severe health implications. Hypertension, commonly known as high blood pressure, is a silent yet potentially life-threatening condition that affects millions worldwide. Early detection and intervention are crucial for managing hypertension and preventing complications such as heart disease, stroke, and kidney failure.
# 
# To address this challenge, we harness the power of machine learning techniques, particularly Support Vector Machines (SVM), to develop a predictive model. SVMs are well-suited for classification tasks and offer robust performance even with limited data. By leveraging SVMs, we aim to create a reliable tool that healthcare professionals can use to identify individuals at risk of hypertension before the onset of symptoms.
# 
# Our approach involves generating synthetic data to simulate blood pressure and cholesterol levels in a sample of patients. These two input features are crucial indicators of hypertension risk and are widely used in clinical practice for assessing cardiovascular health. Additionally, we include an output column indicating whether each patient is classified as high-risk or low-risk for hypertension based on predefined thresholds.
# 
# Once we have generated the synthetic dataset, we will proceed with hyper-parameter tuning to test and fit three model forms of SVM: linear, radial basis function (RBF), and polynomial kernels. Hyper-parameter tuning allows us to optimize the performance of each SVM model by fine-tuning parameters such as the regularization parameter (C) and kernel-specific parameters.
# 
# Throughout our analysis, we will evaluate the performance of each SVM model using metrics such as accuracy, precision, recall, and F1-score. We will also visualize the results and discuss the implications of our findings for hypertension risk prediction in clinical practice.
# 
# By the end of this project, we aim to provide insights into the effectiveness of SVM models for predicting hypertension risk and demonstrate the potential of machine learning in improving patient outcomes in healthcare.

# ## Data Preparation

# ### 1. Set up
# Import necessary libraries

# import necessary libraries
import pandas as pd
import numpy as np

# set random seed to ensure that results are repeatable
np.random.seed(1)

# ### 2. Synthesize the data
# Let's set the number of observations that we will synthesize.

sample_size = 500

# Generate synthetic data for blood pressure and cholesterol levels

# Generating features (blood pressure and cholesterol levels)
blood_pressure = np.random.randint(90, 180, sample_size)
cholesterol = np.random.randint(120, 300, sample_size)


# Assign hypertension labels based on predefined thresholds, indicating whether a patient has hypertension or not.

# Generating labels (0: no hypertension, 1: hypertension)
hypertension = np.where((blood_pressure > 140) | (cholesterol > 200), 1, 0)


# ### 3. Create Dataframe
# Since our goal is to generate data that we can analyze with another notebook (for practice), let's save this data to a csv.
# 
# First we will create a dataframe with the data we just similated...

# Creating DataFrame
df = pd.DataFrame({'BloodPressure': blood_pressure, 'Cholesterol': cholesterol, 'Hypertension': hypertension})


df.head()

# ### 3. Add some random noise to the model
# Now, let's obscure the model by adding noise to the data by adding errors that are randomly selected from a norma distribution

# Adding noise to 'blood_pressure' and 'cholesterol' columns
df['BloodPressure'] = df['BloodPressure'] + np.random.uniform(-5, 5, sample_size)
df['Cholesterol'] = df['Cholesterol'] + np.random.uniform(-10, 10, sample_size)

# Display the modified DataFrame
print(df.head())


# ### 4. Exploration of the data
# We have a two input variables and one target variable. For this analysis, the target variable is hypertension.
# 
# Let's explore our data...

# generate a basic summary of the data
df.info()

# generate a statistical summary of the numeric value in the data
df.describe()

# ### 5. Save the data frame to a csv
# Lastly, let's save the data we created to a csv file. This saved data will be used to practice svm models on data.

df.to_csv('hypertension.csv', index=False)




```
**Feedback:**
Data Generation (30%) - 0.45/0.45
• Creativity and relevance of the chosen scenario - 0.15/0.15
• Correct implementation of data generation techniques - 0.15/0.15
• Clarity and structure of the data (including labeling and documentation) - 0.15/0.15

Model Implementation and Tuning (40%) - 0.6/0.6
• Correct implementation of SVC/SVR models with the three specified kernels - 0.2/0.2
• Effective use of hyper-parameter tuning, with a clear rationale for chosen parameters - 0.2/0.2
• Documentation of model performance metrics, with comparisons to justify the best-performing model - 0.2/0.2

Analysis and Discussion (30%)- 0.45/0.45
• Clarity and depth of the introduction and explanation of the chosen problem and dataset - 0.225/0.225
• Critical analysis of model results, including insights into performance variations across different kernels - 0.225/0.225

---

# Student 14
```python
# ## WE03 

# ### Importing Libraries

import pandas as pd
from sklearn.svm import SVC
from matplotlib import pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

np.random.seed(1)

# ### Getting data from csv file

df = pd.read_csv("C:/Users/hrush/OneDrive/Desktop/DSP/Week - 3/mobile_app_subscription_conversion_2.csv") 
df.head(6)

# ### Train/Test Split of Data

# Use sklearn to split df into a training set and a test set

X = df[['NumofFreeTrialDays','AppUsageFrequency']]
y = df['ConvertedToSubscription']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)


# ### Dataframe to store different values of SVM models

demo_svm = pd.DataFrame({"model": [], "Accuracy": [], "Precision": [], "Recall": [], "F1": [], "F2": [], "Parameters": []})

# ### Creating a fbeta 2 scorer


from sklearn.metrics import make_scorer
from sklearn.metrics import fbeta_score
f2_scorer = make_scorer(fbeta_score, beta=2)


# ### SVM classification model using linear kernal

# defining parameter range 
param_grid = {'C': [0.01, 0.1, 0.5, 1, 5, 10, 50, 100],  
              'kernel': ['linear']}
  
#grid = GridSearchCV(SVC(), param_grid, scoring='f1', refit = True, verbose = 3, n_jobs=-1) 

grid = GridSearchCV(SVC(), param_grid, scoring=f2_scorer, refit = True, verbose = 3, n_jobs=-1) 
  
# fitting the model for grid search 
_ = grid.fit(X_train, y_train)

# print best parameter after tuning 
print(grid.best_params_) 
  
# print how our model looks after hyper-parameter tuning 
print(grid.best_estimator_)

y_pred = grid.predict(X_test) 

recall = recall_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
f2 = fbeta_score(y_test, y_pred, beta=2)

demo_svm = pd.concat([demo_svm, pd.DataFrame({"model": ["SVM Linear"], "Accuracy": [accuracy], "Precision": [precision], "Recall": [recall], "F1": [f1], "F2": [f2], "Parameters": [grid.best_params_]})])


# ### Displaying result

demo_svm


# ### SVM classification model using rbf kernal

# defining parameter range 
param_grid = {'C': [0.1, 1, 10, 100,200,300],  
              'gamma': [1, 0.1, 0.01, 0.001, 0.0001], 
              'kernel': ['rbf']}
  
grid = GridSearchCV(SVC(), param_grid, scoring=f2_scorer, refit = True, verbose = 3, n_jobs=-1) 
  
# fitting the model for grid search 
_ = grid.fit(X_train, y_train)

# print best parameter after tuning 
print(grid.best_params_) 
  
# print how our model looks after hyper-parameter tuning 
print(grid.best_estimator_)

y_pred = grid.predict(X_test) 

recall = recall_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
f2 = fbeta_score(y_test, y_pred, beta=2)

demo_svm = pd.concat([demo_svm, pd.DataFrame({"model": ["SVM rbf"], "Accuracy": [accuracy], "Precision": [precision], "Recall": [recall], "F1": [f1], "F2": [f2],"Parameters": [grid.best_params_]})])

# ### SVM classification model using polynomial kernal

# defining parameter range 
param_grid = {'C': [0.01, 0.1, 0.5, 1, 5, 10, 50, 100],  
              'coef0': [0.01, 0.1, 0.5, 1, 5, 10, 50, 100,200,300],
              'kernel': ['poly']}
  
grid = GridSearchCV(SVC(), param_grid, scoring=f2_scorer, refit = True, verbose = 3, n_jobs=-1) 
  
# fitting the model for grid search 
_ = grid.fit(X_train, y_train)

# print best parameter after tuning 
print(grid.best_params_) 
  
# print how our model looks after hyper-parameter tuning 
print(grid.best_estimator_)

y_pred = grid.predict(X_test) 

recall = recall_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
f2 = fbeta_score(y_test, y_pred, beta=2)

demo_svm= pd.concat([demo_svm, pd.DataFrame({"model": ["SVM Poly"], "Accuracy": [accuracy], "Precision": [precision], "Recall": [recall], "F1": [f1], "F2": [f2], "Parameters": [grid.best_params_]})])

# ### Displaying results from all 3 SVM models

demo_svm



# ### Sorting the results by ascending order

demo_svm.sort_values(by="F2", ascending=False)

# ### Summary
# 
# From the results we can see that F2 score for SVM rbf and SVM Poly is highest with 82.7% thereby indicating it's better suited for this data as recall is preferred  as subscription models focuses more on false negatives. A balanced dataset was selected so that performance metrics are meaningful and model doesn't favour one class over the other just because of class distribution.The three different SVM models were evaluated using a grid search approach to optimize parameters: linear kernel, radial basis function (RBF) kernel, and polynomial kernel. Both rbf and poly got perfect recall scores as they identified all positive cases at the cost of precision showing high number of false positives. SVM linear performed poorly among all three models.
# 
# The focus is to identify users who are likely to convert from the free trial to subscription. As for business models having subscription models as their primary source of revenue retaining existing customers and converting trial users to subscribers is crucial for business growth.
# A False negative for this scenario is that a model wrongly predicts a user will not convert when they would have converted.It is more costly as it shows a missed opportunity for revenue. Therefore recall has to be maximized that means identifying all actual conversions become crucial.
# 
# 
# 
# 

# ## WE03 SVM

# ## Introduction

# I have created a dataset for binary classification and it's about whether an user would subscribe to an app based on the usage frequency and number of trial days they participated. With class '1' being converted to subscription and class '0' indicating not converted. The focus is on the F2 score, to prioritize minimizing false negatives, crucial for subscription-based business models.

# ### Importing Libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


np.random.seed(1)

# ### Generating Data

# Generate sample data
sample_size = 300
free_trial = np.random.randint(7, 31, size=sample_size)  # Randomly choose between 7 to 30 days
app_usage = np.random.randint(1, 5, size=sample_size)  # Randomly choose between 1 to 4 times

# ### Generate the final subscription status where 1 represents subscription and 0 represents no subscription

final_subscription = np.array([1] * (sample_size // 2) + [0] * (sample_size // 2))
np.random.shuffle(final_subscription)  # Shuffle to mix conversion statuses

# ### Dataframe to hold generated data

# Create DataFrame
df = pd.DataFrame({
    'NumofFreeTrialDays': free_trial,
    'AppUsageFrequency': app_usage,
    'ConvertedToSubscription': final_subscription
})

# ### Displaying Dataframe

# Display the DataFrame
print(df)

# ### Storing the data to csv file

# Save the DataFrame to a CSV file
df.to_csv('mobile_app_subscription_conversion_2.csv', index=False)

# ### Distribution of classes

# Print distribution of 'ConvertedToSubscription'
print(df['ConvertedToSubscription'].value_counts())

# ### Visualize the distribution of 'ConvertedToSubscription' using a bar plot


df['ConvertedToSubscription'].value_counts().plot(kind='bar')
plt.title('Distribution of ConvertedToSubscription')
plt.xlabel('Subscription Conversion')
plt.ylabel('Count')
plt.xticks([0, 1], ['Not Converted (0)', 'Converted (1)'], rotation=0)
plt.show()


```
**Feedback:**
Data Generation (30%) - 0.45/0.45
• Creativity and relevance of the chosen scenario - 0.15/0.15
• Correct implementation of data generation techniques - 0.15/0.15
• Clarity and structure of the data (including labeling and documentation) - 0.15/0.15

Model Implementation and Tuning (40%) - 0.6/0.6
• Correct implementation of SVC/SVR models with the three specified kernels - 0.2/0.2
• Effective use of hyper-parameter tuning, with a clear rationale for chosen parameters - 0.2/0.2
• Documentation of model performance metrics, with comparisons to justify the best-performing model - 0.2/0.2

Analysis and Discussion (30%)- 0.45/0.45
• Clarity and depth of the introduction and explanation of the chosen problem and dataset - 0.225/0.225
• Critical analysis of model results, including insights into performance variations across different kernels - 0.225/0.225

---

# Student 15
```python
# #### Importing Libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, recall_score
from sklearn.metrics import precision_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler

np.random.seed(1)

# ####  Create a dataframe: Age and income

sample_size = 40 
min_age = 28
max_age = 50 
income_min = 40000 
income_max = 80000

df = pd.DataFrame({'age': np.linspace(min_age,max_age,sample_size).astype(int),'income': np.linspace(income_min,income_max,sample_size).astype(int)})
df

# The DataFrame has two columns: 'age' and 'income'. The 'age' column contains 40 evenly spaced values between 28 and 50, and the 'income' column contains 40 evenly spaced values between 40000 and 80000. The DataFrame has 40 rows, with each row representing a unique combination of age and income. The 'age' and 'income' columns are of type integer.

# #### Add a 'Loan Approved' column to the DataFrame with random values

df['Loan Approved'] = ((df['age'] > 38 ) |  (df['income'] > 52000)).astype(int)
df

# #### Add random noise to the 'income' column

df['income'] = df['income'] + np.random.uniform(-40,40, sample_size)

# #### Plot a scatter plot 

plt.figure(figsize=(10, 9))
plt.scatter(df['age'], df['income'], c=df['Loan Approved'], cmap='cool', edgecolors='k', alpha=0.8, s=100)
plt.scatter(df['age'], df['income'], c='red', marker='o', label='Data Points')
plt.colorbar(label='Loan Approved')
plt.title('Scatter Plot of age vs income')
plt.xlabel('age')
plt.ylabel('income')
plt.legend()
plt.grid(True)
plt.show()

# The resulting plot shows the relationship between 'age' and 'income' for the data points in the DataFrame, with the color of each data point corresponding to the value of the 'Loan Approved' column. The plot also includes a color bar, title, axis labels, legend, and grid to help interpret the data

# #### Save the DataFrame to a CSV file

df.to_csv('loan_approved.csv', index=False)

# ###  Split the data into a training set and a test set

# Use sklearn to split df into a training set and a test set

X = df.iloc[:,:2]
y = df['Loan Approved']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=1)


# ### Modeling the Data

performance = pd.DataFrame({"model": [], "Accuracy": [], "Precision": [], "Recall": [], "F1": [], "F2": [], "Parameters": []})

# generate a fbeta 2 scorer
from sklearn.metrics import make_scorer
from sklearn.metrics import fbeta_score
f2_scorer = make_scorer(fbeta_score, beta=2)

# ### 3.1 Fit a SVM classification model using linear kernal

# defining parameter range 
param_grid = {'C': [0.01, 0.1, 0.5, 1, 5, 10, 50, 100],  
              'kernel': ['linear']}
  

grid = GridSearchCV(SVC(), param_grid, scoring=f2_scorer, refit = True, verbose = 3, n_jobs=-1) 
  
# fitting the model for grid search 
_ = grid.fit(X_train, y_train)

# #### Hyperparameter Tuning and Model Evaluation for SVM with Linear Kernel

# print best parameter after tuning 
print(grid.best_params_) 
  
# print how our model looks after hyper-parameter tuning 
print(grid.best_estimator_)

y_pred = grid.predict(X_test) 

recall = recall_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
f2 = fbeta_score(y_test, y_pred, beta=2)

performance = pd.concat([performance, pd.DataFrame({"model": ["SVM Linear"], "Accuracy": [accuracy], "Precision": [precision], "Recall": [recall], "F1": [f1], "F2": [f2], "Parameters": [grid.best_params_]})])


# ### 3.2 Fit a SVM classification model using rbf kernal

# defining parameter range 
param_grid = {'C': [0.1, 1, 10, 100],  
              'gamma': [1, 0.1, 0.01, 0.001, 0.0001], 
              'kernel': ['rbf']}
  
grid = GridSearchCV(SVC(), param_grid, scoring=f2_scorer, refit = True, verbose = 3, n_jobs=-1) 
  
_ = grid.fit(X_train, y_train)

# #### Perform the grid search to find the best hyperparameters

# print best parameter after tuning 
print(grid.best_params_) 
  
# print how our model looks after hyper-parameter tuning 
print(grid.best_estimator_)

y_pred = grid.predict(X_test) 

recall = recall_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
f2 = fbeta_score(y_test, y_pred, beta=2)

performance = pd.concat([performance, pd.DataFrame({"model": ["SVM rbf"], "Accuracy": [accuracy], "Precision": [precision], "Recall": [recall], "F1": [f1], "F2": [f2],"Parameters": [grid.best_params_]})])

# ### 3.3 Fit a SVM classification model using polynomial kernal

# defining parameter range 
param_grid = {'C': [0.01, 0.1, 0.5, 1, 5, 10, 50, 100],  
              'coef0': [0.01, 0.1, 0.5, 1, 5, 10, 50, 100],
              'kernel': ['poly']}
  
grid = GridSearchCV(SVC(), param_grid, scoring=f2_scorer, refit = True, verbose = 3, n_jobs=-1) 
  
# fitting the model for grid search 
_ = grid.fit(X_train, y_train)

# #### Hyperparameter Tuning and Model Evaluation for SVM with Polynomial Kernel

# print best parameter after tuning 
print(grid.best_params_) 
  
# print how our model looks after hyper-parameter tuning 
print(grid.best_estimator_)

y_pred = grid.predict(X_test) 

recall = recall_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
f2 = fbeta_score(y_test, y_pred, beta=2)

performance = pd.concat([performance, pd.DataFrame({"model": ["SVM Poly"], "Accuracy": [accuracy], "Precision": [precision], "Recall": [recall], "F1": [f1], "F2": [f2], "Parameters": [grid.best_params_]})])

# ## 4.0 Summary

# #### Performance Metrics for SVM Models with Hyperparameter Tuning

performance.sort_values(by="F2", ascending=False)

# The resulting sorted DataFrame shows that the SVM Linear model has the highest F2 score, followed by the SVM Poly model, and then the SVM rbf model. The Parameters column shows the best hyperparameters found for each model during the grid search. The resulting sorted DataFrame can be used to compare the performance of different models and to select the best model based on the desired performance metric.

# Model: This column indicates the type of model used for classification. In this case, SVM Linear, SVM Poly, and SVM rbf are the models being evaluated.
# 
# Accuracy: This column shows the accuracy of each model on the test data. Accuracy measures the proportion of correctly classified instances out of the total instances.
# 
# Precision: Precision is the ratio of correctly predicted positive observations to the total predicted positive observations. It indicates how many of the predicted positive instances are actually positive.
# 
# Recall: Recall, also known as sensitivity or true positive rate, measures the proportion of correctly predicted positive observations out of all actual positives. It indicates the model's ability to correctly identify positive instances.
# 
# F1 Score: The F1 score is the harmonic mean of precision and recall. It provides a balance between precision and recall, with values closer to 1 indicating better performance.
# 
# F2 Score: The F2 score is a variant of the F1 score that weighs recall higher than precision. It is useful when recall is more important than precision, such as in cases where false negatives are more concerning than false positives.
# 
# Parameters: This column lists the hyperparameters used for each model. These hyperparameters were determined to be the best after tuning using techniques such as grid search or random search.
# 
# Overall, the table provides a summary of the performance of different SVM models with their corresponding hyperparameters. It allows for easy comparison of their performance metrics to determine which model performs best for the given classification task


```
**Feedback:**
Data Generation (30%) - 0.4/0.45
• Creativity and relevance of the chosen scenario - 0.15/0.15
• Correct implementation of data generation techniques - 0.15/0.15
• Clarity and structure of the data (including labeling and documentation) - 0.1/0.15

Model Implementation and Tuning (40%) - 0.6/0.6
• Correct implementation of SVC/SVR models with the three specified kernels - 0.2/0.2
• Effective use of hyper-parameter tuning, with a clear rationale for chosen parameters - 0.2/0.2
• Documentation of model performance metrics, with comparisons to justify the best-performing model - 0.2/0.2

Analysis and Discussion (30%)- 0.275/0.45
• Clarity and depth of the introduction and explanation of the chosen problem and dataset - 0.05/0.225
• Critical analysis of model results, including insights into performance variations across different kernels - 0.225/0.225

Need to create a dedicated notebook for data generation as this keeps your code more organized, no introduction section is provided and need to mention whether you choose regression or classification task in introduction or data generation notebook. So, 0.225 marks were deducted.

---

# Student 16
```python
# ## Customer Purchase data generation

# Predicting whether a customer will make a purchase or not is the main goal of customer purchase prediction. This dataset contains 2 features i.e., "Time Spent [minutes]" (Time spent on the website/app), "Previous Purchases" (purchase history) and target variable is "Likelihood of Purchase". Target variable is defined like based on random probability, if random probability is more than 30%, then its assumed as likely to buy or else Not.

# import numpy and pandas
import numpy as np
import pandas as pd

np.random.seed(1)

time_spent = np.random.randint(5, 120, 300)  # Random time spent in minutes (between 5 and 120)
previous_purchases = np.random.randint(0, 10, 300)  # Random previous purchases (between 0 and 10)

# Generating the target variable based on a random probability, and assumed if probability more than 30% then they customer willing to purchase and assumed it as 'YES' or else 'NO'
purchase_probability = np.random.rand(300)
likelihood_of_purchase = ['Yes' if p > 0.3 else 'No' for p in purchase_probability]

data = {
    'Time Spent [minutes]': time_spent,
    'Previous Purchases': previous_purchases,
    'Likelihood of Purchase': likelihood_of_purchase
}
df = pd.DataFrame(data)
df.head()

#Data looks good, no null/missing values
df.isna().sum()



df['Likelihood of Purchase'].value_counts()

# generated data looks unbalanced the reason being, it is synthetic data

df.to_csv('customer_purchase_prediction.csv', index=False)

# # SVM Models for Customer Purchase Prediction

# ## 1. Setup

# import numpy and pandas and sklearn libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from sklearn.preprocessing import LabelEncoder

# create a fbeta 2 scorer
from sklearn.metrics import make_scorer
from sklearn.metrics import fbeta_score

np.random.seed(1)

# ## 2. Load data

#Loading data from csv and assigned to data variable.
data = pd.read_csv("customer_purchase_prediction.csv")

#To print 5 records
data.head()

#Generate descriptive statistics
data.describe()

#Checking if there are any records with no data
data.isna().sum()

#Spliting input(X) and target(y) features
X=data[['Time Spent [minutes]','Previous Purchases']]
y=data['Likelihood of Purchase']

# Create LabelEncoder object
label_encoder = LabelEncoder()
# Fit and transform the target variable
y_encoded = label_encoder.fit_transform(y)

y_encoded

# splitting the data into testing and training set. Test data is 20% and training data is 80%
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2)

# ## 3. Model the data

f2_scorer = make_scorer(fbeta_score, beta=2)

# ### 3.1 Fit a SVM classification model using linear kernal

# defining parameter range 
param_grid = {'C': [0.001, 0.1, 0.5, 1, 5, 10, 50,  100], 'kernel': ['linear']}

grid = GridSearchCV(SVC(), param_grid, scoring=f2_scorer, refit = True, verbose = 3, n_jobs=-1)

# fitting the model for grid search 
grid.fit(X_train, y_train)

y_pred = grid.predict(X_test)
recall = recall_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
f2 = fbeta_score(y_test, y_pred, beta=2)

linear = pd.DataFrame({
    "model": ["SVM Linear"],
    "Accuracy": [accuracy],
    "Precision": [precision],
    "Recall": [recall],
    "F1": [f1],
    "F2": [f2],
    "Parameters": [grid.best_params_]
})

linear.head()

# ### 3.2 Fit a SVM classification model using rbf kernal

# defining parameter range 
param_grid = {'C': [0.1, 1, 10, 100],
              'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
              'kernel': ['rbf']}

grid = GridSearchCV(SVC(), param_grid, scoring=f2_scorer, refit = True, verbose = 3, n_jobs=-1)

# fitting the model for grid search 
grid.fit(X_train, y_train)

y_pred = grid.predict(X_test)
recall = recall_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
f2 = fbeta_score(y_test, y_pred, beta=2)

rbf = pd.DataFrame({
    "model": ["SVM rbf"],
    "Accuracy": [accuracy],
    "Precision": [precision],
    "Recall": [recall],
    "F1": [f1],
    "F2": [f2],
    "Parameters": [grid.best_params_]
})
rbf.head()

# ### 3.3 Fit a SVM classification model using polynomial kernal

# defining parameter range 
param_grid = {'C': [0.01, 0.1, 0.5, 1, 5, 10, 50, 100],
              'coef0': [0.01, 0.1, 0.5, 1, 5, 10, 50, 100],
              'kernel': ['poly']}

grid = GridSearchCV(SVC(), param_grid, scoring=f2_scorer, refit = True, verbose = 3, n_jobs=-1)

# fitting the model for grid search 
grid.fit(X_train, y_train)

y_pred = grid.predict(X_test)
recall = recall_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
f2 = fbeta_score(y_test, y_pred, beta=2)

poly = pd.DataFrame({
    "model": ["SVM poly"],
    "Accuracy": [accuracy],
    "Precision": [precision],
    "Recall": [recall],
    "F1": [f1],
    "F2": [f2],
    "Parameters": [grid.best_params_]
})

linear.head()

# ## 4.0 Summary

pd.concat([linear,rbf,poly], axis=0)

# ## Analysis
# 
# ### SVM Linear:
# As all positive incidents (clients who made a purchase) were accurately identified, the linear SVM model achieved 100% recall. The precision score, however, is somewhat poor, indicating that a significant number of the cases that were marked as positive were actually false positives, because its synthetic and imbalanced data. It indicates that although the model accurately determines which customers are most likely to make a purchase, it also incorrectly labels users who do not intend to buy as potential customers.
# 
# ### SVM Radial Basis Function:
# The performance of the linear SVM model and the radial basis function kernel SVM model was similar. Perfect recall was achieved, but precision was less, indicating that misunderstanding non-purchasing customers is a like issue.
# 
# ### SVM poly
# With perfect recall but less precision, the polynomial kernel SVM model produced results that were in line with the linear and rbf models.
# 
# ## Discussion
# 
# Since the F2 score gives priority to recall, a critical factor in identifying potential customers in imbalanced datasets. It was selected for use in customer purchase prediction. It emphasizes reducing false negatives more, which is in line with the objective of capturing all possible customers while taking accuracy into account.
# 
# The three SVM kernels (linear, RBF, and polynomial) performs the same way when it comes to accuracy, precision, recall, F1 score, and F2 score. This consistency shows that, in this case, the model performance is not much affected by the type of kernel selected. The models accurately record every customer purchase scenarios, as indicated by their 100% recall score. The precision score of 66.6%, however, indicates that a significant proportion of expected positive cases might be false positives. The accuracy of 66.6% seems appropriate; however, the models' F2 score of 91% indicates that recall is prioritized over precision. This approach aligns with the goal of consumer purchase prediction, which places a premium on reducing false negatives. Changes are present in the parameter configurations of each SVM kernel, resulting in distinct values for 'C', 'gamma', and 'coef0'. To maximize the effectiveness of the model and its ability to generalize, more research into parameter adjustment may be necessary.
# 
# To sum up, even though the SVM models show good predictive ability in locating possible customers, efforts should focus on improving accuracy for more precise targeting and resource distribution in customer buying predictions.


```
**Feedback:**
Data Generation (30%) - 0.45/0.45
• Creativity and relevance of the chosen scenario - 0.15/0.15
• Correct implementation of data generation techniques - 0.15/0.15
• Clarity and structure of the data (including labeling and documentation) - 0.15/0.15

Model Implementation and Tuning (40%) - 0.6/0.6
• Correct implementation of SVC/SVR models with the three specified kernels - 0.2/0.2
• Effective use of hyper-parameter tuning, with a clear rationale for chosen parameters - 0.2/0.2
• Documentation of model performance metrics, with comparisons to justify the best-performing model - 0.2/0.2

Analysis and Discussion (30%)- 0.35/0.45
• Clarity and depth of the introduction and explanation of the chosen problem and dataset - 0.125/0.225
• Critical analysis of model results, including insights into performance variations across different kernels - 0.225/0.225

Need to mention whether you choose regression or classification task in introduction or data generation notebook. So, 0.1 marks were deducted.

---

# Student 17
```python
# ##                                                                 Loan Approval Model 

# #### In this notebook, we will synthesize data with two inputs and one target namely 'Income in US Dollars', 'Credit Score' and 'Loan Approved'. We will try to fit this data to SVC/SVR with the three kernels discussed in class - linear, poly, and rbf.

# ### Step 1 Importing necessary libraries.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, recall_score, accuracy_score, f1_score, precision_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler



#  Seed is set to 1 for reproducibility in order to obtain consistent results

np.random.seed(1)

# ### 2.1 - Synthesize data 

# controlled variables are 'Income in US Dollars', and 'Credit Score' values ranges from 300(poor credit) to 850(excellent credit).

#number of observations
sample_size = 30

#minimum income 
income_min = 0

#maximum income
income_max = 350000

# minimum credit score
credit_score_min = 300

#maximum credit score
credit_score_max = 850

df = pd.DataFrame({'Income in US Dollars': np.linspace(income_min,income_max,sample_size).astype(int),'Credit Score': np.linspace(credit_score_min,credit_score_max,sample_size).astype(int)})
df

# here, the logic may seem very simple and unrealistic. I have come up with this approach as I am limited by the number of inputs which control the target
df['Loan Approved'] = ((df['Credit Score'] > 690) |  (df['Income in US Dollars'] > 100000)).astype(int)
df

# ### 2.2 Adding random noise to the 'Income in US Dollars' column for variability to make it real

df['Income in US Dollars'] = df['Income in US Dollars'] + np.random.uniform(-25,25, sample_size)

df['Income in US Dollars'] = df['Income in US Dollars'].astype(int)
df


# ### 2.3 Visualizing the data 

#using scatter plot to understand the relationship between 'Income in US Dollars', 'Credit Score' and 'Loan Approved'
plt.figure(figsize=(8, 7))
plt.scatter(df['Income in US Dollars'], df['Credit Score'], c=df['Loan Approved'], cmap='cool', edgecolors='k', alpha=0.8, s=100)
plt.scatter(df['Income in US Dollars'], df['Credit Score'], c='blue', marker='o', label='Data Points')
plt.colorbar(label='Loan Approved')
plt.title('Scatter Plot of Income in US Dollars vs. Credit Score')
plt.xlabel('Income in US Dollars')
plt.ylabel('Credit Score')
plt.legend()
plt.grid(True)
plt.show()

# #### Saving the synthesized data to a CSV file 'Loan Approved.csv'

df.to_csv('approved_for_loan.csv', index=False)

# ### 2.4 Train and Split Data

# Using sklearn to split df into a training set and a test set

#Input
X = df.iloc[:,:2]

#Target
y = df['Loan Approved']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)


# ### Step 3 Modeling the Data

# Dataframe to view all evaluation metrics in one view
performance = pd.DataFrame({"model": [], "Accuracy": [], "Precision": [], "Recall": [], "F1": [], "F2": [], "Parameters": []})

# generate a fbeta 2 scorer
from sklearn.metrics import make_scorer
from sklearn.metrics import fbeta_score
f2_scorer = make_scorer(fbeta_score, beta=2)

# ### 3.1 Fit a SVM classification model using linear kernel

# defining parameter range
 # C value represents regularization. Higher the C value, lower the regularization.
param_grid = {'C': [0.01, 0.1, 0.5, 1, 5, 10, 50, 100],  
              'kernel': ['linear']}
  

grid = GridSearchCV(SVC(), param_grid, scoring=f2_scorer, refit = True, verbose = 3, n_jobs=-1) 
  
# fitting the model for grid search 
_ = grid.fit(X_train, y_train)

# #### Tuning the SVM model and predicting the test set data

# print best parameter after tuning 
print(grid.best_params_) 
  
# print how our model looks after hyper-parameter tuning 
print(grid.best_estimator_)

y_pred = grid.predict(X_test) 

# calculating performance metrics to evaluate
recall = recall_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
f2 = fbeta_score(y_test, y_pred, beta=2)

performance = pd.concat([performance, pd.DataFrame({"model": ["SVM Linear"], "Accuracy": [accuracy], "Precision": [precision], "Recall": [recall], "F1": [f1], "F2": [f2], "Parameters": [grid.best_params_]})])


# ### 3.2 Fit a SVM classification model using rbf(Radial Basis Function) kernel

# defining parameter range 
# C value represents regularization. Higher the C value, lower the regularization.
param_grid = {'C': [0.1, 1, 10, 100],  
              'gamma': [1, 0.1, 0.01, 0.001, 0.0001], 
              'kernel': ['rbf']}
  
grid = GridSearchCV(SVC(), param_grid, scoring=f2_scorer, refit = True, verbose = 3, n_jobs=-1) 
  
_ = grid.fit(X_train, y_train)

# #### Targeting for the best parameters by printing them after tuning. After hyperparameter tuning, storing the performance metrics of SVM(rbf Kernel).

# print best parameter after tuning 
print(grid.best_params_) 
  
# observing how the model looks post hyper-parameter tuning 
print(grid.best_estimator_)

y_pred = grid.predict(X_test) 

recall = recall_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
f2 = fbeta_score(y_test, y_pred, beta=2)

performance = pd.concat([performance, pd.DataFrame({"model": ["SVM rbf"], "Accuracy": [accuracy], "Precision": [precision], "Recall": [recall], "F1": [f1], "F2": [f2],"Parameters": [grid.best_params_]})])

# ### 3.3 Fit a SVM classification model using polynomial kernel

# defining parameter range 
param_grid = {'C': [0.01, 0.1, 0.5, 1, 5, 10, 50, 100],  
              'coef0': [0.01, 0.1, 0.5, 1, 5, 10, 50, 100],
              'kernel': ['poly']}
  
grid = GridSearchCV(SVC(), param_grid, scoring=f2_scorer, refit = True, verbose = 3, n_jobs=-1) 
  
# fitting the model for grid search 
_ = grid.fit(X_train, y_train)

# #### Appending the evalution metrics to the dataframe post tuning the model,  

# print best parameter after tuning 
print(grid.best_params_) 
  
# print how our model looks after hyper-parameter tuning 
print(grid.best_estimator_)

y_pred = grid.predict(X_test) 

recall = recall_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
f2 = fbeta_score(y_test, y_pred, beta=2)

performance = pd.concat([performance, pd.DataFrame({"model": ["SVM Poly"], "Accuracy": [accuracy], "Precision": [precision], "Recall": [recall], "F1": [f1], "F2": [f2], "Parameters": [grid.best_params_]})])

# ### 4.0 Summary

# #### Here the 'performance' dataframe is been sorted by the F2 column in descending order

performance.sort_values(by="F2", ascending=False)

# #### In summary, the linear kernel SVM model excelled in all evaluation metrics, including accuracy, precision, recall, f1, and f2, achieving a perfect 100% score. The polynomial kernel, while demonstrating perfect precision (100%), fell short in other performance areas. The RBF kernel with parameters {'C':0.01} maintained perfect recall  (100%), but suffered in accuracy (88.88%) had excellent score in f1 (94.1%) and f2(97.5%). SVM with linear kernel seems to be a balanced option for this classification task.


```
**Feedback:**
Data Generation (30%) - 0.4/0.45
• Creativity and relevance of the chosen scenario - 0.15/0.15
• Correct implementation of data generation techniques - 0.15/0.15
• Clarity and structure of the data (including labeling and documentation) - 0.1/0.15

Model Implementation and Tuning (40%) - 0.6/0.6
• Correct implementation of SVC/SVR models with the three specified kernels - 0.2/0.2
• Effective use of hyper-parameter tuning, with a clear rationale for chosen parameters - 0.2/0.2
• Documentation of model performance metrics, with comparisons to justify the best-performing model - 0.2/0.2

Analysis and Discussion (30%)- 0.35/0.45
• Clarity and depth of the introduction and explanation of the chosen problem and dataset - 0.125/0.225
• Critical analysis of model results, including insights into performance variations across different kernels - 0.225/0.225

Need to create a dedicated notebook for data generation as this keeps your code more organized and should mention whether you choose classification or regression task in introduction or data generation notebook. So, 0.15 marks were deducted.

---

# Student 18
```python
# In this analysis, we're exploring the relationship between age and weight to determine if there's a correlation. We've employed Support Vector Machines (SVMs) with three different kernel functions: linear, radial basis function (RBF), and polynomial. These models aim to classify whether a relationship exists or not based on the input features of age and weight. By comparing the performance of these models, we can assess their effectiveness in capturing the relationship between the variables.

# ## 1. Importing Required Packages

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
from sklearn.metrics import fbeta_score
f2_scorer = make_scorer(fbeta_score, beta=2)

np.random.seed(1)

# ## 2.Generating Data

# We are generating data with two input columns age and weight.

# Generate random values for age and weight
age = np.random.randint(15, 56, size=200)
weight = np.random.uniform(40, 151, size=200)

# Create a DataFrame
df = pd.DataFrame({'age': age, 'weight': weight})

df

# Let's define the output variable, relationship_status with a set of conditions.

df['relationship_status'] = (df['age'] > 30) & (df['weight'] < 70)
df['relationship_status'] = df['relationship_status'].astype(int)

df

df

# Splitting dataframes X and y into test and train sets for further processing in the ratio of 30:70

X = df[['age','weight']]
y = df['relationship_status']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

X_train.shape

# ## 3. Model the data

performance = pd.DataFrame({"model": [], "Accuracy": [], "Precision": [], "Recall": [], "F1": [], "F2": [], "Parameters": []})

# ### 3.1 Fit a SVM classification model using linear kernal

# defining parameter range 
param_grid = {'C': [0.01, 0.1, 0.5, 1, 5, 10, 50, 100],  
              'kernel': ['linear']}
  
#grid = GridSearchCV(SVC(), param_grid, scoring='f1', refit = True, verbose = 3, n_jobs=-1) 

grid = GridSearchCV(SVC(), param_grid, scoring= f2_scorer, refit = True, verbose = 3, n_jobs=-1) 
  
# fitting the model for grid search 
_ = grid.fit(X_train, y_train)

# print best parameter after tuning 
print(grid.best_params_) 
  
# print how our model looks after hyper-parameter tuning 
print(grid.best_estimator_)

y_pred = grid.predict(X_test) 

recall = recall_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
f2 = fbeta_score(y_test, y_pred, beta=2)

performance = pd.concat([performance, pd.DataFrame({"model": ["SVM Linear"], "Accuracy": [accuracy], "Precision": [precision], "Recall": [recall], "F1": [f1], "F2": [f2], "Parameters": [grid.best_params_]})])


performance

# ### 3.2 Fit a SVM classification model using rbf kernal

# defining parameter range 
param_grid = {'C': [0.1, 1, 10, 100],  
              'gamma': [1, 0.1, 0.01, 0.001, 0.0001], 
              'kernel': ['rbf']}
  
grid = GridSearchCV(SVC(), param_grid, scoring=f2_scorer, refit = True, verbose = 3, n_jobs=-1) 
  
# fitting the model for grid search 
_ = grid.fit(X_train, y_train)

# print best parameter after tuning 
print(grid.best_params_) 
  
# print how our model looks after hyper-parameter tuning 
print(grid.best_estimator_)

y_pred = grid.predict(X_test) 

recall = recall_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
f2 = fbeta_score(y_test, y_pred, beta=2)

performance = pd.concat([performance, pd.DataFrame({"model": ["SVM rbf"], "Accuracy": [accuracy], "Precision": [precision], "Recall": [recall], "F1": [f1], "F2": [f2],"Parameters": [grid.best_params_]})])

# ### 3.3 Fit a SVM classification model using polynomial kernal

# defining parameter range 
param_grid = {'C': [0.01, 0.1, 0.5, 1, 5, 10, 50, 100],  
              'coef0': [0.01, 0.1, 0.5, 1, 5, 10, 50, 100],
              'kernel': ['poly']}
  
grid = GridSearchCV(SVC(), param_grid, scoring=f2_scorer, refit = True, verbose = 3, n_jobs=-1) 
  
# fitting the model for grid search 
_ = grid.fit(X_train, y_train)

# print best parameter after tuning 
print(grid.best_params_) 
  
# print how our model looks after hyper-parameter tuning 
print(grid.best_estimator_)

y_pred = grid.predict(X_test) 

recall = recall_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
f2 = fbeta_score(y_test, y_pred, beta=2)

performance = pd.concat([performance, pd.DataFrame({"model": ["SVM Poly"], "Accuracy": [accuracy], "Precision": [precision], "Recall": [recall], "F1": [f1], "F2": [f2], "Parameters": [grid.best_params_]})])

performance

# ## 4. Summary

# The evaluation results of three Support Vector Machine (SVM) models with different kernel types linear, radial basis function (rbf), and polynomial—reveal notable performance characteristics. The linear SVM model demonstrates strong accuracy and balanced precision and recall, suggesting its effectiveness in general classification tasks. Conversely, the rbf SVM model exhibits exceptional accuracy and recall, indicating its proficiency in capturing all positive instances, with a better precision compared to the linear model. The SVM with RBF and Polynomial kernels achieved perfect precision scores, indicating no false positives in their predictions. However, the Linear kernel SVM shows slightly lower precision and recall scores compared to the others. Both the RBF and polynomial kernel SVMs demonstrate high accuracy and perfect precision, potentially due to overfitting. Notably, rbf and poly models exhibited high F1 and F2 scores, reflecting a balance between precision and recall, and the ability to capture both true positives and true negatives effectively.


```
**Feedback:**
Data Generation (30%) - 0.4/0.45
• Creativity and relevance of the chosen scenario - 0.15/0.15
• Correct implementation of data generation techniques - 0.15/0.15
• Clarity and structure of the data (including labeling and documentation) - 0.1/0.15

Model Implementation and Tuning (40%) - 0.6/0.6
• Correct implementation of SVC/SVR models with the three specified kernels - 0.2/0.2
• Effective use of hyper-parameter tuning, with a clear rationale for chosen parameters - 0.2/0.2
• Documentation of model performance metrics, with comparisons to justify the best-performing model - 0.2/0.2

Analysis and Discussion (30%)- 0.35/0.45
• Clarity and depth of the introduction and explanation of the chosen problem and dataset - 0.125/0.225
• Critical analysis of model results, including insights into performance variations across different kernels - 0.225/0.225

Need to create a dedicated notebook for data generation as this keeps your code more organized and should mention whether you choose classification or regression task in introduction or data generation notebook. So, 0.15 marks were deducted.

---

# Student 19
```python
# #### Here I have imported necessary libraries for data manipulation, visualization, and machine learning with scikit-learn for Support Vector Machine (SVM) classification with data split, evaluation metrics, feature scaling and hyperparameter tuning and considered setting a random seed for reproducibility in order to obtain consistent results when randomness is involved

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler

np.random.seed(1)

# #### Generated a sample dataframe with sample size of 25 rows, 'Experience in Years' values ranges from 0 to 15, and 'No.of Skills' values ranges from 1 to 12.

sample_size = 25
min_work_exp = 0
max_work_exp = 15
skills_min = 1
skills_max = 12

df = pd.DataFrame({'Experience in Years': np.linspace(min_work_exp,max_work_exp,sample_size).astype(float),'No.of Skills': np.linspace(skills_min,skills_max,sample_size).astype(int)})
df

df['Selected for Interview'] = ((df['Experience in Years'] > 7) |  (df['No.of Skills'] > 5)).astype(int)
df

# #### Introducing random noise to the 'No.of Skills' column in the DataFrame to simulate variability. aiming to ensure that the data reflects some degree of randomness, to enhance the practicality of the dataset.

df['No.of Skills'] = df['No.of Skills'] + np.random.uniform(-25,25, sample_size)

# #### Plotting a scatter plot to understand the relationship between 'Experience in Years' and 'No.of Skills' with color-coded points on label 'Selected for Interview'.

plt.figure(figsize=(8, 7))
plt.scatter(df['Experience in Years'], df['No.of Skills'], c=df['Selected for Interview'], cmap='cool', edgecolors='k', alpha=0.8, s=100)
plt.scatter(df['Experience in Years'], df['No.of Skills'], c='blue', marker='o', label='Data Points')
plt.colorbar(label='Selected for Interview')
plt.title('Scatter Plot of Experience in Years vs. No.of Skills')
plt.xlabel('Experience in Years')
plt.ylabel('No.of Skills')
plt.legend()
plt.grid(True)
plt.show()

# #### Here the dataframe 'df' has been saved to a CSV file named as 'selected_for_interview.csv' where index is not included.

df.to_csv('selected_for_interview.csv', index=False)

# ### Train and Split Data

# Use sklearn to split df into a training set and a test set

X = df.iloc[:,:2]
y = df['Selected for Interview']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=1)


# ### Modeling the Data

performance = pd.DataFrame({"model": [], "Accuracy": [], "Precision": [], "Recall": [], "F1": [], "F2": [], "Parameters": []})

# generate a fbeta 2 scorer
from sklearn.metrics import make_scorer
from sklearn.metrics import fbeta_score
f2_scorer = make_scorer(fbeta_score, beta=2)

# ### 3.1 Fit a SVM classification model using linear kernal

# defining parameter range 
param_grid = {'C': [0.01, 0.1, 0.5, 1, 5, 10, 50, 100],  
              'kernel': ['linear']}
  

grid = GridSearchCV(SVC(), param_grid, scoring=f2_scorer, refit = True, verbose = 3, n_jobs=-1) 
  
# fitting the model for grid search 
_ = grid.fit(X_train, y_train)

# #### aiming to observe the best parameter by printting them after tuning, SVM model is tuned and then predicted the test set data. Also, calculated the performance metrics in order to evaluate

# print best parameter after tuning 
print(grid.best_params_) 
  
# print how our model looks after hyper-parameter tuning 
print(grid.best_estimator_)

y_pred = grid.predict(X_test) 

recall = recall_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
f2 = fbeta_score(y_test, y_pred, beta=2)

performance = pd.concat([performance, pd.DataFrame({"model": ["SVM Linear"], "Accuracy": [accuracy], "Precision": [precision], "Recall": [recall], "F1": [f1], "F2": [f2], "Parameters": [grid.best_params_]})])


# ### 3.2 Fit a SVM classification model using rbf kernal

# defining parameter range 
param_grid = {'C': [0.1, 1, 10, 100],  
              'gamma': [1, 0.1, 0.01, 0.001, 0.0001], 
              'kernel': ['rbf']}
  
grid = GridSearchCV(SVC(), param_grid, scoring=f2_scorer, refit = True, verbose = 3, n_jobs=-1) 
  
_ = grid.fit(X_train, y_train)

# #### aiming to observe the best parameters by printing them after tuning. After hyperparameter tuning, evaluated and stored the performance metrics of SVM(RBF Kernel).

# print best parameter after tuning 
print(grid.best_params_) 
  
# print how our model looks after hyper-parameter tuning 
print(grid.best_estimator_)

y_pred = grid.predict(X_test) 

recall = recall_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
f2 = fbeta_score(y_test, y_pred, beta=2)

performance = pd.concat([performance, pd.DataFrame({"model": ["SVM rbf"], "Accuracy": [accuracy], "Precision": [precision], "Recall": [recall], "F1": [f1], "F2": [f2],"Parameters": [grid.best_params_]})])

# ### 3.3 Fit a SVM classification model using polynomial kernal

# defining parameter range 
param_grid = {'C': [0.01, 0.1, 0.5, 1, 5, 10, 50, 100],  
              'coef0': [0.01, 0.1, 0.5, 1, 5, 10, 50, 100],
              'kernel': ['poly']}
  
grid = GridSearchCV(SVC(), param_grid, scoring=f2_scorer, refit = True, verbose = 3, n_jobs=-1) 
  
# fitting the model for grid search 
_ = grid.fit(X_train, y_train)

# #### Here the best paramters are printed, tuned the model, evaluated the performance metrics and finally appended the results to the dataframe.

# print best parameter after tuning 
print(grid.best_params_) 
  
# print how our model looks after hyper-parameter tuning 
print(grid.best_estimator_)

y_pred = grid.predict(X_test) 

recall = recall_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
f2 = fbeta_score(y_test, y_pred, beta=2)

performance = pd.concat([performance, pd.DataFrame({"model": ["SVM Poly"], "Accuracy": [accuracy], "Precision": [precision], "Recall": [recall], "F1": [f1], "F2": [f2], "Parameters": [grid.best_params_]})])

# ## 4.0 Summary

# #### Here the 'performance' dataframe is been sorted by the F2 column in descending order

performance.sort_values(by="F2", ascending=False)

# #### To conclude, with the three SVM models that are evaluated, the RBF kernel has achieved the highest accuracy (90%) and demonstrated excellent recall (100%). The Polynomial kernel, despite its high precision (100%), had slightly lower overall performance. The Linear kernel, while maintaining a precision of 100%, showed a lower accuracy (80%) and moderate recall (66.67%). The choice of the RBF kernel with parameters {'C': 1, 'gamma': 0.1} appears to be the most balanced for this classification task


```
**Feedback:**
Data Generation (30%) - 0.4/0.45
• Creativity and relevance of the chosen scenario - 0.15/0.15
• Correct implementation of data generation techniques - 0.15/0.15
• Clarity and structure of the data (including labeling and documentation) - 0.1/0.15

Model Implementation and Tuning (40%) - 0.6/0.6
• Correct implementation of SVC/SVR models with the three specified kernels - 0.2/0.2
• Effective use of hyper-parameter tuning, with a clear rationale for chosen parameters - 0.2/0.2
• Documentation of model performance metrics, with comparisons to justify the best-performing model - 0.2/0.2

Analysis and Discussion (30%)- 0.45/0.45
• Clarity and depth of the introduction and explanation of the chosen problem and dataset - 0.225/0.225
• Critical analysis of model results, including insights into performance variations across different kernels - 0.225/0.225

Need to create a dedicated notebook for data generation as this keeps your code more organized. So, 0.05 marks were deducted

---

# Student 20
```python
# ## Predicting Population Growth in Elephant Reserves using Support Vector Classifier

# ### Introduction:
# In this notebook, I aim to predict the population growth in elephant reserves based on two key factors: reserve size (in square kilometers) and funding for elephant conservation (in USD). The population rise category, which is classified as low, moderate, or large, is the output variable. In order to do this, I created synthetic data for funding and reserve size, then I used a function to classify population growth according to these numbers. Following that, I saved the generated synthetic data to use it in the second notebook to train and evaluate machine learning models that would forecast the population increase.

import pandas as pd
import numpy as np

np.random.seed(1)

# Number of reserves
sample_size = 100

# Minimum and maximum values for reserve size and funding
min_reserve_size = 500
max_reserve_size = 2000
min_funding = 50000
max_funding = 120000

# Generate synthetic data
df = pd.DataFrame({
    'Reserve Size (sq km)': np.random.randint(min_reserve_size, max_reserve_size, sample_size),
    'Funding for Elephant Conservation (USD)': np.random.randint(min_funding, max_funding, sample_size)
})

# Define a function to categorize the population increase
def categorize_population_increase(row):
    if row['Reserve Size (sq km)'] < 1000 and row['Funding for Elephant Conservation (USD)'] < 75000:
        return 'Low Increase'
    elif row['Reserve Size (sq km)'] < 1500 or row['Funding for Elephant Conservation (USD)'] < 100000:
        return 'Moderate Increase'
    else:
        return 'High Increase'

# Apply the function to create the target variable
df['Population Increase Category'] = df.apply(categorize_population_increase, axis=1)

# Display the dataframe
print(df)

df.head()

#Check how many rows and columns in the dataset
df.shape

# We can see that our dataset has 100 rows or observations and 3 columns or variables.

df.describe()

# Let's now address any missing values we might have in our dataset. 

# Check for missing values
df.isnull().sum()

#Check the variable types
df.dtypes

# #### Data Exploration: 
# The average reserve size is roughly 1377.29 square kilometers, and the average funding amount is 87508.01 USD. The reserve size spans from 515 to 1995 sq km, while the funding runs from 51741 to 119595 USD. Both of these variables display normal distributions. The distribution of the population growth categories is uniform. The dataset is complete for additional analysis because there are no missing values.

# Saving the generated synthetic data to use it in the second notebook to train and evaluate machine learning models

df.to_csv('./ElephantConservation.csv', index=False)

# ### Introduction
# In this notebook, I used hyper-parameter tuning to test and fit three different model forms of Support Vector Machines (SVM) for classification: linear kernel, radial basis function (RBF) kernel, and polynomial kernel. Based on the size of the reserve and the amount of funds allocated to elephant conservation, the goal is to predict the population increase category in elephant reserves. The synthetic data created in the first notebook has been loaded and split into training and test sets. I then used GridSearchCV to optimize each SVM model form's parameters using hyper-parameter tuning. Lastly, I evaluated each model form's performance using a range of metrics, including recall, accuracy, precision, F1 and F2 scores.

# ## 1. Set up

# Import modules

import pandas as pd
from sklearn.svm import SVC
from matplotlib import pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, f1_score, fbeta_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV

np.random.seed(1)

# ## 2. Load data

# Load data (it's already cleaned and preprocessed)

df = pd.read_csv('./ElephantConservation.csv')
df.head(3)

# Splitting the data into features and target
X = df[['Reserve Size (sq km)', 'Funding for Elephant Conservation (USD)']]
y = df['Population Increase Category']

# Splitting the data into training and test sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# #### Reasoning for 0.2 Test Size:
# We have 100 samples for testing with a test size of 0.2, which is often considered to be adequate to obtain an accurate evaluation of the model's performance.
# A greater test size may cause underfitting because of insufficient training data, whereas a smaller test size may cause overfitting on the training set. Between these two extremes, a test size of 0.2 is a reasonable ratio.

# ## 3. Model the data

# ### 3.1 Fit a SVM classification model using linear kernal

# defining parameter range
param_grid_linear = {'C': [0.01, 0.1, 0.5, 1, 5, 10, 50, 100],  
              'kernel': ['linear']}

# create a fbeta 2 scorer
from sklearn.metrics import make_scorer
f2_scorer = make_scorer(fbeta_score, beta=2)

# Recall is an important parameter in the context of elephant conservation because it assesses the model's accuracy in identifying reserves with high or moderate population increases, which are the target categories for conservation efforts. The f2 beta scorer is a measure that takes precision into account but also highlights the significance of recall. A high recall rate indicates that the model is successfully detecting these reserves, which is necessary for the proper distribution of resources and money. Because it supports the objective of precisely identifying reserves with a large or moderate population increase, a critical component of sustainable elephant conservation efforts, the f2 beta scorer is thus well-suited for this challenge.

# Hyper-parameter tuning with GridSearchCV
grid = GridSearchCV(SVC(), param_grid_linear, scoring=f2_scorer, refit=True, verbose=3, n_jobs=-1)
_ = grid.fit(X_train, y_train)

# print best parameter after tuning
print(grid.best_params_)

# print how our model looks after hyper-parameter tuning
print(grid.best_estimator_)

# Predictions on the test set
y_pred = grid.predict(X_test)

# Model performance metrics
recall = recall_score(y_test, y_pred, average='macro')
precision = precision_score(y_test, y_pred, average='macro')
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average='macro')
f2 = fbeta_score(y_test, y_pred, beta=2, average='macro')

# Adding performance metrics to the dataframe
performance_linear = pd.DataFrame({
    "model": ["SVM Linear"],
    "Accuracy": [accuracy],
    "Precision": [precision],
    "Recall": [recall],
    "F1": [f1],
    "F2": [f2],
    "Parameters": [grid.best_params_]
})
print(performance_linear)

performance_linear

# ### 3.2 Fit a SVM classification model using rbf kernal

# defining parameter range for RBF kernel
param_grid_rbf = {'C': [0.1, 1, 10, 100],  
              'gamma': [1, 0.1, 0.01, 0.001, 0.0001], 
              'kernel': ['rbf']}

# Hyper-parameter tuning with GridSearchCV for RBF kernel
grid_rbf = GridSearchCV(SVC(), param_grid_rbf, scoring=f2_scorer, refit=True, verbose=3, n_jobs=-1)
_ = grid_rbf.fit(X_train, y_train)

# print best parameter after tuning for RBF kernel
print(grid_rbf.best_params_)

# print how our model looks after hyper-parameter tuning for RBF kernel
print(grid_rbf.best_estimator_)

# Predictions on the test set for RBF kernel
y_pred_rbf = grid_rbf.predict(X_test)

# Model performance metrics for RBF kernel
recall_rbf = recall_score(y_test, y_pred_rbf, average='macro')
precision_rbf = precision_score(y_test, y_pred_rbf, average='macro')
accuracy_rbf = accuracy_score(y_test, y_pred_rbf)
f1_rbf = f1_score(y_test, y_pred_rbf, average='macro')
f2_rbf = fbeta_score(y_test, y_pred_rbf, beta=2, average='macro')

# Adding performance metrics to the dataframe for RBF kernel
performance_rbf = pd.DataFrame({
    "model": ["SVM RBF"],
    "Accuracy": [accuracy_rbf],
    "Precision": [precision_rbf],
    "Recall": [recall_rbf],
    "F1": [f1_rbf],
    "F2": [f2_rbf],
    "Parameters": [grid_rbf.best_params_]
})
print(performance_rbf)

performance_rbf

# ### 3.3 Fit a SVM classification model using polynomial kernal

# defining parameter range for polynomial kernel
param_grid_poly = {'C': [0.01, 0.1, 0.5, 1, 5, 10, 50, 100],  
              'coef0': [0.01, 0.1, 0.5, 1, 5, 10, 50, 100],
              'kernel': ['poly']}

# Hyper-parameter tuning with GridSearchCV for polynomial kernel
grid_poly = GridSearchCV(SVC(), param_grid_poly, scoring=f2_scorer, refit=True, verbose=3, n_jobs=-1)
_ = grid_poly.fit(X_train, y_train)

# print best parameter after tuning for polynomial kernel
print(grid_poly.best_params_)

# print how our model looks after hyper-parameter tuning for polynomial kernel
print(grid_poly.best_estimator_)

# Predictions on the test set for polynomial kernel
y_pred_poly = grid_poly.predict(X_test)

# Model performance metrics for polynomial kernel
recall_poly = recall_score(y_test, y_pred_poly, average='macro')
precision_poly = precision_score(y_test, y_pred_poly, average='macro')
accuracy_poly = accuracy_score(y_test, y_pred_poly)
f1_poly = f1_score(y_test, y_pred_poly, average='macro')
f2_poly = fbeta_score(y_test, y_pred_poly, beta=2, average='macro')

# Adding performance metrics to the dataframe for polynomial kernel
performance_poly = pd.DataFrame({
    "model": ["SVM Polynomial"],
    "Accuracy": [accuracy_poly],
    "Precision": [precision_poly],
    "Recall": [recall_poly],
    "F1": [f1_poly],
    "F2": [f2_poly],
    "Parameters": [grid_poly.best_params_]
})
print(performance_poly)

performance_poly

pd.concat([performance_linear, performance_rbf, performance_poly], axis=0)

# ## Analysis of Results:

# 1. Linear Kernel Mode:
# - Accuracy: 95%
# - Precision: 64.81%
# - Recall: 66.67%
# - F1 Score: 65.71%
# - F2 Score: 66.28%
# 
# With its high accuracy and balanced precision and recall, the linear kernel model demonstrated outstanding results. This suggests that there may be a significantly linear relationship between reserve size, funding, and population growth. Larger reserves and more funding are typically linked to faster population expansion.
# 
# 2. RBF Kernel Model:
# - Accuracy: 85%
# - Precision: 28.33%
# - Recall: 33.33%
# - F1 Score: 30.63%
# - F2 Score: 32.20%
# 
# It appears that the data may not be suitable for a non-linear separation because the RBF kernel model performed poorly than the linear model. This may suggest that there are fewer complicated relationships and a more direct relationship between reserve size and funding and population growth.
# 
# 3. Polynomial Kernel Model:
# - Accuracy: 85%
# - Precision: 28.33%
# - Recall: 33.33%
# - F1 Score: 30.63%
# - F2 Score: 32.20%
# 
# The polynomial kernel model underperformed the linear model, much like the RBF model did. This gives further support to the assumption that there is a great linear link between the input variables and the population increase category.

# ## Discussion:

# The findings imply that key factors influencing elephant reserve population expansion are reserve size and money for conservation programs. There is a strong linear relationship between these factors and population growth, as indicated by the linear kernel model's performance. This research emphasizes how crucial it is to allocate enough funds to maintain elephant numbers over the long term.
# 
# On the other hand, the less complex and non-linear nature of the correlation between these parameters and population growth may be shown by the non-linear models' poorer performance (polynomial kernels and RBF). This knowledge can help direct conservation efforts by concentrating on simple tactics like expanding the size of reserves and funding levels.
# 
# All things considered, this analysis shows how machine learning algorithms can offer insightful information for conservation planning. In order to further increase prediction accuracy, future research could examine additional factors impacting population growth.


```
**Feedback:**
Data Generation (30%) - 0.45/0.45
• Creativity and relevance of the chosen scenario - 0.15/0.15
• Correct implementation of data generation techniques - 0.15/0.15
• Clarity and structure of the data (including labeling and documentation) - 0.15/0.15

Model Implementation and Tuning (40%) - 0.6/0.6
• Correct implementation of SVC/SVR models with the three specified kernels - 0.2/0.2
• Effective use of hyper-parameter tuning, with a clear rationale for chosen parameters - 0.2/0.2
• Documentation of model performance metrics, with comparisons to justify the best-performing model - 0.2/0.2

Analysis and Discussion (30%)- 0.45/0.45
• Clarity and depth of the introduction and explanation of the chosen problem and dataset - 0.225/0.225
• Critical analysis of model results, including insights into performance variations across different kernels - 0.225/0.225

---

# Student 21
```python
# # Data Generation Notebook
# ---

# ## Introduction
# In this notebook, we will generate synthetic data to simulate a futuristic scenario where threat levels are classified based on Quantum Fluctuation and Hyperspace Distortion. The goal is to create a classification dataset for training a machine learning model.

# ## Importing Modules

import pandas as pd
import numpy as np
np.random.seed(38559388)

# ## Data Generation
# - **Quantum Fluctuation:** Represents the quantum energy fluctuations measured in a space region.
# - **Hyperspace Distortion:** Indicates the degree of distortion in the hyperspace fabric.
# 
# **The target variable is:**
# 
# - **Threat Level:** A binary classification indicating whether the combination of Quantum Fluctuation and Hyperspace Distortion poses a threat.
# 
# ---

# ## Data Generation Process
# 
# The data generation process involves randomizing the Quantum Fluctuation and Hyperspace Distortion values and assigning a threat level based on a certain threshold. We aim to create a diverse dataset that captures the essence of this futuristic scenario.
# 
# ---

def generate_classification_data(size=1000):
    quantum_fluctuation = np.random.uniform(0, 1, size)
    hyperspace_distortion = np.random.uniform(0, 1, size)
    threat_level = np.where(quantum_fluctuation + hyperspace_distortion > 1, 1, 0)  # Binary classification

    return pd.DataFrame({'Quantum_Fluctuation': quantum_fluctuation, 'Hyperspace_Distortion': hyperspace_distortion, 'Threat_Level': threat_level})

data = generate_classification_data()

data.head()

# ## Data Saving
# 
# We will save the generated synthetic data into a CSV file for later use in machine learning tasks. This CSV file will be read in another notebook for training and evaluating a classification model.

data.to_csv('classification_data.csv', index=False)

# # Model Fitting Notebook
# ---

# ## Introduction
# This notebook explores the performance of Support Vector Machine (SVM) classification models with different kernels – linear, radial basis function (rbf), and polynomial. The dataset used for this evaluation represents a futuristic scenario where threat levels are classified based on Quantum Fluctuation and Hyperspace Distortion. We aim to determine the most effective SVM kernel for this classification task through hyperparameter tuning and model evaluation.
# 
# ---

# ## Metric
# - We choose recall as our main metric because In the context of the our dataset, where the goal is to predict the "Threat_Level" based on features such as "Quantum_Fluctuation" and "Hyperspace_Distortion," a good reason to make recall the main metric are the potential consequences of missing actual threats.
# 
# - In a security threat detection scenario, where the "Threat_Level" represents the severity of a security threat, prioritizing recall may be crucial because Missing a genuine threat (false negatives) may have severe consequences.
# 
# -  Prioritizing recall ensures that the model is sensitive to the presence of threats and strives to capture all actual threats, minimizing the risk of overlooking potentially harmful situations.
# 
# ---

# > **Note:** The Following code is inspired from Prof Smith's Notebook 'SVM_model_wt_tuning.ipynb'.
# 
# ---

# ## 1. Setup

# Importing the required modules

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, f1_score

np.random.seed(38559388)

# ## 2. Load data

df = pd.read_csv('classification_data.csv')
df.head(3)

# Use sklearn to split df into a training set and a test set

X = df[['Quantum_Fluctuation', 'Hyperspace_Distortion']]
y = df['Threat_Level']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ## 3. Model the data

performance = pd.DataFrame({"model": [], "Accuracy": [], "Precision": [], "Recall": [], "F1": [], "Parameters": []})

# ### 3.1 Fit a SVM classification model using linear kernal

# defining parameter range 
param_grid = {'C': [0.001,0.01, 0.1, 0.5, 1, 5],  
              'kernel': ['linear']}
  
grid = GridSearchCV(SVC(), param_grid, scoring='recall', refit = True, verbose = 3, n_jobs=-1) 
  
# fitting the model for grid search 
_ = grid.fit(X_train, y_train)

# print best parameter after tuning 
print(grid.best_params_) 
  
# print how our model looks after hyper-parameter tuning 
print(grid.best_estimator_)

y_pred = grid.predict(X_test) 

recall = recall_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

performance = pd.concat([performance, pd.DataFrame({"model": ["SVM Linear"], "Accuracy": [accuracy], "Precision": [precision], "Recall": [recall], "F1": [f1], "Parameters": [grid.best_params_]})])


# ### 3.2 Fit a SVM classification model using rbf kernal

# defining parameter range 
param_grid = {'C': [0.1, 1, 10, 100],  
              'gamma': [1, 0.1, 0.01, 0.001, 0.0001], 
              'kernel': ['rbf']}
  
grid = GridSearchCV(SVC(), param_grid, scoring='recall', refit = True, verbose = 3, n_jobs=-1) 
  
# fitting the model for grid search 
_ = grid.fit(X_train, y_train)

# print best parameter after tuning 
print(grid.best_params_) 
  
# print how our model looks after hyper-parameter tuning 
print(grid.best_estimator_)

y_pred = grid.predict(X_test) 

recall = recall_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

performance = pd.concat([performance, pd.DataFrame({"model": ["SVM rbf"], "Accuracy": [accuracy], "Precision": [precision], "Recall": [recall], "F1": [f1], "Parameters": [grid.best_params_]})])

# ### 3.3 Fit a SVM classification model using polynomial kernal

# defining parameter range 
param_grid = {'C': [0.001,0.01, 0.1, 0.5, 1, 5],  
              'coef0': [0.001,0.01, 0.1, 0.5, 1, 5],
              'kernel': ['poly']}
  
grid = GridSearchCV(SVC(), param_grid, scoring='recall', refit = True, verbose = 3, n_jobs=-1) 
  
# fitting the model for grid search 
_ = grid.fit(X_train, y_train)

# print best parameter after tuning 
print(grid.best_params_) 
  
# print how our model looks after hyper-parameter tuning 
print(grid.best_estimator_)

y_pred = grid.predict(X_test) 

recall = recall_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

performance = pd.concat([performance, pd.DataFrame({"model": ["SVM Poly"], "Accuracy": [accuracy], "Precision": [precision], "Recall": [recall], "F1": [f1], "Parameters": [grid.best_params_]})])

# ## 4.0 Summary

performance.sort_values(by="Recall", ascending=False)

# ## 5.0 Inference

# 
# The differences in performance across the three SVM kernels (polynomial, RBF, and linear) could be attributed to the inherent characteristics of each kernel function and how well those characteristics align with the underlying patterns in the dataset.

# ## SVM Poly (Polynomial Kernel):
# 
# **Flexibility in Capturing Non-linear Relationships:**
# - **Strength:** The polynomial kernel is known for capturing non-linear relationships in the data. It may have been more effective in capturing the complex relationships between the features and the threat level.
# 
# ---
# 
# ## SVM RBF (Radial Basis Function Kernel):
# 
# **Generalization to Complex Patterns:**
# - **Strength:** The RBF kernel is versatile and can model complex decision boundaries. It may generalize well to intricate patterns present in the dataset.
# 
# ---
# 
# ## SVM Linear (Linear Kernel):
# 
# **Sensitivity to Linear Separability:**
# - **Linear Separation:** The linear kernel assumes that the data is linearly separable. If the underlying relationship in the dataset has a linear nature, the linear kernel can perform well.
# 
# ---

# ## 6.0 Analysis

# - The dataset might have characteristics that favor one type of decision boundary over others. For instance, the relationship between features and threat level might be non-linear,which is the reason why the polynomial kernel is performing better.
# 
# - The effectiveness of each kernel could be influenced by how well the hyperparameters are tuned. The grid search process might have identified hyperparameters that align well with the characteristics of each kernel.
# 
# - The effectiveness of kernels can be influenced by the preprocessing steps applied to the data. Scaling, handling outliers, and other preprocessing techniques can impact model performance.


```
**Feedback:**
Data Generation (30%) - 0.45/0.45
• Creativity and relevance of the chosen scenario - 0.15/0.15
• Correct implementation of data generation techniques - 0.15/0.15
• Clarity and structure of the data (including labeling and documentation) - 0.15/0.15

Model Implementation and Tuning (40%) - 0.6/0.6
• Correct implementation of SVC/SVR models with the three specified kernels - 0.2/0.2
• Effective use of hyper-parameter tuning, with a clear rationale for chosen parameters - 0.2/0.2
• Documentation of model performance metrics, with comparisons to justify the best-performing model - 0.2/0.2

Analysis and Discussion (30%)- 0.275/0.45
• Clarity and depth of the introduction and explanation of the chosen problem and dataset - 0.225/0.225
• Critical analysis of model results, including insights into performance variations across different kernels - 0.05/0.225

The analysis section lacks clarity on determining the best-performing model. It fails to represent why the polynomial kernel was only considered as better performer, especially when the RBF kernel exhibits same performance parameters. So, 0.175 marks were deducted.

---

# Student 22
```python
# # Week03 - SVM models - Linear, RBF, Polynomial
# 
# In this notebook, we have a synthetic dataset representing contract bid prediction depending on years of experience and contract bid amount in dollars. The dataset consists of 1000 samples, with two input features: Years of experience and Contract bid amount in dollar, and a binary output variable indicating Contract bid winning (0 for rejection and 1 for approval). The Contract_winning prediction function uses the applicant's Years of experience and COntract bid amount as input.If both values meet predefined thresholds, the function predicts result; otherwise, it predicts loss. We'll explore how the model makes predictions by analyzing the relationship between years of experience and contract bidding amount through visualization and discussion. 

# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, f1_score

# ## Sample Size and Range Parameters definition

 # Define threshold values
experience_threshold = 11  # Years of experience threshold
bid_amount_threshold = 500000  

# ## Generating Target Variable by applying conditions on input columns


# Set random seed for reproducibility
np.random.seed(42)

# Generate synthetic data
num_rows = 1000

# Generate contract bid amounts (in dollars) between $10,000 and $1,000,000
contract_bid_amounts = np.random.randint(10000, 1000000, size=num_rows)

# Generate years of experience ranging from 1 to 20 years
years_of_experience = np.random.randint(1, 21, size=num_rows)

# Create DataFrame
data = pd.DataFrame({
    'Contract Bid Amount ($)': contract_bid_amounts,
    'Years of Experience': years_of_experience
})

# Display the first few rows of the generated data
print(data.head())


def generate_target_variable(contract_bid_amounts, years_of_experience):
    

    # Initialize an empty list to store target variable values
    target_variable = []

    # Iterate through each row of input data
    for bid_amount, experience in zip(contract_bid_amounts, years_of_experience):
    # Check if the company meets both criteria to win the contract
        if experience >= experience_threshold and bid_amount <= bid_amount_threshold:
            target_variable.append(1)  # Win the contract
        else:
            target_variable.append(0)  # Do not win the contract

    return target_variable

# Generate target variable using the function
target_variable = generate_target_variable(data['Contract Bid Amount ($)'], data['Years of Experience'])

# Add the target variable to the DataFrame
data['Winning Contract'] = target_variable

# Display the first few rows of the DataFrame with the target variable
print(data.head())

data['Winning Contract'].value_counts()

# ## Visualize the data

# Visualize the data
plt.figure(figsize=(8, 6))
plt.scatter(data[data['Winning Contract'] == 0]['Contract Bid Amount ($)'], data[data['Winning Contract'] == 0]['Years of Experience'], color='red', label='Lost', alpha=0.5)
plt.scatter(data[data['Winning Contract'] == 1]['Contract Bid Amount ($)'], data[data['Winning Contract'] == 1]['Years of Experience'], color='blue', label='Won', alpha=0.5)
plt.xlabel('Contract Bid Amount ($)')
plt.ylabel('Years of Experience')
plt.title('Synthetic Classification of Contract Bidding')
plt.legend()
plt.show()


data.head(10)

# Save the dataset to a CSV file
data.to_csv('synthetic_Contract_Bidding_data.csv', index=False)

# Load the dataset
data = pd.read_csv('synthetic_Contract_Bidding_data.csv')

# Split the data into features (X) and target variable (y)
X = data[['Contract Bid Amount ($)', 'Years of Experience']]
y = data['Winning Contract']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# ## Model the data
# 
# First, let's create a dataframe to load the model performance metrics into.

performance = pd.DataFrame({"model": [], "Accuracy": [], "Precision": [], "Recall": [], "F1": [], "F2": [], "Parameters": []})

# create a fbeta 2 scorer
from sklearn.metrics import make_scorer
from sklearn.metrics import fbeta_score
f2_scorer = make_scorer(fbeta_score, beta=2)


# ## Fit a SVM classification model using linear kernal

# defining parameter range 
param_grid = {'C': [0.01, 0.1, 0.5, 1, 5, 10, 50, 100],  
              'kernel': ['linear']}
  
#grid = GridSearchCV(SVC(), param_grid, scoring='f1', refit = True, verbose = 3, n_jobs=-1) 

grid = GridSearchCV(SVC(), param_grid, scoring=f2_scorer, refit = True, verbose = 3, n_jobs=-1) 
  
# fitting the model for grid search 
_ = grid.fit(X_train, y_train)

# print best parameter after tuning 
print(grid.best_params_) 
  
# print how our model looks after hyper-parameter tuning 
print(grid.best_estimator_)

y_pred = grid.predict(X_test) 

recall = recall_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
f2 = fbeta_score(y_test, y_pred, beta=2)

performance = pd.concat([performance, pd.DataFrame({"model": ["SVM Linear"], "Accuracy": [accuracy], "Precision": [precision], "Recall": [recall], "F1": [f1], "F2": [f2], "Parameters": [grid.best_params_]})])


performance

# Calculate confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Create a heatmap for the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Spectral_r', xticklabels=['Predicted Negative', 'Predicted Positive'], yticklabels=['Actual Negative', 'Actual Positive'])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()

# ## SVC with Linear Kernel result analysis:
# 
# ### Introduction:
# 
# In this analysis, we explore the performance of the Support Vector Classifier (SVC) with a linear kernel for Winning Contract. The dataset consists of synthetic data representing various features related to Contract Bid applicants, such as years of experience and contract bidding amount to predict whether they will win or not.
# 
# ### Evaluation Metrics:
# 
# We evaluate the model using the following metrics:
# 
# Accuracy: Measures the proportion of correctly classified instances out of all instances.
# 
# Precision: Measures the proportion of true positive predictions out of all positive predictions.
# 
# Recall: Measures the proportion of true positive predictions out of all actual positive instances.
# 
# F1 Score: The harmonic mean of precision and recall, providing a balanced measure between the two.
# 
# ### Results:
# 
# SVC with Linear Kernel:
# 
# Accuracy: 78%
# 
# Precision: 77%
# 
# Recall: 78.5%
# 
# F1 Score: 77.7%
# 
# F2 Score: 78.25%
# 
# Parameters: {'C': 0.01, 'kernel': 'linear'}
# 
# ### Discussion:
# 
# The SVC with a linear kernel assumes that the data is linearly separable in the input space. It operates effectively when the underlying data can be distinctly separated by a linear boundary. In this case, the decision boundary is linear, signifying that it separates the data into classes using a straight line.
# 
# Analyzing the performance metrics:
# 
# Accuracy: The model achieves an accuracy of 78%, indicating that approximately 78% of the instances are correctly classified.
# 
# Precision: With a precision of 77%, the model correctly identifies around 77% of the positive predictions out of all positive predictions made.
# 
# Recall: The recall score of 78.57% implies that the model captures roughly 78.57% of the actual positive instances.
# 
# F1 Score: The F1 score, which considers both precision and recall, is computed as 77.78%. This harmonic mean provides a balanced measure of the model's performance.
# 
# Parameters: The model is trained with a regularization parameter (C) set to 0.01 and a linear kernel.
# 
# Overall, the SVC with a linear kernel demonstrates moderate performance in predicting contract bid result based on the provided features. While the model achieves reasonable accuracy, precision, recall, and F1-score, further exploration with different kernels or fine-tuning of parameters may be necessary to improve its performance.

# ## Fit a SVM classification model using rbf kernal

# defining parameter range 
param_grid = {'C': [0.1, 1, 10, 100],  
              'gamma': [1, 0.1, 0.01, 0.001, 0.0001], 
              'kernel': ['rbf']}
  
grid = GridSearchCV(SVC(), param_grid, scoring=f2_scorer, refit = True, verbose = 3, n_jobs=-1) 
  
# fitting the model for grid search 
_ = grid.fit(X_train, y_train)

# print best parameter after tuning 
print(grid.best_params_) 
  
# print how our model looks after hyper-parameter tuning 
print(grid.best_estimator_)

y_pred = grid.predict(X_test) 

recall = recall_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
f2 = fbeta_score(y_test, y_pred, beta=2)

performance = pd.concat([performance, pd.DataFrame({"model": ["SVM rbf"], "Accuracy": [accuracy], "Precision": [precision], "Recall": [recall], "F1": [f1], "F2": [f2],"Parameters": [grid.best_params_]})])

performance

# Calculate confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Create a heatmap for the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Reds', xticklabels=['Predicted Negative', 'Predicted Positive'], yticklabels=['Actual Negative', 'Actual Positive'])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()

# ## SVC with RBF Kernel result analysis:
# 
# ### Introduction:
# 
# In this analysis, we explore the performance of the Support Vector Classifier (SVC) with a radial basis function (RBF) kernel for Contract bid winning prediction. The dataset consists of synthetic data representing various features related to contract bid, years of experience and contract bidding amount, to predict whether a bid will be won or lost.
# 
# The RBF kernel is chosen for its ability to capture non-linear relationships in the data. Unlike the linear kernel, which assumes linear separability in the input space, the RBF kernel computes similarity between data points in a high-dimensional space, allowing for more complex patterns to be captured.
# 
# 
# ### Evaluation Metrics:
# 
# We evaluate the model based on the following metrics:
# 
# Accuracy: Measures the proportion of correctly classified instances out of all instances.
# 
# Precision: Measures the proportion of true positive predictions out of all positive predictions.
# 
# Recall: Measures the proportion of true positive predictions out of all actual positive instances.
# 
# F1 Score: The harmonic mean of precision and recall, providing a balanced measure between the two.
# 
# ### Results - SVC with RBF Kernel:
# 
# Accuracy: 77.5%
# 
# Precision: 76.77%
# 
# Recall: 77.55%
# 
# F1 Score: 77.16%
# 
# F2 Score: 77.39%
# 
# Parameters: {'C': 100, 'gamma': 0.01, 'kernel': 'rbf'}
# 
# ### Discussion:
# 
# The SVC with a radial basis function (RBF) kernel differs from the linear kernel by its ability to capture non-linear relationships in the data. It computes similarity between data points in a high-dimensional space, allowing for more complex patterns to be captured.
# 
# Analyzing the performance metrics:
# 
# Accuracy: The model achieves an accuracy of 77.5%, indicating that approximately 77.5% of the instances are correctly classified.
# 
# Precision: With a precision of 76.77%, the model correctly identifies around 76.77% of the positive predictions out of all positive predictions made.
# 
# Recall: The recall score of 77.55% implies that the model captures roughly 77.55% of the actual positive instances.
# 
# F1 Score: The F1 score, which considers both precision and recall, is computed as 77.16%. This harmonic mean provides a balanced measure of the model's performance.
# 
# F2 Score: The F2 score, which places more emphasis on recall, is slightly higher at 77.39%.
# 
# Overall, the SVC with an RBF kernel demonstrates moderate performance in predicting Contract bids. It leverages the non-linear nature of the RBF kernel to capture more complex relationships in the data, resulting in comparable performance to the linear kernel. Further experimentation with different parameter settings or kernel types may be necessary to improve performance further.

# ## Fit a SVM classification model using polynomial kernal

# defining parameter range 
param_grid = {'C': [0.01, 0.1, 0.5, 1, 5, 10, 50, 100],  
              'coef0': [0.01, 0.1, 0.5, 1, 5, 10, 50, 100],
              'kernel': ['poly']}
  
grid = GridSearchCV(SVC(), param_grid, scoring=f2_scorer, refit = True, verbose = 3, n_jobs=-1) 
  
# fitting the model for grid search 
_ = grid.fit(X_train, y_train)

# print best parameter after tuning 
print(grid.best_params_) 
  
# print how our model looks after hyper-parameter tuning 
print(grid.best_estimator_)

y_pred = grid.predict(X_test) 

recall = recall_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
f2 = fbeta_score(y_test, y_pred, beta=2)

performance = pd.concat([performance, pd.DataFrame({"model": ["SVM Poly"], "Accuracy": [accuracy], "Precision": [precision], "Recall": [recall], "F1": [f1], "F2": [f2], "Parameters": [grid.best_params_]})])

performance

# Calculate confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Create a heatmap for the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges', xticklabels=['Predicted Negative', 'Predicted Positive'], yticklabels=['Actual Negative', 'Actual Positive'])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()

# ## SVC with Polynomial Kernel result analysis:
# 
# ### Introduction:
# 
# 
# In this analysis, we explore the performance of the Support Vector Classifier (SVC) with a radial basis function (RBF) kernel for Contract bid winning prediction. The dataset consists of synthetic data representing various features related to contract bid, years of experience and contract bidding amount, to predict whether a bid will be won or lost.
# 
# The polynomial kernel is selected for its ability to capture non-linear relationships in the data. Unlike the linear kernel, which assumes linear separability in the input space, the polynomial kernel computes similarity between data points using polynomial functions, allowing for more complex patterns to be captured.
# 
# ### Evaluation Metrics:
# 
# We evaluate the model using the following metrics:
# 
# Accuracy: Measures the proportion of correctly classified instances out of all instances.
# Precision: Measures the proportion of true positive predictions out of all positive predictions.
# Recall: Measures the proportion of true positive predictions out of all actual positive instances.
# F1 Score: The harmonic mean of precision and recall, providing a balanced measure between the two.
# 
# ### Results: SVC with Polynomial Kernel:
# 
# Accuracy: 71.5%
# 
# Precision: 65.41%
# 
# Recall: 88.78%
# 
# F1 Score: 75.32%
# 
# F2 Score: 82.86%
# 
# Parameters: {'C': 0.5, 'coef0': 0.01, 'kernel': 'poly'}
# 
# ### Discussion:
# 
# The SVC with a polynomial kernel is adept at capturing non-linear relationships in the data, making it suitable for complex datasets like housing price prediction. By using polynomial functions to compute similarity between data points, the model can capture intricate patterns that may not be discernible with a linear kernel.
# 
# Analyzing the performance metrics:
# 
# Accuracy: The model achieves an accuracy of 71.5%, indicating that approximately 71.5% of the instances are correctly classified.
# 
# Precision: With a precision of 65.41%, the model correctly identifies around 65.41% of the positive predictions out of all positive predictions made.
# 
# Recall: The recall score of 88.78% implies that the model captures roughly 88.78% of the actual positive instances.
# 
# F1 Score: The F1 score, which considers both precision and recall, is computed as 75.32%. This harmonic mean provides a balanced measure of the model's performance.
# 
# F2 Score: The F2 score, which places more emphasis on recall, is slightly higher at 82.86%.
# 
# Overall, the SVC with a polynomial kernel demonstrates moderate performance in predicting bid winning. Leveraging the polynomial kernel's ability to capture non-linear relationships, the model achieves reasonable accuracy, precision, recall, and F1-score. Further fine-tuning of parameters or exploration of different kernel types may be necessary to enhance performance further.

# ## Summary
# 
# The provided SVM models, including Linear, RBF, and Polynomial kernels, were evaluated based on performance metrics and parameters.Linear and RBF SVMs show nearly identical results, suggesting that the data might be linearly separable or the RBF parameters had minimal impact.Polynomial SVM performed notably worse, indicating unsuitability for the dataset or suboptimal parameter selection.Precision was perfect for all models, but recall and F1-score were significantly lower for Polynomial SVM.

performance.sort_values(by="F2", ascending=False)


```
**Feedback:**
Data Generation (30%) - 0.4/0.45
• Creativity and relevance of the chosen scenario - 0.15/0.15
• Correct implementation of data generation techniques - 0.15/0.15
• Clarity and structure of the data (including labeling and documentation) - 0.1/0.15

Model Implementation and Tuning (40%) - 0.6/0.6
• Correct implementation of SVC/SVR models with the three specified kernels - 0.2/0.2
• Effective use of hyper-parameter tuning, with a clear rationale for chosen parameters - 0.2/0.2
• Documentation of model performance metrics, with comparisons to justify the best-performing model - 0.2/0.2

Analysis and Discussion (30%)- 0.45/0.45
• Clarity and depth of the introduction and explanation of the chosen problem and dataset - 0.225/0.225
• Critical analysis of model results, including insights into performance variations across different kernels - 0.225/0.225

Need to create a dedicated notebook for data generation as this keeps your code more organized. So, 0.05 marks were deducted

---

# Student 23
```python
# ### SVM Subscription Prediction (Classification)
# 
# In this demonstration, we'll utilize Support Vector Machines (SVM) to predict subscription likelihood based on age and income in a synthetic dataset.
# 
# The synthetic dataset we're using is generated to predict subscription likelihood based on two features: age and income. Customers with an age below 30 and income above $50,000 are considered more likely to subscribe. Thus, the dataset has two input columns representing age and income, and one output column representing subscription status (1 for subscribe, 0 for not subscribe).
# 
# We'll employ the SVM class from scikit-learn to fit models to the data using different kernels: linear, radial basis function (RBF), and polynomial. Additionally, we'll utilize GridSearchCV to find the best hyperparameters for each model. The scoring metric used for hyperparameter tuning is a custom F2 score, with a higher weight on recall than precision.
# 
# This choice of metric is motivated by the nature of the problem. In subscription prediction, a false negative (predicting a customer won't subscribe when they actually would) is more detrimental than a false positive (predicting a customer will subscribe when they won't). Thus, we prioritize recall to minimize false negatives, while still considering precision to avoid unnecessary marketing efforts.

# ## Data Generation

# ### Importing Libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV

# Setting random seed for reproducibility
np.random.seed(1)

# ### Generating data on Subscription Predication based on Age and Income

sample_size = 100

# Generate synthetic data for input columns
age = np.random.randint(18, 65, size=sample_size)  # Random age between 18 and 65
income = np.random.uniform(20000, 100000, size=sample_size)  # Random income between $20,000 and $100,000

# Generate synthetic data for output column (subscribe/not subscribe)
# Assume that customers with age below 30 and income above $50,000 are more likely to subscribe
subscribe = ((age < 30) & (income > 50000)).astype(int)

# Create DataFrame
df = pd.DataFrame({'age': age, 'income': income, 'subscribe': subscribe})

# Save DataFrame to CSV file
##df.to_csv('subscription_data.csv', index=False)

# My generated Data
df.head(10)

# Plotting my variables
fig = plt.figure()
ax = fig.add_subplot()
colors = np.array(["blue", "red"])
ax.scatter(df['age'], df['income'], c=colors[np.ravel(df['subscribe'])])
ax.set_xlabel('Age')
ax.set_ylabel('Income')
ax.set_title('Subscription Prediction based on Age and Income')
plt.show()

# ## Model the Data

# Split the data into features (X) and target variable (y)
X = df[['age', 'income']]
y = df['subscribe']

# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# Creating a dataframe to load the model performance metrics into.

# Create a dataframe to store model performance metrics
performance = pd.DataFrame({"model": [], "Accuracy": [], "Precision": [], "Recall": [], "F1": [], "F2": [], "Parameters": []})

# create a fbeta 2 scorer
from sklearn.metrics import make_scorer
from sklearn.metrics import fbeta_score
f2_scorer = make_scorer(fbeta_score, beta=2)

# ### Fitting using Linear Kernel

# defining parameter range 
param_grid = {'C': [0.01, 0.1, 0.5, 1, 5, 10, 50, 100], 'kernel': ['linear']}
  
#grid = GridSearchCV(SVC(), param_grid, scoring='f1', refit = True, verbose = 3, n_jobs=-1) 

grid = GridSearchCV(SVC(), param_grid, scoring=f2_scorer, refit = True, verbose = 3, n_jobs=-1) 
  
# fitting the model for grid search 
_ = grid.fit(X_train, y_train)

# print best parameter after tuning 
print(grid.best_params_) 
  
# print how our model looks after hyper-parameter tuning 
print(grid.best_estimator_)

y_pred = grid.predict(X_test) 

recall = recall_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
f2 = fbeta_score(y_test, y_pred, beta=2)

performance = pd.concat([performance, pd.DataFrame({"model": ["SVM Linear"], "Accuracy": [accuracy], "Precision": [precision], "Recall": [recall], "F1": [f1], "F2": [f2], "Parameters": [grid.best_params_]})])


performance

# ### Fitting Using RBF Kernel

# defining parameter range 
param_grid = {'C': [0.1, 1, 10, 100], 'gamma': [1, 0.1, 0.01, 0.001, 0.0001],'kernel': ['rbf']}
  
grid = GridSearchCV(SVC(), param_grid, scoring=f2_scorer, refit = True, verbose = 3, n_jobs=-1) 
  
# fitting the model for grid search 
_ = grid.fit(X_train, y_train)

# print best parameter after tuning 
print(grid.best_params_) 
  
# print how our model looks after hyper-parameter tuning 
print(grid.best_estimator_)

y_pred = grid.predict(X_test) 

recall = recall_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
f2 = fbeta_score(y_test, y_pred, beta=2)

performance = pd.concat([performance, pd.DataFrame({"model": ["SVM rbf"], "Accuracy": [accuracy], "Precision": [precision], "Recall": [recall], "F1": [f1], "F2": [f2],"Parameters": [grid.best_params_]})])

performance

# ### Fitting using Polynomial Kernel

# defining parameter range 
param_grid = {'C': [0.01, 0.1, 0.5, 1, 5, 10, 50, 100],  
              'coef0': [0.01, 0.1, 0.5, 1, 5, 10, 50, 100],
              'kernel': ['poly']}
  
grid = GridSearchCV(SVC(), param_grid, scoring=f2_scorer, refit = True, verbose = 3, n_jobs=-1) 
  
# fitting the model for grid search
_ = grid.fit(X_train, y_train)

# print best parameter after tuning 
print(grid.best_params_) 
  
# print how our model looks after hyper-parameter tuning 
print(grid.best_estimator_)

y_pred = grid.predict(X_test) 

recall = recall_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
f2 = fbeta_score(y_test, y_pred, beta=2)

performance = pd.concat([performance, pd.DataFrame({"model": ["SVM Poly"], "Accuracy": [accuracy], "Precision": [precision], "Recall": [recall], "F1": [f1], "F2": [f2], "Parameters": [grid.best_params_]})])

# ## Summary

performance.sort_values(by="F2", ascending=False)

# #### SVM Linear
# The SVM model with a polynomial kernel achieved an accuracy of 96.67%, indicating excellent overall performance. It has a precision of 100.0%, meaning that all positive predictions are correct. However, the recall is 75.0%, indicating that it misses some positive instances. The F1 score is 85.71%, and the F2 score is 78.95%.
# 
# #### SVM Poly
# The SVM model with a polynomial kernel achieved an accuracy of 96.67%, indicating excellent overall performance. It has a precision of 100.0%, meaning that all positive predictions are correct. However, the recall is 75.0%, indicating that it misses some positive instances. The F1 score is 85.71%, and the F2 score is 78.95%.
# 
# #### SVM rbf
#  The SVM model with a radial basis function (RBF) kernel achieved an accuracy of 86.67%. However, it has a precision, recall, F1 score, and F2 score of 0.0%, indicating poor performance. This suggests that the model failed to correctly classify any positive instances.
#  
# Overall, the polynomial kernel SVM model achieved the highest accuracy and balanced precision and recall scores, making it the best-performing model among the three tested. The linear kernel SVM also performed reasonably well but had lower precision. The RBF kernel SVM performed poorly, failing to correctly classify any positive instances.


```
**Feedback:**
Data Generation (30%) - 0.4/0.45
• Creativity and relevance of the chosen scenario - 0.15/0.15
• Correct implementation of data generation techniques - 0.15/0.15
• Clarity and structure of the data (including labeling and documentation) - 0.1/0.15

Model Implementation and Tuning (40%) - 0.6/0.6
• Correct implementation of SVC/SVR models with the three specified kernels - 0.2/0.2
• Effective use of hyper-parameter tuning, with a clear rationale for chosen parameters - 0.2/0.2
• Documentation of model performance metrics, with comparisons to justify the best-performing model - 0.2/0.2

Analysis and Discussion (30%)- 0.45/0.45
• Clarity and depth of the introduction and explanation of the chosen problem and dataset - 0.225/0.225
• Critical analysis of model results, including insights into performance variations across different kernels - 0.225/0.225

Need to create a dedicated notebook for data generation as this keeps your code more organized. So, 0.05 marks were deducted

---

# Student 24
```python
# ### Data Generation File

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification

np.random.seed(1)

# Generating synthetic data
X, y = make_classification(n_samples=1000, n_features=2, n_redundant=0, n_clusters_per_class=2, random_state=42)

# Adding noise

X[:, 0] += np.abs(np.random.normal(0, 50, size=X.shape[0]))  # Adding noise to Email Length
X[:, 1] += np.abs(np.random.normal(0, 5, size=X.shape[0]))  # Adding noise to Number of Links


# Adding classification condition for spam detection
y[(X[:, 0] > 40) & (X[:, 1] > 3)] = 1
y[(X[:, 0] <= 40) | (X[:, 1] <= 3)] = 0




# Creating a DataFrame for the synthetic data
data = pd.DataFrame({'Email_Length': X[:, 0], 'Number_of_Links': X[:, 1], 'Spam_Label': y})

# Displaying the DataFrame
print(data.head())


# Counting the number of samples in the spam cluster
spam_cluster_count = np.sum(y == 1)
print("Number of samples in the spam cluster:", spam_cluster_count)


# Plotting the synthetic data
plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', marker='o', edgecolors='k')
plt.xlabel('Email Length')
plt.ylabel('Number of Links')
plt.title('Synthetic Email Spam Detection Data')
plt.grid(True)
plt.show()

data.to_csv('data_gen_svm.csv', index=False)

# # Email Spam Detection Analysis with SVM Models
# 
# ## Introduction
# Email spam detection is a critical task in today's digital age to protect users from unwanted and potentially harmful emails. In this notebook, we aim to develop a classification model to distinguish between spam and non-spam emails using Support Vector Machines (SVM) with different kernel functions.
# 
# ## Dataset
# The dataset used in this analysis consists of synthetic email features such as email length and the number of links, along with labels indicating whether an email is classified as spam or not.
# 
# ## Objective
# Our objective is to explore the performance of SVM models with different kernel functions, namely linear, radial basis function (RBF), and polynomial kernels, in classifying spam and non-spam emails. We will use hyperparameter tuning to optimize the models and compare their performance metrics to identify the most effective approach for email spam detection.

# ## 1. Setup

# Import modules

import pandas as pd
from sklearn.svm import SVC
from matplotlib import pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV

np.random.seed(1)

# ## 2. Load data

df = pd.read_csv('data_gen_svm.csv') # let's use the same data as we did in the logistic regression example
df.head(10)

# Splitting the DataFrame into features (X) and target variable (y)
X = df[['Email_Length', 'Number_of_Links']]
y = df['Spam_Label']

# Splitting the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# ## 3. Model the data

# First, let's create a dataframe to load the model performance metrics into.

performance = pd.DataFrame({"model": [], "Accuracy": [], "Precision": [], "Recall": [], "F1": [], "Parameters": []})

# ### 3.1 Fit a SVM classification model using linear kernal

# defining parameter range 
param_grid = {'C': [0.01, 0.1, 0.5, 1, 5, 10, 50, 100],  
              'kernel': ['linear']}
  

grid = GridSearchCV(SVC(), param_grid, scoring='f1', refit=True, verbose=3, n_jobs=-1)

  
# fitting the model for grid search 
_ = grid.fit(X_train, y_train)

# print best parameter after tuning 
print(grid.best_params_) 
  
# print how our model looks after hyper-parameter tuning 
print(grid.best_estimator_)

y_pred = grid.predict(X_test) 

recall = recall_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

performance = pd.concat([performance, pd.DataFrame({"model": ["SVM Linear"], "Accuracy": [accuracy], "Precision": [precision], "Recall": [recall], "F1": [f1], "Parameters": [grid.best_params_]})])


performance

# ### 3.2 Fit a SVM classification model using rbf kernal

# defining parameter range 
param_grid = {'C': [0.1, 1, 10, 100],  
              'gamma': [1, 0.1, 0.01, 0.001, 0.0001], 
              'kernel': ['rbf']}
  
grid = GridSearchCV(SVC(), param_grid, scoring='f1', refit = True, verbose = 3, n_jobs=-1) 
  
# fitting the model for grid search 
_ = grid.fit(X_train, y_train)

# print best parameter after tuning 
print(grid.best_params_) 
  
# print how our model looks after hyper-parameter tuning 
print(grid.best_estimator_)

y_pred = grid.predict(X_test) 

recall = recall_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

performance = pd.concat([performance, pd.DataFrame({"model": ["SVM rbf"], "Accuracy": [accuracy], "Precision": [precision], "Recall": [recall], "F1": [f1], "Parameters": [grid.best_params_]})])

# ### 3.3 Fit a SVM classification model using polynomial kernal

# defining parameter range 
param_grid = {'C': [0.01, 0.1, 0.5, 1, 5, 10, 50, 100],  
              'coef0': [0.01, 0.1, 0.5, 1, 5, 10, 50, 100],
              'kernel': ['poly']}
  
grid = GridSearchCV(SVC(), param_grid, scoring='f1', refit = True, verbose = 3, n_jobs=-1) 
  
# fitting the model for grid search 
_ = grid.fit(X_train, y_train)

# print best parameter after tuning 
print(grid.best_params_) 
  
 
print(grid.best_estimator_)

y_pred = grid.predict(X_test) 

recall = recall_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

performance = pd.concat([performance, pd.DataFrame({"model": ["SVM Poly"], "Accuracy": [accuracy], "Precision": [precision], "Recall": [recall], "F1": [f1], "Parameters": [grid.best_params_]})])

performance.sort_values(by="F1", ascending=False)

# ## Summary

# Discussion:
# 
# In the analysis, I aimed to build a classification model for **email spam detection** using Support Vector Machines (SVMs) with different kernel functions (linear, polynomial, and radial basis function (RBF)). The performance of each model is evaluated using various metrics including Accuracy, Precision, Recall, and F1 score.
# 
# The results indicate that the SVM model with an RBF kernel achieved the highest F1 score of 0.978, with an accuracy of 99.0%, precision of 97.8%, and recall of 97.8%. This model was trained with hyperparameters {'C': 100, 'gamma': 0.01}. The SVM model with a polynomial kernel also performed well with an F1 score of 0.968 and an accuracy of 98.5%, trained with hyperparameters {'C': 100, 'coef0': 50}. 
# 
# However, the SVM model with a linear kernel showed lower performance compared to the other two models, with an F1 score of 0.65, accuracy of 85.5%, precision of 75%, and recall of 57.4%. This suggests that the linear kernel may not be suitable for this classification task, as it failed to capture the non-linear relationships present in the data.
# 
# Overall, the analysis highlights the importance of selecting appropriate kernel functions in SVM models for optimal performance in classification tasks. In this particular case of email spam detection, non-linear kernels such as RBF and polynomial kernels outperformed the linear kernel, indicating the presence of non-linear relationships in the data that are better captured by these kernels.


```
**Feedback:**
Data Generation (30%) - 0.45/0.45
• Creativity and relevance of the chosen scenario - 0.15/0.15
• Correct implementation of data generation techniques - 0.15/0.15
• Clarity and structure of the data (including labeling and documentation) - 0.15/0.15

Model Implementation and Tuning (40%) - 0.6/0.6
• Correct implementation of SVC/SVR models with the three specified kernels - 0.2/0.2
• Effective use of hyper-parameter tuning, with a clear rationale for chosen parameters - 0.2/0.2
• Documentation of model performance metrics, with comparisons to justify the best-performing model - 0.2/0.2

Analysis and Discussion (30%)- 0.45/0.45
• Clarity and depth of the introduction and explanation of the chosen problem and dataset - 0.225/0.225
• Critical analysis of model results, including insights into performance variations across different kernels - 0.225/0.225

---

# Student 25
```python
# # Synthetic Data Generation Notebook: Student Performance Data

# ## Introduction
# In this scenario, we aim to predict student performance using a dataset that includes three features: Effort, Intelligence, and a binary target variable called Performance (1 if the student performs well, 0 otherwise). Our goal is to build a model that can effectively classify students based on these features.
# 
# We'll follow these steps to generate the data:
# 1. **Import Libraries**: Import the necessary libraries including NumPy, Pandas, and functions from sklearn.
# 2. **Generate Random Data Points**: Generates synthetic data for student attributes (effort and intelligence) and performance. This function should use make_regression from sklearn to create synthetic data with two features representing effort and intelligence.
# 3. **Create DataFrame**: Create a DataFrame data using the generated synthetic data. This DataFrame should have columns for effort, intelligence, and performance.
# 4. **Save Data to CSV**: Save the synthetic data to a CSV file named 'synthetic_student_performance_data.csv' using to_csv() method of Pandas DataFrame.

# ## Step 1: Import Libraries
# In this step, we import the necessary libraries for generating synthetic data, processing it, visualizing it, and saving it to a file.

import numpy as np
from sklearn.datasets import make_regression
import pandas as pd
import matplotlib.pyplot as plt

np.random.seed(42) 

# ## Step 2: Generate Random Data Points
# We'll generate 1000 data points for our dataset.

# Define parameters
n_samples = 1000

# Generate features and target using make_regression
X, _ = make_regression(n_samples=n_samples, n_features=2, noise=0.1, random_state=42)

# Transform the synthetic features to represent student effort and intelligence
effort = 50 + 5 * X[:, 0]  # mean=50, std=5
intelligence = 25 + 10 * X[:, 1]     # mean=25, std=10
    
# Generate synthetic data for student performance using a simple formula
performance = 0.6 * effort + 0.4 * intelligence + np.random.normal(loc=0, scale=15, size=n_samples)    

# ## Step 3: Create DataFrame and Visualize Data
# We visualize the synthetic data using scatter plots to understand the relationships between the features (effort, intelligence) and the target variable (performance). We create a subplot with two plots: one for performance vs. effort, and another for performance vs. intelligence.

# Create DataFrame
data = pd.DataFrame({'Effort': effort, 'Intelligence': intelligence, 'Performance': performance})

# Visualize the data
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.scatter(data['Effort'], data['Performance'], alpha=0.5)
plt.xlabel('Effort')
plt.ylabel('Performance')
plt.title('Performance vs Effort')

plt.subplot(1, 2, 2)
plt.scatter(data['Intelligence'], data['Performance'], alpha=0.5)
plt.xlabel('Intelligence')
plt.ylabel('Performance')
plt.title('Performance vs Intelligence')

plt.tight_layout()
plt.show()

# Define the threshold for classifying successful students
threshold = data['Performance'].mean()  # You can adjust this threshold as needed

# Define a function to classify students based on performance
def classify_performance(performance, threshold):
    if performance >= threshold:
        return 1  # Successful
    else:
        return 0  # Unsuccessful

# Apply the classification function to create a binary output variable
data['Performance'] = data['Performance'].apply(lambda x: classify_performance(x, threshold))

# Display the first few rows of the synthetic data with the binary output variable
print(data.head())

# ## Step 4: Saving Synthetic Data to a CSV File
# Finally, we save the synthetic data to a CSV file named 'synthetic_student_performance_data.csv' using the to_csv method of the Pandas DataFrame.

# Save the synthetic data to a CSV file
data.to_csv('synthetic_student_performance_data.csv', index=False)

# Display the first few rows of the synthetic data
print(data.head())



# # Hyperparameter Tuning and Model Fitting for SVM with Three Kernels
# 
# ## Assignment 3
# 
# 
# We’ll use GridSearchCV to find the best hyperparameters for our SVM model. We’ll test different kernels (rbf, linear, and polynomial) and optimize the model based on the custom beta score. The optimization process ensures that our model balances recall and precision effectively.
# 
# The choice of kernel depends on the underlying relationships in the data. Experiment with different kernels, cross-validate, and fine-tune hyperparameters to achieve optimal performance.
# 
# ### Step 1: Import necessary libraries
# In this step, we import the required libraries for data manipulation, model training, model evaluation, and hyperparameter tuning. Libraries such as pandas for handling data, scikit-learn for machine learning functionalities, and necessary modules for Support Vector Machine (SVM) are imported.

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import make_scorer, fbeta_score, recall_score, precision_score, accuracy_score, f1_score

# ### Step 2: Load the data
# The data is loaded from a CSV file using pandas read_csv function. Features (X) and the target variable (y) are separated. The data is split into training and testing sets using the train_test_split function from scikit-learn.

df = pd.read_csv('./synthetic_student_performance_data.csv') # let's use the same data as we did in the logistic regression example
df.head()

X = df[['Effort', 'Intelligence']]
y = df['Performance']

# Assuming the data is already loaded and split into features X and target y
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# ### Step 3: Define a custom scoring function
# A custom scoring function is defined using the F-beta score with beta equal to 2. This scoring function will be used to evaluate the models during hyperparameter tuning.

# Define a custom scoring function using fbeta_score with beta=2
f2_scorer = make_scorer(fbeta_score, beta=2)

# ### Step 4: Define parameter grid for GridSearchCV
# Different parameter grids are defined for each SVM kernel type (linear, radial basis function (RBF), and polynomial). These grids contain the hyperparameters that will be searched over during the hyperparameter tuning process.

param_grid_linear = {'C': [0.01, 0.1, 0.5, 1, 5, 10, 50, 100], 'kernel': ['linear']}
param_grid_rbf = {'C': [0.1, 1, 10, 100], 'gamma': [1, 0.1, 0.01, 0.001, 0.0001], 'kernel': ['rbf']}
param_grid_poly = {'C': [0.01, 0.1, 0.5, 1, 5, 10, 50, 100], 'coef0': [0.01, 0.1, 0.5, 1, 5, 10, 50, 100], 'kernel': ['poly']}

# ### Step 5: Initialize GridSearchCV for each kernel
# GridSearchCV is initialized for each kernel type (linear, RBF, and polynomial) with the corresponding parameter grid and custom scoring function. Cross-validation with 5 folds is used for hyperparameter tuning.

grid_linear = GridSearchCV(SVC(), param_grid_linear, scoring=f2_scorer, cv=5)
grid_rbf = GridSearchCV(SVC(), param_grid_rbf, scoring=f2_scorer, cv=5)
grid_poly = GridSearchCV(SVC(), param_grid_poly, scoring=f2_scorer, cv=5)

# ### Step 6: Fit the models
# The GridSearchCV objects are fitted to the training data. This step performs an exhaustive search over the specified hyperparameter values for each kernel type to find the best combination that optimizes the chosen scoring metric.

grid_linear.fit(X_train, y_train)
grid_rbf.fit(X_train, y_train)
grid_poly.fit(X_train, y_train)

# ### Step 7: Print best parameters and best estimators
# Print the best hyperparameters and best estimators (trained models) for each kernel type. These represent the optimal configurations found during the hyperparameter tuning process.

print("Best parameters for linear kernel:", grid_linear.best_params_)
print("Best parameters for rbf kernel:", grid_rbf.best_params_)
print("Best parameters for polynomial kernel:", grid_poly.best_params_)

print("Best estimator for linear kernel:", grid_linear.best_estimator_)
print("Best estimator for rbf kernel:", grid_rbf.best_estimator_)
print("Best estimator for polynomial kernel:", grid_poly.best_estimator_)

# ### Step 8: Evaluate performance on test data
# A function is defined to evaluate the performance of each model on the test data. Metrics such as accuracy, precision, recall, F1 score, and F2 score are computed and stored in a DataFrame. The evaluation metrics for each kernel type are used for comparison.

def evaluate_model(grid, best_params, X_test, y_test):
    y_pred = grid.predict(X_test)
    recall = recall_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    f2 = fbeta_score(y_test, y_pred, beta=2)
    
    # Create a DataFrame to store the evaluation metrics
    evaluation_metrics = pd.DataFrame({
        'Accuracy': [accuracy],
        'Precision': [precision],
        'Recall': [recall],
        'F1 Score': [f1],
        'F2 Score': [f2],
        'Best Parameters': [best_params]
    })
    
    return evaluation_metrics

# Evaluate models and store evaluation metrics
linear_evaluation = evaluate_model(grid_linear, grid_linear.best_params_, X_test, y_test)
rbf_evaluation = evaluate_model(grid_rbf, grid_rbf.best_params_, X_test, y_test)
poly_evaluation = evaluate_model(grid_poly, grid_poly.best_params_, X_test, y_test)

# Concatenate the evaluation metrics DataFrames
evaluation_table = pd.concat([linear_evaluation, rbf_evaluation, poly_evaluation], axis=0, keys=['Linear Kernel', 'RBF Kernel', 'Polynomial Kernel'])

# Print the concatenated table
print(evaluation_table)

# ### Performance Comparison of SVM Kernels
# 
# > ### Linear Kernel
# The SVM model with a linear kernel achieved an accuracy of 62.00%. While it demonstrated a decent precision of 67.13%, it struggled slightly with recall at 58.90%. Consequently, the F1 score, a balance between precision and recall, settled at 62.74%. The F2 score, favoring recall more than precision, stood at 60.38%. The best parameters for this model were a regularization parameter (C) of 100 with a linear kernel. When the relationship between features and student performance is approximately linear. Suppose you have student data with features like study hours, attendance, and previous grades. Linear kernels work well when these features contribute linearly to predicting final exam scores.
# 
# > ### RBF (Radial Basis Function) Kernel
# The RBF kernel SVM model yielded a similar accuracy of 62.67% compared to the linear kernel. It showcased a precision of 65.27% and a relatively higher recall of 66.87%. Consequently, the F1 score was 66.06%, indicating a balanced performance. The F2 score, emphasizing recall, was slightly higher at 66.54%. The best parameters included a C value of 100 and a gamma of 0.0001 with an RBF kernel. When the relationship is non-linear and complex. Consider student data where the impact of features isn’t straightforward. RBF kernels capture intricate patterns, such as identifying students who excel in specific subjects regardless of their overall performance1.
# 
# > ### Polynomial Kernel
# Utilizing a polynomial kernel, the SVM model achieved an accuracy of 62.33%. It demonstrated a precision of 64.20% and the highest recall among the three kernels at 69.33%. The F1 score, reflecting a balance between precision and recall, was 66.67%, and the F2 score, emphasizing recall, was 68.24%. The best parameters consisted of a C value of 0.01, a coef0 value of 100, and a polynomial kernel. When there are polynomial relationships between features and performance. Imagine a dataset with features related to student engagement, such as participation in extracurricular activities, social interactions, and study group involvement. Polynomial kernels can capture these non-linear interactions effectively
# 
# ### Accuracy: 
# Accuracy measures the proportion of correctly predicted instances (students) out of the total.
# When overall correctness is crucial, such as predicting whether a student will pass or fail a course. Suppose we want to predict whether a student will graduate based on historical data. High accuracy is essential to minimize false predictions.
# 
# ### Precision
# Precision represents the proportion of true positive predictions (correctly identified successful students) out of all positive predictions.
# When avoiding false positives (misclassifying a student as successful when they are not) is critical. Identifying students eligible for scholarships. We want to ensure that students predicted as eligible indeed meet the criteria.
# 
# ### Recall (Sensitivity)
# Recall measures the proportion of true positive predictions out of all actual positive instances (successful students). When identifying all successful students is crucial, even if it means some false positives. Detecting at-risk students who need intervention. We want to minimize missing any struggling students.
# 
# ### F1 Score
# F1 Score balances precision and recall. It considers both false positives and false negatives. When you want a single metric that considers both precision and recall. Evaluating a model that predicts student dropout. F1 Score helps find a balance between identifying dropouts and avoiding false alarms.
# 
# ### F2 Score
# Suppose we are predicting student success (e.g., passing an exam). We want to minimize false negatives (i.e., failing to identify students who need intervention) while still considering false positives (misclassifying some students as needing intervention).
# 
# ### Conclusion
# In summary, all three SVM models exhibited comparable accuracies, with slight variations in precision, recall, and F1 and F2 scores. The choice of kernel type should be made based on the specific requirements of the problem, considering factors such as the importance of precision versus recall. For instance, if identifying students at risk of failing (low recall) is crucial, the polynomial kernel may be preferred due to its higher recall. However, if a balanced performance is desired, the RBF kernel could be considered. Additionally, the linear kernel may be preferred for its interpretability and efficiency, especially in scenarios with high-dimensional data.




```
**Feedback:**
Data Generation (30%) - 0.45/0.45
• Creativity and relevance of the chosen scenario - 0.15/0.15
• Correct implementation of data generation techniques - 0.15/0.15
• Clarity and structure of the data (including labeling and documentation) - 0.15/0.15

Model Implementation and Tuning (40%) - 0.6/0.6
• Correct implementation of SVC/SVR models with the three specified kernels - 0.2/0.2
• Effective use of hyper-parameter tuning, with a clear rationale for chosen parameters - 0.2/0.2
• Documentation of model performance metrics, with comparisons to justify the best-performing model - 0.2/0.2

Analysis and Discussion (30%)- 0.45/0.45
• Clarity and depth of the introduction and explanation of the chosen problem and dataset - 0.225/0.225
• Critical analysis of model results, including insights into performance variations across different kernels - 0.225/0.225

---

# Student 26
```python
# In this assignment a synthetic dataset which predicts if an email is spam or not based on word frequencies and email length. After this I will be using the SVC classification model with linear, poly and rbf kernel. For the scoring metric I will be going for the custom beta score , with beta of 2. I want to minimize the false positives, if the model detects a useful mail as spam and the user deletes it might be an issue. 

# import the libraries
import pandas as pd
from sklearn.svm import SVC
from matplotlib import pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV

np.random.seed(1)


# ## Generating the dataset

# If the word frequency is more than 60 percent and email length is more than 100 words then it is a sapm email. Else it is not.

word_frequencies = np.random.uniform(0, 1, 1000) + np.random.normal(0, 0.1, 1000)  # adding noise
email_length = np.random.normal(100, 20,1000) + np.random.normal(0, 5, 1000)  # adding noise

spam_data = pd.DataFrame({
        'Word_Frequencies': word_frequencies,        
        'Email_Length': email_length       
    })
    
spam_data['Spam'] = np.where(
        (email_length > 100) & (word_frequencies > 0.6),
        1,  # Spam
        0   # Not spam
    )

# adding some noise to the dataset to make it more realistic
spam_data['Spam'] = spam_data['Spam']
 

# ## Checking the first few rows of the dataset 

spam_data.head(10)

# ## Split the dataset into train test split 


X = spam_data[['Word_Frequencies','Email_Length']]
y = spam_data['Spam']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# ## Standardize the dataset

scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

X_train

# ## Model the data 

dt_performance = pd.DataFrame({"model": [], "Accuracy": [], "Precision": [], "Recall": [], "F1": [], "F2": [], "Parameters": []})
# create a fbeta 2 scorer
from sklearn.metrics import make_scorer
from sklearn.metrics import fbeta_score
f2_scorer = make_scorer(fbeta_score, beta=2)


# ## Fit SVM Classifier using a linear kernel 

# defining parameter range 
param_grid = {'C': [0.01, 0.5, 1, 10, 100],  
              'kernel': ['linear']}
  
#grid = GridSearchCV(SVC(), param_grid, scoring='f1', refit = True, verbose = 3, n_jobs=-1) 

grid = GridSearchCV(SVC(), param_grid, scoring=f2_scorer, refit = True, verbose = 3, n_jobs=-1) 
  
# fitting the model for grid search 
_ = grid.fit(X_train, y_train)

# print best parameter after tuning 
print(grid.best_params_) 
  
# print how our model looks after hyper-parameter tuning 
print(grid.best_estimator_)

y_pred = grid.predict(X_test) 

recall = recall_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
f2 = fbeta_score(y_test, y_pred, beta=2)

dt_performance = pd.concat([dt_performance, pd.DataFrame({"model": ["SVM Linear"], "Accuracy": [accuracy], "Precision": [precision], "Recall": [recall], "F1": [f1], "F2": [f2], "Parameters": [grid.best_params_]})])


# ## Fit SVM Classifier using a rbf kernel

# defining parameter range 
param_grid = {'C': [0.01,0.5,1, 10,100],  
              'kernel': ['rbf']}
  
#grid = GridSearchCV(SVC(), param_grid, scoring='f1', refit = True, verbose = 3, n_jobs=-1) 

grid = GridSearchCV(SVC(), param_grid, scoring=f2_scorer, refit = True, verbose = 3, n_jobs=-1) 
  
# fitting the model for grid search 
_ = grid.fit(X_train, y_train)

# print best parameter after tuning 
print(grid.best_params_) 
  
# print how our model looks after hyper-parameter tuning 
print(grid.best_estimator_)

y_pred = grid.predict(X_test) 

recall = recall_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
f2 = fbeta_score(y_test, y_pred, beta=2)

dt_performance = pd.concat([dt_performance, pd.DataFrame({"model": ["SVM rbf"], "Accuracy": [accuracy], "Precision": [precision], "Recall": [recall], "F1": [f1], "F2": [f2],"Parameters": [grid.best_params_]})])

# ## Fit SVM Classifier using a polynomial kernel 

# defining parameter range 
param_grid = {'C': [0.01, 0.5, 10, 100],  
              'coef0': [0.01, 0.1, 0.5, 1, 5, 10, 50, 100],
              'kernel': ['poly']}
  
grid = GridSearchCV(SVC(), param_grid, scoring=f2_scorer, refit = True, verbose = 3, n_jobs=-1) 
  
# fitting the model for grid search 
_ = grid.fit(X_train, y_train)

# print best parameter after tuning 
print(grid.best_params_) 
  
# print how our model looks after hyper-parameter tuning 
print(grid.best_estimator_)

y_pred = grid.predict(X_test) 

recall = recall_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
f2 = fbeta_score(y_test, y_pred, beta=2)

dt_performance = pd.concat([dt_performance, pd.DataFrame({"model": ["SVM Poly"], "Accuracy": [accuracy], "Precision": [precision], "Recall": [recall], "F1": [f1], "F2": [f2], "Parameters": [grid.best_params_]})])

# ## Summary 

dt_performance.sort_values(by="F2", ascending=False)

# I will be going with SVM that has an rbf kernel as it gives me the highest F2 score




```
**Feedback:**
Data Generation (30%) - 0.4/0.45
• Creativity and relevance of the chosen scenario - 0.15/0.15
• Correct implementation of data generation techniques - 0.15/0.15
• Clarity and structure of the data (including labeling and documentation) - 0.1/0.15

Model Implementation and Tuning (40%) - 0.6/0.6
• Correct implementation of SVC/SVR models with the three specified kernels - 0.2/0.2
• Effective use of hyper-parameter tuning, with a clear rationale for chosen parameters - 0.2/0.2
• Documentation of model performance metrics, with comparisons to justify the best-performing model - 0.2/0.2

Analysis and Discussion (30%)- 0.275/0.45
• Clarity and depth of the introduction and explanation of the chosen problem and dataset - 0.225/0.225
• Critical analysis of model results, including insights into performance variations across different kernels - 0.05/0.225

Need to create a dedicated notebook for data generation as this keeps your code more organized and need to provide detailed analysis of the model results. So, 0.225 marks were deducted.

---

# Student 27
```python
# ## SVM Classification

# The exam pass/fail dataset is a synthetic dataset generated to study factors that contribute to students passing or failing an exam.
# 
# It contains 100 samples with three features - study hours, previous exam scores, and pass/fail label.
# 
# The study hours feature contains randomly generated values between 0 and 10 hours.
# 
# The previous scores feature contains randomly generated exam scores between 0 and 100.
# 
# The pass/fail label indicates whether a student passed the exam (1) or failed (0).
# 
# It was generated based on assumed logic that students who study over 5 hours and have previous scores above 60 will pass the exam.

# ### 1. Setup

# import modules

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV

# Setting random seed for reproducibility
np.random.seed(1)

# ### 2. Load Data

# Load data

# Define sample size
sample_size = 100

# Generate synthetic data for input columns
study_hours = np.random.uniform(0, 10, size=sample_size)  # Random study hours from 0 to 10
previous_scores = np.random.uniform(0, 100, size=sample_size)  # Random previous exam scores from 0 to 100

# Generate synthetic data for output column (pass/fail)
# Assume that students who study for more than 5 hours and have a previous score of more than 60 pass the exam
pass_fail = ((study_hours > 5) & (previous_scores > 60)).astype(int)

# Create DataFrame
df = pd.DataFrame({'study_hours': study_hours, 'previous_scores': previous_scores, 'pass_fail': pass_fail})

# Save DataFrame to CSV file
df.to_csv('exam_pass_fail_data.csv', index=False)

df.head()

# ### 3. Model the data

fig = plt.figure()
ax = fig.add_subplot()
colors = np.array(["blue", "red"])
ax.scatter(df['study_hours'], df['previous_scores'], c=colors[np.ravel(df['pass_fail'])])
ax.set_xlabel('Study Hours')
ax.set_ylabel('Previous Scores')
ax.set_title('Pass/Fail Prediction based on Study Hours and Previous Scores')
plt.show()

# # SVM Analysis

# Split the data into features (X) and target variable (y)
X = df[['study_hours', 'previous_scores']]
y = df['pass_fail']

# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# Create a dataframe to store model performance metrics
performance = pd.DataFrame({"model": [], "Accuracy": [], "Precision": [], "Recall": [], "F1": [], "F2": [], "Parameters": []})

# create a fbeta 2 scorer
from sklearn.metrics import make_scorer
from sklearn.metrics import fbeta_score
f2_scorer = make_scorer(fbeta_score, beta=2)

# ### SVM classification model using linear kernal

# defining parameter range 
param_grid = {'C': [0.01, 0.1, 0.5, 1, 5, 10, 50, 100], 'kernel': ['linear']}
  
#grid = GridSearchCV(SVC(), param_grid, scoring='f1', refit = True, verbose = 3, n_jobs=-1) 

grid = GridSearchCV(SVC(), param_grid, scoring=f2_scorer, refit = True, verbose = 3, n_jobs=-1) 
  
# fitting the model for grid search 
_ = grid.fit(X_train, y_train)

# print best parameter after tuning 
print(grid.best_params_) 
  
# print how our model looks after hyper-parameter tuning 
print(grid.best_estimator_)

y_pred = grid.predict(X_test) 

recall = recall_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
f2 = fbeta_score(y_test, y_pred, beta=2)

performance = pd.concat([performance, pd.DataFrame({"model": ["SVM Linear"], "Accuracy": [accuracy], "Precision": [precision], "Recall": [recall], "F1": [f1], "F2": [f2], "Parameters": [grid.best_params_]})])


performance

# ### SVM classification model using rbf kernal

# defining parameter range 
param_grid = {'C': [0.1, 1, 10, 100],  
              'gamma': [1, 0.1, 0.01, 0.001, 0.0001], 
              'kernel': ['rbf']}
  
grid = GridSearchCV(SVC(), param_grid, scoring=f2_scorer, refit = True, verbose = 3, n_jobs=-1) 
  
# fitting the model for grid search 
_ = grid.fit(X_train, y_train)

# print best parameter after tuning 
print(grid.best_params_) 
  
# print how our model looks after hyper-parameter tuning 
print(grid.best_estimator_)

y_pred = grid.predict(X_test) 

recall = recall_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
f2 = fbeta_score(y_test, y_pred, beta=2)

performance = pd.concat([performance, pd.DataFrame({"model": ["SVM rbf"], "Accuracy": [accuracy], "Precision": [precision], "Recall": [recall], "F1": [f1], "F2": [f2],"Parameters": [grid.best_params_]})])

performance

# ### SVM classification model using polynomial kernal

# defining parameter range 
param_grid = {'C': [0.01, 0.1, 0.5, 1, 5, 10, 50, 100],  
              'coef0': [0.01, 0.1, 0.5, 1, 5, 10, 50, 100],
              'kernel': ['poly']}
  
grid = GridSearchCV(SVC(), param_grid, scoring=f2_scorer, refit = True, verbose = 3, n_jobs=-1) 
  
# fitting the model for grid search 
_ = grid.fit(X_train, y_train)

# print best parameter after tuning 
print(grid.best_params_) 
  
# print how our model looks after hyper-parameter tuning 
print(grid.best_estimator_)

y_pred = grid.predict(X_test) 

recall = recall_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
f2 = fbeta_score(y_test, y_pred, beta=2)

performance = pd.concat([performance, pd.DataFrame({"model": ["SVM Poly"], "Accuracy": [accuracy], "Precision": [precision], "Recall": [recall], "F1": [f1], "F2": [f2], "Parameters": [grid.best_params_]})])

# ## 4. Summary

performance.sort_values(by="F2", ascending=False)

# The RBF SVM achieved the best performance with an accuracy of 96.67%, precision of 100%, recall of 85.71%, F1 score of 92.31%, and F2 score of 88.24%. It used parameters C=100 and gamma=0.001.
# 
# The polynomial SVM matched the performance of the RBF SVM on all evaluation metrics, with accuracy 96.67%, precision 100%, recall 85.71%, F1 92.31%, and F2 88.24%. Its parameters were C=0.5, coef0=50.
# 
# The linear SVM scored slightly lower than the other two models, with accuracy 93.33%, precision 100%, recall 71.43%, F1 score 83.33%, and F2 score 75.76%. Its parameter was C=50.
# 
# Overall, both the RBF and polynomial kernels provided the best results, significantly outperforming the linear SVM. The combination of high accuracy, precision, recall and F1 scores demonstrates these models' ability to classify the data points correctly while also providing good coverage of the positive cases.


```
**Feedback:**
Data Generation (30%) - 0.4/0.45
• Creativity and relevance of the chosen scenario - 0.15/0.15
• Correct implementation of data generation techniques - 0.15/0.15
• Clarity and structure of the data (including labeling and documentation) - 0.1/0.15

Model Implementation and Tuning (40%) - 0.6/0.6
• Correct implementation of SVC/SVR models with the three specified kernels - 0.2/0.2
• Effective use of hyper-parameter tuning, with a clear rationale for chosen parameters - 0.2/0.2
• Documentation of model performance metrics, with comparisons to justify the best-performing model - 0.2/0.2

Analysis and Discussion (30%)- 0.45/0.45
• Clarity and depth of the introduction and explanation of the chosen problem and dataset - 0.225/0.225
• Critical analysis of model results, including insights into performance variations across different kernels - 0.225/0.225

Need to create a dedicated notebook for data generation as this keeps your code more organized. So, 0.05 marks were deducted

---

# Student 28
```python
# ## SVC Model Generation:

# #### 1. Importing important libraries

import pandas as pd
from sklearn.svm import SVC
from matplotlib import pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV

np.random.seed(1)

# #### 2. Load data

df = pd.read_csv(r'C:\Users\Tushar\Downloads/synthesiesd_churn_model_datas.csv') # let's use the same data as we did in the logistic regression example
df.head(3)

# Use sklearn to split df into a training set and a test set

X = df[['MonthlySpend','ContractDuration']]
y = df['Churn']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# #### 3. Model the data

performance = pd.DataFrame({"model": [], "Accuracy": [], "Precision": [], "Recall": [], "F1": [], "F2": [], "Parameters": []})

# create a fbeta 2 scorer
from sklearn.metrics import make_scorer
from sklearn.metrics import fbeta_score
f2_scorer = make_scorer(fbeta_score, beta=2)

# #### 3.1 Fit a SVM classification model using linear kernal

# defining parameter range 
param_grid = {'C': [0.01, 0.1, 0.5, 1, 5, 10, 50, 100],  
              'kernel': ['linear']}
  
grid = GridSearchCV(SVC(), param_grid, scoring=f2_scorer, refit = True, verbose = 3, n_jobs=-1) 
  
# fitting the model for grid search 
_ = grid.fit(X_train, y_train)

# print best parameter after tuning 
print(grid.best_params_) 
  
# print how our model looks after hyper-parameter tuning 
print(grid.best_estimator_)

y_pred = grid.predict(X_test) 

recall = recall_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
f2 = fbeta_score(y_test, y_pred, beta=2)

performance = pd.concat([performance, pd.DataFrame({"model": ["SVM Linear"], "Accuracy": [accuracy], "Precision": [precision], "Recall": [recall], "F1": [f1], "F2": [f2], "Parameters": [grid.best_params_]})])


performance

# #### 3.2 Fit a SVM classification model using rbf kernal

# defining parameter range 
param_grid = {'C': [0.1, 1, 10, 100],  
              'gamma': [1, 0.1, 0.01, 0.001, 0.0001], 
              'kernel': ['rbf']}
  
grid = GridSearchCV(SVC(), param_grid, scoring=f2_scorer, refit = True, verbose = 3, n_jobs=-1) 
  
# fitting the model for grid search 
_ = grid.fit(X_train, y_train)

# print best parameter after tuning 
print(grid.best_params_) 
  
# print how our model looks after hyper-parameter tuning 
print(grid.best_estimator_)

y_pred = grid.predict(X_test) 

recall = recall_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
f2 = fbeta_score(y_test, y_pred, beta=2)

performance = pd.concat([performance, pd.DataFrame({"model": ["SVM rbf"], "Accuracy": [accuracy], "Precision": [precision], "Recall": [recall], "F1": [f1], "F2": [f2],"Parameters": [grid.best_params_]})])

performance

# #### 3.3 Fit a SVM classification model using polynomial kernal

# defining parameter range 
param_grid = {'C': [0.01, 0.1, 0.5, 1, 5, 10, 50, 100],  
              'coef0': [0.01, 0.1, 0.5, 1, 5, 10, 50, 100],
              'kernel': ['poly']}
  
grid = GridSearchCV(SVC(), param_grid, scoring=f2_scorer, refit = True, verbose = 3, n_jobs=-1) 
  
# fitting the model for grid search 
_ = grid.fit(X_train, y_train)

# print best parameter after tuning 
print(grid.best_params_) 
  
# print how our model looks after hyper-parameter tuning 
print(grid.best_estimator_)

y_pred = grid.predict(X_test) 

recall = recall_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
f2 = fbeta_score(y_test, y_pred, beta=2)

performance = pd.concat([performance, pd.DataFrame({"model": ["SVM Poly"], "Accuracy": [accuracy], "Precision": [precision], "Recall": [recall], "F1": [f1], "F2": [f2], "Parameters": [grid.best_params_]})])

performance

performance.sort_values(by="F2", ascending=False)

# ##### Discussion:
# 
# The SVM RBF model shows high accuracy and precision, indicating that it correctly identifies a significant portion of customers who are likely to churn. The recall value is decent, suggesting that the model captures a good proportion of actual churn cases.
# The F1 and F2 scores are well-balanced, considering both precision and recall.
# 
# The SVM Poly model performs similarly to the RBF model, with high accuracy, precision, and balanced F1 and F2 scores. The parameters suggest a lower regularization strength and a non-linear kernel (poly), indicating flexibility in handling complex relationships in the data.
# 
# The SVM Linear model shows high precision, indicating fewer false positives. It is good for situations where avoiding false positives is crucial. However, the recall is lower compared to the RBF and Poly models, suggesting that it might miss some actual churn cases.
# 
# Overall, The choice between these models depends on the business context and the cost associated with false positives and false negatives. If the cost of missing a churn case is high, the RBF or Poly models might be preferred due to their higher recall.
# If precision is more critical to avoid unnecessary interventions for non-churn cases, the SVM Linear model might be the choice.
# It's important to consider the business implications and interpretability of the model when making a final selection.



# ## Data Generation:

# Import necessary libraries
import pandas as pd
from sklearn.datasets import make_classification
import seaborn as sns
import matplotlib.pyplot as plt

# Set style for seaborn
sns.set(style="whitegrid")

# Plot the scatter plots with different colors for churned and not churned customers
plt.figure(figsize=(12, 6))

# Set a seed for reproducibility
seed = 1

# Define the number of samples and features
num_samples = 1000
num_features = 2

# Generate synthetic data for customer churn prediction
X, y = make_classification(
    n_samples=num_samples,
    n_features=num_features,
    n_informative=num_features,
    n_redundant=0,
    n_clusters_per_class=1,
    weights=[0.8],  # 80% of samples labeled as not churned, 20% as churned
    flip_y=0,
    random_state=seed
)

# Create a DataFrame to store the synthetic data with descriptive feature names
columns = ['MonthlySpend', 'ContractDuration', 'Churn']
df = pd.DataFrame(data=X, columns=['MonthlySpend', 'ContractDuration'])
df['Churn'] = y

# Display the first few rows of the synthetic data
df.head()

# ####  Discussion: 

# In the synthetic data we generated, we have two features: MonthlySpend and ContractDuration. Let's discuss how these features might affect customer churn in a hypothetical scenario:
# 
# ###### MonthlySpend:
# 
# Higher values of MonthlySpend might indicate that the customer is spending more money on the telecom services.
# A higher spending customer may be less likely to churn because they are investing more in the services and might find them valuable.
# 
# ###### ContractDuration:
# 
# ContractDuration represents the duration of the customer's contract with the telecom company. Customers with longer contract durations may be less likely to churn as they have committed to the service for a more extended period.
# 
# Longer contracts might suggest satisfaction or loyalty, making customers less prone to switching to another provider. In a real-world scenario, the impact of features on churn could be more complex and context-dependent. These interpretations are based on common assumptions, and the actual impact can vary based on the specific industry, customer behavior, and business dynamics.

# ### Visualization

plt.subplot(1, 2, 1)
sns.scatterplot(x='MonthlySpend', y='ContractDuration', hue='Churn', data=df, palette='viridis', alpha=0.7)
plt.title('Scatter Plot of MonthlySpend vs. ContractDuration')

plt.subplot(1, 2, 2)
sns.countplot(x='Churn', data=df, palette='viridis')
plt.title('Churn Count')

plt.tight_layout()
plt.show()

df.to_csv(r'C:\Users\Tushar\Downloads/synthesiesd_churn_model_datas.csv', index=False)


```
**Feedback:**
Data Generation (30%) - 0.45/0.45
• Creativity and relevance of the chosen scenario - 0.15/0.15
• Correct implementation of data generation techniques - 0.15/0.15
• Clarity and structure of the data (including labeling and documentation) - 0.15/0.15

Model Implementation and Tuning (40%) - 0.6/0.6
• Correct implementation of SVC/SVR models with the three specified kernels - 0.2/0.2
• Effective use of hyper-parameter tuning, with a clear rationale for chosen parameters - 0.2/0.2
• Documentation of model performance metrics, with comparisons to justify the best-performing model - 0.2/0.2

Analysis and Discussion (30%)- 0.45/0.45
• Clarity and depth of the introduction and explanation of the chosen problem and dataset - 0.225/0.225
• Critical analysis of model results, including insights into performance variations across different kernels - 0.225/0.225

---

# Student 29
```python
# Problem statement
# 
# For this assignment, you can choose either a regression problem, or a classification problem.
# 
# Create a data generation notebook to create synthetic data of your choice. This data should have two input columns, and one output column. 
# 
# I have provided many examples of data generation notebooks (also, Sklearn has a synthetic data generation functions). 
# 
# Create a dataset for your scenario (come up with something on your own -- what is the data representing? Be creative). 
# 
# Using the data you generated, use hyper-parameter tuning to test and fit 3 model forms. SVC/SVR with the three kernels discussed in class - linear, rbf, and poly.
# 
# Introduce your analysis in the notebook, and close with a discussion of your results and the details of your analysis.

# My data generation and analysis
# 
# I will choose classification problem for this assignment.
# I will syntesis data using sklearn.dataset make_blob function.
# 
# I will genrate data for student's take home assignment marks and final exam marks as input and based on this I will genrate output column as student is Honest or NOT:
# 
# Here are my assumption while making dataset:
# 1.  If a student gets low marks in both exam and assignment, they are Honest.
# 2.  If a student gets high marks in both exam and assignment, they are Honest.
# 3.  Otherwise, they are NOT honest.
# 
# This data will create 4 cluster with 2 different class as HONEST as TRUE and HONEST as FALSE

# Importing required libraries

import pandas as pd
import numpy as np

from sklearn.datasets import make_blobs
from matplotlib import pyplot as plt



from sklearn.svm import SVC

from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, f1_score

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV

np.random.seed(1)

# Creating a dataset

X, y = make_blobs(n_samples=100, centers=4,center_box=(1,9), n_features=2, random_state=42)

np.unique(y)

df=pd.DataFrame(data=X,columns=['Assignment marks','Exam marks'])
df['y']=y
df.head()

df[(df['y']==1) | (df['y']==2)]

# As we wanted 2 class output here output have 4 class

def change_label(x):
    if x==1 or x==3:
        x=True
    if x==0 or x==2:
        x=False
    return x

change_label(0)

df['y']=df['y'].apply(change_label)

df['y']

df[df['y']==True].head()

# Plot the generated data
plt.figure(figsize=(10, 6))
plt.scatter(df['Assignment marks'], df['Exam marks'], c=df['y'], cmap='viridis')
plt.title('marks distribution')
plt.xlabel('Assignment marks')
plt.ylabel('Exam marks')
plt.colorbar(label='Class')
plt.grid(True)
plt.show()

df.columns=['Assignment marks', 'Exam marks', 'Conclusion']

df.head()

X=df.drop(columns='Conclusion')
y=df.Conclusion
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=1,test_size=0.2)

model=SVC(kernel='linear')
svm_classifier = model.fit(X_train,y_train)

y_pred=model.predict(X_test)

from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, f1_score

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy_score: %.2f" % accuracy)

recall = recall_score(y_test, y_pred)
print("Recall: %.2f" % recall)

precision = precision_score(y_test, y_pred)
print("Precision: %.2f" % precision)

f1 = f1_score(y_test, y_pred)
print("F1: %.2f" % f1)

model=SVC(kernel='rbf')
svm_classifier = model.fit(X_train,y_train)
y_pred=model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy_score: %.2f" % accuracy)

recall = recall_score(y_test, y_pred)
print("Recall: %.2f" % recall)

precision = precision_score(y_test, y_pred)
print("Precision: %.2f" % precision)

f1 = f1_score(y_test, y_pred)
print("F1: %.2f" % f1)

model=SVC(kernel='poly')
svm_classifier = model.fit(X_train,y_train)
y_pred=model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy_score: %.2f" % accuracy)

recall = recall_score(y_test, y_pred)
print("Recall: %.2f" % recall)

precision = precision_score(y_test, y_pred)
print("Precision: %.2f" % precision)

f1 = f1_score(y_test, y_pred)
print("F1: %.2f" % f1)

# Discussion based on above results -
# 
# When examining methods to detect student honesty during tests, the SVM with RBF Kernel emerged as the most effective. It had the highest accuracy, meaning it made the most correct classifications. It also struck a good balance in identifying both honest and dishonest students without wrongly accusing too many innocent ones. 
# 
# On the other hand, the Linear SVC method, while it successfully detected all cases of cheating, often misidentified honest students as cheaters, leading to significant errors. This suggests it lacked precision and sometimes accused students unfairly. 
# 
# The Polynomial Kernel method performed decently but missed some instances of cheating. It was accurate and careful but not as effective as the RBF Kernel method, which was the best at making accurate and fair judgments regarding student honesty during tests. Therefore, the SVM with RBF Kernel stands out as the most reliable choice for discerning both honest and dishonest behavior during examinations.




```
**Feedback:**
Data Generation (30%) - 0.4/0.45
• Creativity and relevance of the chosen scenario - 0.15/0.15
• Correct implementation of data generation techniques - 0.15/0.15
• Clarity and structure of the data (including labeling and documentation) - 0.1/0.15

Model Implementation and Tuning (40%) - 0.6/0.6
• Correct implementation of SVC/SVR models with the three specified kernels - 0.2/0.2
• Effective use of hyper-parameter tuning, with a clear rationale for chosen parameters - 0.2/0.2
• Documentation of model performance metrics, with comparisons to justify the best-performing model - 0.2/0.2

Analysis and Discussion (30%)- 0.45/0.45
• Clarity and depth of the introduction and explanation of the chosen problem and dataset - 0.225/0.225
• Critical analysis of model results, including insights into performance variations across different kernels - 0.225/0.225

Need to create a dedicated notebook for data generation as this keeps your code more organized. So, 0.05 marks were deducted

---

# Student 30
```python
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
import matplotlib.pyplot as plt

np.random.seed(11)


# Generate synthetic classification data using make_classification
X, y = make_classification(n_samples=num_samples, n_features=2, n_informative=2, n_redundant=0,
                           n_clusters_per_class=1, class_sep=1.0, random_state=11)

# Map target values to labels
label_map = {0: 'Not SPAM', 1: 'SPAM'}
y_labels = [label_map[val] for val in y]

# Generate synthetic data for word frequency and email length
word_frequency = np.random.randint(0, 100, num_samples)
email_length = np.random.randint(50, 1000, num_samples)

# Create a DataFrame to store the synthetic data
data = pd.DataFrame({
    'Word Frequency': word_frequency,
    'Email Length': email_length,
    'Target': y_labels
})

# Display the first few rows of the generated synthetic data
print(data.head())

data['Target'] = data['Target'].replace({'Not SPAM': 0, 'SPAM': 1})

X = data[['Word Frequency', 'Email Length']]
y = data['Target']

# Create scatter plot
plt.figure(figsize=(8, 6))
plt.scatter(X[y == 0]['Word Frequency'], X[y == 0]['Email Length'], color='blue', label='Not SPAM')
plt.scatter(X[y == 1]['Word Frequency'], X[y == 1]['Email Length'], color='red', label='SPAM')
plt.xlabel('Word Frequency')
plt.ylabel('Email Length')
plt.title('Email Classification')
plt.legend()
plt.show()

data.head()

data.to_csv(r'C:\Users\keval\OneDrive\Desktop\Assignments\DSP Assignments\SPAM_Email_Dataset', index=False)

# # SVM Demonstration
# 
# In this tutorial we will demonstrate how to use the `SVM` class in `scikit-learn` to perform logistic regression on a dataset. 
# 
# The synthetic dataset we will use is the cancer dataset that is produced by the data_gen notebook. 
# 
# This is a simple dataset that predicts if someone has cancer based on the number of kilograms of tobacco they have smoked in total.
# 
# This dataset, therefore, has only one feature and a binary target variable (1 is they have cancer, 0 if they don't).
# 
# We will use the `SVM` class to fit a model to the data and then plot the decision boundary.
# 
# We will also use the `SVM` class to predict the probability of a person having cancer based on the number of kilograms of tobacco they have smoked.
# 
# We will use GridSearchCV to find the best hyper-parameters for the model - and we will test rbf, linear and polynomial kernels. The scoring metric we will use is custom beta score, with a beta of 2 (which means we are more interested in recall than precision).
# 
# The reason for this metric is that their is a difference between the cost of a false positive and a false negative in this case. A false negative is much more costly than a false positive, as it means someone with cancer is not being treated. But, we cannot fully ignore precision, as we don't want to be treating people who don't have cancer.

# ## 1. Setup

# Import modules

import pandas as pd
from sklearn.svm import SVC
from matplotlib import pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV

np.random.seed(11)

# ## 2. Load data

# Load data (it's already cleaned and preprocessed)

# Uncomment the following snippet of code to debug problems with finding the .csv file path
# This snippet of code will exit the program and print the current working directory.
#import os
#print(os.getcwd())

df = pd.read_csv(r'C:\Users\keval\OneDrive\Desktop\Assignments\DSP Assignments\SPAM_Email_Dataset') # let's use the same data as we did in the logistic regression example
df.head(3)

# Use sklearn to split df into a training set and a test set

X = df[['Word Frequency','Email Length']]
y = df['Target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)


# ## 3. Model the data

# First, let's create a dataframe to load the model performance metrics into.

performance = pd.DataFrame({"model": [], "Accuracy": [], "Precision": [], "Recall": [], "F1": [], "F2": [], "Parameters": []})

# create a fbeta 2 scorer
from sklearn.metrics import make_scorer
from sklearn.metrics import fbeta_score
f2_scorer = make_scorer(fbeta_score, beta=2)


# ### 3.1 Fit a SVM classification model using linear kernal

# defining parameter range 
param_grid = {'C': [0.01, 0.1, 0.5, 1, 5, 10, 50, 100],  
              'kernel': ['linear']}
  
#grid = GridSearchCV(SVC(), param_grid, scoring='f1', refit = True, verbose = 3, n_jobs=-1) 

grid = GridSearchCV(SVC(), param_grid, scoring=f2_scorer, refit = True, verbose = 3, n_jobs=-1) 
  
# fitting the model for grid search 
_ = grid.fit(X_train, y_train)

# print best parameter after tuning 
print(grid.best_params_) 
  
# print how our model looks after hyper-parameter tuning 
print(grid.best_estimator_)

y_pred = grid.predict(X_test) 

recall = recall_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
f2 = fbeta_score(y_test, y_pred, beta=2)

performance = pd.concat([performance, pd.DataFrame({"model": ["SVM Linear"], "Accuracy": [accuracy], "Precision": [precision], "Recall": [recall], "F1": [f1], "F2": [f2], "Parameters": [grid.best_params_]})])


performance

# ### 3.2 Fit a SVM classification model using rbf kernal

# defining parameter range 
param_grid = {'C': [0.1, 1, 10, 100],  
              'gamma': [1, 0.1, 0.01, 0.001, 0.0001], 
              'kernel': ['rbf']}
  
grid = GridSearchCV(SVC(), param_grid, scoring=f2_scorer, refit = True, verbose = 3, n_jobs=-1) 
  
# fitting the model for grid search 
_ = grid.fit(X_train, y_train)

# print best parameter after tuning 
print(grid.best_params_) 
  
# print how our model looks after hyper-parameter tuning 
print(grid.best_estimator_)

y_pred = grid.predict(X_test) 

recall = recall_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
f2 = fbeta_score(y_test, y_pred, beta=2)

performance = pd.concat([performance, pd.DataFrame({"model": ["SVM rbf"], "Accuracy": [accuracy], "Precision": [precision], "Recall": [recall], "F1": [f1], "F2": [f2],"Parameters": [grid.best_params_]})])

# ### 3.3 Fit a SVM classification model using polynomial kernal

# defining parameter range 
param_grid = {'C': [0.01, 0.1, 0.5, 1, 5, 10, 50, 100],  
              'coef0': [0.01, 0.1, 0.5, 1, 5, 10, 50, 100],
              'kernel': ['poly']}
  
grid = GridSearchCV(SVC(), param_grid, scoring=f2_scorer, refit = True, verbose = 3, n_jobs=-1) 
  
# fitting the model for grid search 
_ = grid.fit(X_train, y_train)

# print best parameter after tuning 
print(grid.best_params_) 
  
# print how our model looks after hyper-parameter tuning 
print(grid.best_estimator_)

y_pred = grid.predict(X_test) 

recall = recall_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
f2 = fbeta_score(y_test, y_pred, beta=2)

performance = pd.concat([performance, pd.DataFrame({"model": ["SVM Poly"], "Accuracy": [accuracy], "Precision": [precision], "Recall": [recall], "F1": [f1], "F2": [f2], "Parameters": [grid.best_params_]})])

# ## 4.0 Summary

# Important point to note here is that the data is nonlinear, hence using a linear kernel SVM is not be the best choice. Here the data is nonlinearly separable, so one should consider using a nonlinear kernel.
# The results of the SVM models indicate varying levels of performance across different kernels. The SVM model with a polynomial kernel achieved an accuracy of 47.67% and exhibited relatively high recall (77.63%) compared to precision (48.96%). The F1 score and F2 score for this model were 0.60 and 0.69, respectively. Similarly, the SVM model with an RBF kernel also achieved an accuracy of 47.67% but displayed lower precision (48.47%) and recall (51.97%). The F1 score and F2 score for this model were 0.50 and 0.51, respectively. In contrast, the SVM model with a linear kernel achieved a slightly higher accuracy of 49.33%. However, this model's precision, recall, F1 score, and F2 score were all zero, indicating poor performance in correctly identifying positive instances. This suggests that the linear SVM model may not be suitable for the dataset's characteristics, which could be nonlinear in nature. 

performance.sort_values(by="F2", ascending=False)


```
**Feedback:**
Data Generation (30%) - 0.35/0.45
• Creativity and relevance of the chosen scenario - 0.15/0.15
• Correct implementation of data generation techniques - 0.15/0.15
• Clarity and structure of the data (including labeling and documentation) - 0.05/0.15

Model Implementation and Tuning (40%) - 0.6/0.6
• Correct implementation of SVC/SVR models with the three specified kernels - 0.2/0.2
• Effective use of hyper-parameter tuning, with a clear rationale for chosen parameters - 0.2/0.2
• Documentation of model performance metrics, with comparisons to justify the best-performing model - 0.2/0.2

Analysis and Discussion (30%)- 0.225/0.45
• Clarity and depth of the introduction and explanation of the chosen problem and dataset - 0/0.225
• Critical analysis of model results, including insights into performance variations across different kernels - 0.225/0.225

The introduction and the dataset actually choosen are not the same and the num_samples variable is not defined which is throwing errors at data generation. So, this lead to 0.325 marks deduction

---

# Student 31
```python
# # WE03-SVM
# In this tutorial we will demonstrate how to use the `SVM` class in `scikit-learn` to perform logistic regression on a dataset.
# 
# The synthetic dataset we will use is the purchase_intent dataset that is produced by the data_gen notebook. 
# This is a simple dataset that predicts if someone has purchase intent based on the average order values they have purchased and the frequency of purchases during the week.
# 
# This dataset, therefore, has two features and a binary target variable (1 is they has purchase intent, 0 if they haven't).
# 
# We will use the `SVM` class to fit a model to the data and then plot the decision boundary.
# 
# We will also use the `SVM` class to predict the probability of a person having purchase intent based on the values of purchased order and the frequency of purchase during the week.
# 
# We will use GridSearchCV to find the best hyper-parameters for the model - and we will test rbf, linear and polynomial kernels. The scoring metric we will use is custom beta score, with a beta of 2 (which means we are more interested in recall than precision).

# ## 1. Setup

# Import modules

import pandas as pd
from sklearn.svm import SVC
from matplotlib import pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV

np.random.seed(1)

# ## 2. Load data

# Load data (it's already cleaned and preprocessed)

df = pd.read_csv('purchase_intent.csv') # let's use the same data as we did in the logistic regression example
df.head(25)

# Use sklearn to split df into a training set and a test set

X = df[['avg_order_value','purchase_time_perwk']]
y = df['purchase_intent']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)


# ## 3. Model the data

# First, let's create a dataframe to load the model performance metrics into.

performance = pd.DataFrame({"model": [], "Accuracy": [], "Precision": [], "Recall": [], "F1": [], "F2": [], "Parameters": []})

# create a fbeta 2 scorer
from sklearn.metrics import make_scorer
from sklearn.metrics import fbeta_score
f2_scorer = make_scorer(fbeta_score, beta=2)

# ### 3.1 Fit a SVM classification model using linear kernal

# defining parameter range 
param_grid = {'C': [0.01, 0.1, 0.5, 1, 5, 10, 50, 100],  
              'kernel': ['linear']}
  
#grid = GridSearchCV(SVC(), param_grid, scoring='f1', refit = True, verbose = 3, n_jobs=-1) 

grid = GridSearchCV(SVC(), param_grid, scoring=f2_scorer, refit = True, verbose = 3, n_jobs=-1) 
  
# fitting the model for grid search 
_ = grid.fit(X_train, y_train)

# print best parameter after tuning 
print(grid.best_params_) 
  
# print how our model looks after hyper-parameter tuning 
print(grid.best_estimator_)

y_pred = grid.predict(X_test) 

recall = recall_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
f2 = fbeta_score(y_test, y_pred, beta=2)

performance = pd.concat([performance, pd.DataFrame({"model": ["SVM Linear"], "Accuracy": [accuracy], "Precision": [precision], "Recall": [recall], "F1": [f1], "F2": [f2], "Parameters": [grid.best_params_]})])


performance

# ### 3.2 Fit a SVM classification model using rbf kernal

# defining parameter range 
param_grid = {'C': [0.1, 1, 10, 100],  
              'gamma': [1, 0.1, 0.01, 0.001, 0.0001], 
              'kernel': ['rbf']}
  
grid = GridSearchCV(SVC(), param_grid, scoring=f2_scorer, refit = True, verbose = 3, n_jobs=-1) 
  
# fitting the model for grid search 
_ = grid.fit(X_train, y_train)

# print best parameter after tuning 
print(grid.best_params_) 
  
# print how our model looks after hyper-parameter tuning 
print(grid.best_estimator_)

y_pred = grid.predict(X_test) 

recall = recall_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
f2 = fbeta_score(y_test, y_pred, beta=2)

performance = pd.concat([performance, pd.DataFrame({"model": ["SVM rbf"], "Accuracy": [accuracy], "Precision": [precision], "Recall": [recall], "F1": [f1], "F2": [f2],"Parameters": [grid.best_params_]})])

# ### 3.3 Fit a SVM classification model using polynomial kernal

# defining parameter range 
param_grid = {'C': [0.01, 0.1, 0.5, 1, 5, 10, 50, 100],  
              'coef0': [0.01, 0.1, 0.5, 1, 5, 10, 50, 100],
              'kernel': ['poly']}
  
grid = GridSearchCV(SVC(), param_grid, scoring=f2_scorer, refit = True, verbose = 3, n_jobs=-1) 
  
# fitting the model for grid search 
_ = grid.fit(X_train, y_train)

# print best parameter after tuning 
print(grid.best_params_) 
  
# print how our model looks after hyper-parameter tuning 
print(grid.best_estimator_)

y_pred = grid.predict(X_test) 

recall = recall_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
f2 = fbeta_score(y_test, y_pred, beta=2)

performance = pd.concat([performance, pd.DataFrame({"model": ["SVM Poly"], "Accuracy": [accuracy], "Precision": [precision], "Recall": [recall], "F1": [f1], "F2": [f2], "Parameters": [grid.best_params_]})])

# ## 4.0 Summary

# The comparison among linear, RBF, and poly kernels reveals that the linear kernel consistently outperforms its counterparts, namely the polynomial and radial basis function (RBF) kernels. This superiority is attributed to the simplicity of the linear decision boundary, which has proven to be effective for our specific dataset. However, the polynomial and RBF kernels introduce a higher degree of model complexity, leading to overfitting or reduced generalization on our particular data. The linear SVM's ability to discern linear separability in our dataset contributes to its overall better performance and accuracy when compared to the more intricate kernel functions

performance.sort_values(by="F2", ascending=False)




```
**Feedback:**
Data Generation (30%) - 0/0.45
• Creativity and relevance of the chosen scenario - 0/0.15
• Correct implementation of data generation techniques - 0/0.15
• Clarity and structure of the data (including labeling and documentation) - 0/0.15

Model Implementation and Tuning (40%) - 0.6/0.6
• Correct implementation of SVC/SVR models with the three specified kernels - 0.2/0.2
• Effective use of hyper-parameter tuning, with a clear rationale for chosen parameters - 0.2/0.2
• Documentation of model performance metrics, with comparisons to justify the best-performing model - 0.2/0.2

Analysis and Discussion (30%)- 0.45/0.45
• Clarity and depth of the introduction and explanation of the chosen problem and dataset - 0.225/0.225
• Critical analysis of model results, including insights into performance variations across different kernels - 0.225/0.225

The data generation file is not submitted. So, 0.45 marks were deducted.

---

# Student 32
```python
# ## Introduction
# 
# ### In this notebook, we explore the process of generating synthetic data and applying hyper-parameter tuning to fit three different model forms using Support Vector Machines (SVMs) for classification. The dataset we generated represents a hypothetical scenario where we're analyzing skin condition based on daily sun exposure and water consumption.
# 
# ### Our synthetic dataset consists of two input columns: 'Hours_Of_Sun_Exposure_Per_Day' and 'Glasses_Of_Water_Consumed_Per_Day', and one output column: 'skin_condition'. We created the dataset to simulate the relationship between sun exposure, water consumption, and skin condition.
# 
# ### Using the generated data, we conducted hyper-parameter tuning to test and fit three model forms of SVMs with different kernels: linear, radial basis function (rbf), and polynomial (poly). We utilized GridSearchCV to find the optimal hyper-parameters for each SVM model form.

# ## Importing Important Libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import make_scorer
from sklearn.metrics import fbeta_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
import seaborn as sns

# ## Setting the seed variable

np.random.seed(1)

# ## Definition of Sample Size and Range Parameters

#Size of dataset
sample_size=300

#Declaring minimum and maximum for the 'Glasses_Of_Water_Consumed_Per_Day' column
min_glasses_of_water = 0
max_glasses_of_water = 10

#Declaring minimum and maximum for the 'Hours_Of_Sun_Exposure_Per_Day' column
min_hours_of_sun_exposure = 0
max_hours_of_sun_exposure = 12

# ## Generating synthetic data for input columns

# "input column 1:Hours_Of_Sun_Exposure_Per_Day"
df = pd.DataFrame({'Hours_Of_Sun_Exposure_Per_Day': np.random.randint(min_glasses_of_water, max_glasses_of_water + 1, size=sample_size)})

# "input column 2:Glasses_Of_Water_Consumed_Per_Day"
df['Glasses_Of_Water_Consumed_Per_Day'] = np.random.randint(min_hours_of_sun_exposure, max_hours_of_sun_exposure + 1, size=sample_size)

# ## Checking if the synthetic data is properly generated for the input columns

df

# ## Generating Target Variable (Output column) by applying conditions on input columns

df['skin_condition'] = ((df['Hours_Of_Sun_Exposure_Per_Day'] <= 4) & (df['Glasses_Of_Water_Consumed_Per_Day'] >= 5)).astype(int)

# ## Verifying the correctness of the synthetic dataset generation

df

# ## Adding noise to the  input columns to avoid overfitting 

df['Hours_Of_Sun_Exposure_Per_Day']=df['Hours_Of_Sun_Exposure_Per_Day']+np.random.uniform(-5,5, sample_size)
df['Glasses_Of_Water_Consumed_Per_Day']=df['Glasses_Of_Water_Consumed_Per_Day']+np.random.uniform(-5,5, sample_size)

# ## Verifying Dataset Balance: Counting Zeros and Ones
# ### Counting the occurrences of zeros and ones to ensure dataset balance, thereby maintaining prediction accuracy.

df['skin_condition'].value_counts()

# ## Plotting the graph to visually inspect the positioning of Sunburnt and Healthy Skin categories.

plt.figure(figsize=(8, 6))
plt.scatter(df['Hours_Of_Sun_Exposure_Per_Day'], 
            df['Glasses_Of_Water_Consumed_Per_Day'], 
            c=df['skin_condition'])
plt.xlabel('Hours of Sun Exposure per Day')
plt.ylabel('Glasses of Water Consumed per Day')
plt.title('Scatter Plot of Sun Exposure vs Water Consumption')
plt.colorbar(label='Skin Condition (0: Sunburnt, 1: Healthy)')
plt.grid(True)
plt.show()

# ## Train Test Split

y=df[['skin_condition']]
X=df[['Hours_Of_Sun_Exposure_Per_Day', 'Glasses_Of_Water_Consumed_Per_Day']]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=0)

# ## Creating a variable named "performance" to store the optimal performance metrics for each model type, including Linear, RBF, and Poly.

performance = pd.DataFrame({"model": [], "Accuracy": [], "Precision": [], "Recall": [], "F1": [], "F2": [], "Parameters": []})

# ## Use of f2 scorer
# 
# ### The following line of code defines a custom scoring function named f2_scorer using scikit-learn's make_scorer function. It is configured to compute the F-beta score, where beta is set to 2. The F-beta score incorporates both precision and recall, with beta controlling the balance between the two. 
# ### A higher beta value, like 2 here, prioritizes recall, making the metric more responsive to false negatives.

f2_scorer = make_scorer(fbeta_score, beta=2)

# ## Model the data 

# ## Fit a SVM classification model using linear kernal

# defining parameter range 
param_grid = {'C': [0.01, 0.1, 0.5, 1],  
              'kernel': ['linear']}

grid = GridSearchCV(SVC(), param_grid, scoring=f2_scorer, verbose = 3, n_jobs=-1) 
  
# fitting the model for grid search 
_ = grid.fit(X_train, y_train)

# print best parameter after tuning 
print(grid.best_params_) 
  
# print how our model looks after hyper-parameter tuning 
print(grid.best_estimator_)

y_pred = grid.predict(X_test) 

recall = recall_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
f2 = fbeta_score(y_test, y_pred, beta=2)

performance = pd.concat([performance, pd.DataFrame({"model": ["SVM Linear"], "Accuracy": [accuracy], "Precision": [precision], "Recall": [recall], "F1": [f1], "F2": [f2], "Parameters": [grid.best_params_]})])


# Calculate confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Create a heatmap for the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Predicted Negative', 'Predicted Positive'], yticklabels=['Actual Negative', 'Actual Positive'])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()

# ## Analysis of SVC Linear Model
# 
# ### The SVC linear model appears to be performing well, with a high number of correctly classified instances (83 and 22) 
# ### and a low number of incorrectly classified instances (3 and 12)
# 
# ### Here is a more detailed breakdown of the confusion matrix:
# 
# ### True positives: 83. These are the instances that were correctly classified as positive by the model.
# ### False positives: 3. These are the instances that were incorrectly classified as positive by the model.
# ### True negatives: 22. These are the instances that were correctly classified as negative by the model.
# ### False negatives: 12. These are the instances that were incorrectly classified as negative by the model.
# 
# 
# ## C value:
# ### A low C value (0.01) indicates a high regularization strength. This means the model prioritizes avoiding overfitting over maximizing the margin between classes.

# ## Fit a SVM classification model using rbf kernal

# defining parameter range 
param_grid = {'C': [0.1, 1,10,100],  
              'gamma': [1, 0.1, 0.01, 0.001, 0.0001], 
              'kernel': ['rbf']}
  
grid = GridSearchCV(SVC(), param_grid, scoring=f2_scorer, refit = True, verbose = 3, n_jobs=-1) 
  
# fitting the model for grid search 
_ = grid.fit(X_train, y_train)

# print best parameter after tuning 
print(grid.best_params_) 
  
# print how our model looks after hyper-parameter tuning 
print(grid.best_estimator_)

y_pred = grid.predict(X_test) 

recall = recall_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
f2 = fbeta_score(y_test, y_pred, beta=2)

performance = pd.concat([performance, pd.DataFrame({"model": ["SVM rbf"], 
                                                    "Accuracy": [accuracy], 
                                                    "Precision": [precision], 
                                                    "Recall": [recall], 
                                                    "F1": [f1], 
                                                    "F2": [f2],
                                                    "Parameters": [grid.best_params_]})])

# Calculate confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Create a heatmap for the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Predicted Negative', 'Predicted Positive'], yticklabels=['Actual Negative', 'Actual Positive'])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()

# ## Analysis of SVC RBF Model
# ### The SVC RBF model with the parameters {'C': 100, 'gamma': 0.0001} appears to be performing well, with a total of 105 
# ### correctly classified instances (83 + 22) and only 17 incorrectly classified instances (3 + 12).
# 
# ## Analysis of C and gamma parameters in the SVC RBF model
# 
# ### The provided confusion matrix suggests good performance for the SVC model with an RBF kernel and the parameters:
# ### C = 100: This is a relatively high regularization strength.
# ### gamma = 0.0001: This is a low value for gamma.
# 
# ### Here's how these parameters might influence the model's behavior:
# ### C (regularization parameter):
# ### A high C value (100) reduces the penalty for misclassified points during training. This allows the model to create a more complex decision boundary, potentially improving its ability to fit the training data.
# ### However, a high C value also increases the risk of overfitting, especially with complex models like RBF kernels.
# ### gamma (kernel coefficient):
# 
# ### A low gamma value (0.0001) increases the smoothness of the decision boundary. This means the model is less sensitive to local variations in the data and focuses on capturing broader patterns.
# ### While this can help prevent overfitting, it might also limit the model's ability to capture intricate decision boundaries if they exist in the data.
# 
# ### Combined effect:
# ### The combination of a high C and low gamma suggests a model that:
# ### Prioritizes fitting the training data well by allowing some misclassifications during training (high C).
# ### Maintains a relatively smooth decision boundary to avoid overfitting (low gamma).

# ## Fit a SVM classification model using polynomial kernal

# defining parameter range 
param_grid = {'C': [0.01, 0.1, 0.5, 1, 5, 10, 50, 100],  
              'coef0': [0.01, 0.1, 0.5, 1, 5, 10, 50, 100],
              'kernel': ['poly']}
  
grid = GridSearchCV(SVC(), param_grid, scoring=f2_scorer, refit = True, verbose = 3, n_jobs=-1) 
  
# fitting the model for grid search 
_ = grid.fit(X_train, y_train)

# print best parameter after tuning 
print(grid.best_params_) 
  
# print how our model looks after hyper-parameter tuning 
print(grid.best_estimator_)

y_pred = grid.predict(X_test) 

recall = recall_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
f2 = fbeta_score(y_test, y_pred, beta=2)

performance = pd.concat([performance, pd.DataFrame({"model": ["SVM Poly"], "Accuracy": [accuracy], "Precision": [precision], "Recall": [recall], "F1": [f1], "F2": [f2], "Parameters": [grid.best_params_]})])

# Calculate confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Create a heatmap for the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Predicted Negative', 'Predicted Positive'], yticklabels=['Actual Negative', 'Actual Positive'])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()


# ## Analysis on SVC ploynomial kernel
# ### The SVC poly model with the parameters {'C': 50, 'coef0': 50, 'kernel': 'poly'} appears to be performing well, with a total of 99 correctly classified instances (78 + 21) and only 21 incorrectly classified instances (7 + 14).
# 
# ### Insights based on the parameters used:
# 
# ### C = 50: This is a medium regularization strength. It balances the model's ability to fit the training data and avoid overfitting.
# ### coef0 = 50: This parameter controls the offset of the polynomial kernel. A higher value can increase the model's flexibility but also the risk of overfitting.

# ## Analysis & Discussion

# ### Upon analysis, it is evident that the SVM Poly model outperformed the other models in terms of Accuracy and Precision, achieving an accuracy of 80% and a precision of 91.67%. However, the Recall score for all models appears to be relatively low, indicating that the models may have some difficulty in correctly identifying positive instances.
# 
# ### The choice of kernel also influences the performance of the SVM models. The Poly kernel yielded the highest precision but had a lower recall compared to the RBF and Linear kernels. The RBF kernel demonstrated a balanced performance across various metrics, while the Linear kernel exhibited moderate performance overall.
# 
# ### Here's why SVM Poly is considered the best choice:
# 
# ### High Precision: The SVM Poly model achieved the highest precision of 91.67% among the three models. Precision measures the proportion of true positive predictions among all positive predictions made by the model. In scenarios where minimizing false positives is critical, such as medical diagnosis or fraud detection, high precision is desirable.
# 
# ### Good Accuracy: The SVM Poly model attained an accuracy of 80%, which is comparable to or better than the other models. Accuracy measures the proportion of correct predictions made by the model. While accuracy alone doesn't provide a complete picture of model performance, achieving a reasonable level of accuracy is still important.
# 
# ### Balanced Performance: Although the SVM Poly model exhibited a slightly lower recall compared to the SVM Linear and SVM RBF models, it still maintained a reasonable level of recall at 32.35%. Recall measures the proportion of true positive predictions among all actual positive instances. While higher recall is desirable, the SVM Poly model's recall is acceptable considering its high precision.
# 
# ### Parameter Selection: The hyperparameters selected for the SVM Poly model ('C': 50, 'coef0': 50, 'kernel': 'poly') seemed to be effective in optimizing its performance based on the provided dataset. The combination of these parameters resulted in a model with strong precision and overall good performance.
# 
# ### Considering these factors, the SVM Poly model appears to strike a balance between precision, recall, and accuracy, making it the best choice among the three models for the given dataset and problem context. However, the choice of the best model ultimately depends on the specific requirements and constraints of the application domain.

performance


```
**Feedback:**
Data Generation (30%) - 0.4/0.45
• Creativity and relevance of the chosen scenario - 0.15/0.15
• Correct implementation of data generation techniques - 0.15/0.15
• Clarity and structure of the data (including labeling and documentation) - 0.1/0.15

Model Implementation and Tuning (40%) - 0.6/0.6
• Correct implementation of SVC/SVR models with the three specified kernels - 0.2/0.2
• Effective use of hyper-parameter tuning, with a clear rationale for chosen parameters - 0.2/0.2
• Documentation of model performance metrics, with comparisons to justify the best-performing model - 0.2/0.2

Analysis and Discussion (30%)- 0.45/0.45
• Clarity and depth of the introduction and explanation of the chosen problem and dataset - 0.225/0.225
• Critical analysis of model results, including insights into performance variations across different kernels - 0.225/0.225

Need to create a dedicated notebook for data generation as this keeps your code more organized. So, 0.05 marks were deducted

---

# Student 33
```python
# # WE03-SVM Assignment data_gen

# ## Security Breach Prediction 

# In this notebook, I am going to conduct security breach prediction using Support Vector Machine (SVM). First, I create synthetic data which has two inputs (packet size, failed attempts) and one output (security breach: Yes(0)/No(1)). To create a scenario more close to the real world problem, I am giving the condition to security breach. If packet size is over 1500 and failed attempts are more than 7, I identify there is a security breach. Then, I will fit SVC with three different kernels(linear, poly, rbf) and evaluate models by F2 score. Finally, I will be discussing the results and how this model can be beneficial to the businesses. 

#import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#set seed for reproducibility 
np.random.seed(1)

#use sample size of 1000
sample_size = 1000

#create synthesized data for two inputs and one output
packet_size = np.random.randint(1, 3000, sample_size)
failed_attempts = np.random.randint(1, 10, sample_size)

#set security breach yes with 0 and no with 1
#give condition and logic for security breach. If packet size is over 1500 and failed attempts are more than 7, I identify there is a security breach. 
security_breach = np.where((packet_size > 1500) | (failed_attempts > 7), 0, 1)

#create pandas data frame and show first three rows
df = pd.DataFrame({'Packet_Size': packet_size, 'Failed_Attempts': failed_attempts, 'Security_Breach': security_breach})
df.head(3)

#visualize the data
fig = plt.figure()
ax = fig.add_subplot()
ax.scatter(df.loc[df.Security_Breach== 0]['Packet_Size'], 
           df.loc[df.Security_Breach== 0]['Failed_Attempts'], 
           marker='o', 
           color='red')
ax.scatter(df.loc[df.Security_Breach== 1]['Packet_Size'], 
           df.loc[df.Security_Breach== 1]['Failed_Attempts'], 
           marker='o', 
           color='blue')

ax.legend(["0", "1"], framealpha=0.5)
                                         
ax.set_xlabel('Packet_Size')
ax.set_ylabel('Failed_Attempts')

#export to CSV file called security_breach.csv
df.to_csv('security_breach.csv', index=False)

# # WE03-SVM Assignment model_fit

# ## Security Breach Prediction 

#import necessary libraries 
import pandas as pd
from sklearn.svm import SVC
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

#set seed for reproducibility
np.random.seed(1)

# ## 1.0 Load Data

df = pd.read_csv('security_breach.csv') 
df.head(3)

# ## 2.0 Train Test Split

X = df[['Packet_Size', 'Failed_Attempts']]
y = df['Security_Breach']

#conduct train test split and assign test size 20% 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

performance = pd.DataFrame({"model": [], "Accuracy": [], "Precision": [], "Recall": [], "F1": [],  "F2":[],"Parameters": []})

#create a fbeta 2 scorer
from sklearn.metrics import make_scorer
from sklearn.metrics import fbeta_score
f2_scorer = make_scorer(fbeta_score, beta=2)

# ## 3.0 Fit Models 

from sklearn.dummy import DummyClassifier

dummy_clf = DummyClassifier(strategy="most_frequent")

dummy_clf.fit(X_train, y_train)

#Baseline Train Accuracy
dummy_train_pred = dummy_clf.predict(X_train)

baseline_train_acc = accuracy_score(y_train, dummy_train_pred)

print('Baseline Train Accuracy: {}' .format(baseline_train_acc))

#Baseline Test Accuracy
dummy_test_pred = dummy_clf.predict(X_test)

baseline_test_acc = accuracy_score(y_test, dummy_test_pred)

print('Baseline Test Accuracy: {}' .format(baseline_test_acc))

# ### 3.1 Fit a SVM using linear kernel

#define parameter range 
param_grid = {'C': [ 0.01, 0.1, 1, 10],  
              'kernel': ['linear']}

grid = GridSearchCV(SVC(), param_grid, scoring=f2_scorer, refit = True, verbose = 3, n_jobs=-1) 
  
# fit the model for grid search 
_ = grid.fit(X_train, y_train)

#print best parameter after tuning 
print(grid.best_params_) 
  
#print how our model looks after hyper-parameter tuning 
print(grid.best_estimator_)

y_pred = grid.predict(X_test) 

recall = recall_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
f2 = fbeta_score(y_test, y_pred, beta=2)

performance = pd.concat([performance, pd.DataFrame({"model": ["SVM Linear"], "Accuracy": [accuracy], "Precision": [precision], "Recall": [recall], "F1": [f1], "F2": [f2], "Parameters": [grid.best_params_]})])

# ### 3.2 Fit a SVM using RBF kernel

# defining parameter range 
param_grid = {'C': [0.1, 1, 5, 10],   
              'gamma': [1, 0.1, 0.01, 0.001], 
              'kernel': ['rbf']}
  
grid = GridSearchCV(SVC(), param_grid, scoring=f2_scorer, refit = True, verbose = 3, n_jobs=-1) 
  
# fitting the model for grid search 
_ = grid.fit(X_train, y_train)

# print best parameter after tuning 
print(grid.best_params_) 
  
# print how our model looks after hyper-parameter tuning 
print(grid.best_estimator_)

y_pred = grid.predict(X_test) 

recall = recall_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
f2 = fbeta_score(y_test, y_pred, beta=2)

performance = pd.concat([performance, pd.DataFrame({"model": ["SVM rbf"], "Accuracy": [accuracy], "Precision": [precision], "Recall": [recall], "F1": [f1], "F2": [f2],"Parameters": [grid.best_params_]})])

# ### 3.3 Fit a SVM using ploynomial kernel

# defining parameter range 
param_grid = {'C': [ 0.1, 0.5, 1, 5, 10],  
              'coef0': [0.1, 0.5, 1, 10],
              'kernel': ['poly']}
  
grid = GridSearchCV(SVC(), param_grid, scoring=f2_scorer, refit = True, verbose = 3, n_jobs=-1) 
  
# fit the model for grid search 
_ = grid.fit(X_train, y_train)

# print best parameter after tuning 
print(grid.best_params_) 
  
# print how our model looks after hyper-parameter tuning 
print(grid.best_estimator_)

y_pred = grid.predict(X_test) 

recall = recall_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
f2 = fbeta_score(y_test, y_pred, beta=2)

performance = pd.concat([performance, pd.DataFrame({"model": ["SVM Poly"], "Accuracy": [accuracy], "Precision": [precision], "Recall": [recall], "F1": [f1], "F2": [f2], "Parameters": [grid.best_params_]})])

# ## 4.0 Summary 

#summary results and order descending 
performance.sort_values(by="F2", ascending=False)

# To sum up, I have conducted security breach prediction using data includes packet size and failed attempts. I have created three SVM models using different kernels (linear, poly, rbf) and evaluated performance of each models based on F2 score. The reason I picked F2 score which gives more weight to recall than to precision was that false negative (predicting no security breach and actually there is one) costs more in this scenario. From data gen notebook, I was expecting the data is not linearly separatable. For this data, I was guessing SVC with polynomial and rbf would perform better. Based on F2 score of each models, SVM using polynoimial kernel performed the best out of three different models. SVM using linear kernel performed worst as I expected. By utilizing this model, businesses can identify security breach as soon as possible or find potential accounts that requires monitoring to prevent security breach. 


```
**Feedback:**
Data Generation (30%) - 0.45/0.45
• Creativity and relevance of the chosen scenario - 0.15/0.15
• Correct implementation of data generation techniques - 0.15/0.15
• Clarity and structure of the data (including labeling and documentation) - 0.15/0.15

Model Implementation and Tuning (40%) - 0.6/0.6
• Correct implementation of SVC/SVR models with the three specified kernels - 0.2/0.2
• Effective use of hyper-parameter tuning, with a clear rationale for chosen parameters - 0.2/0.2
• Documentation of model performance metrics, with comparisons to justify the best-performing model - 0.2/0.2

Analysis and Discussion (30%)- 0.45/0.45
• Clarity and depth of the introduction and explanation of the chosen problem and dataset - 0.225/0.225
• Critical analysis of model results, including insights into performance variations across different kernels - 0.225/0.225

---

# Student 34
```python
# # WE03-SVM-Data-Generation
# 
# 
# 
# ### Data Science Programming

# # Introduction

# This notebook is generation of data of performance of students in a course. We have two variables here Average Assignment score and Average Quiz score, with these two variables as input we are trying to predict the course result whether the student has passed or failed that course based on the formula designed.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

# We are taking a class of size 50 and the average scores of assignment and quiz can range from 0 to 100

sample_size = 50
min_val = 0
max_val = 100

# Creating dataframe with two columns avg_assignment_score and avg_quiz_score, generating those column values randomly using np.random.uniform function in numpy

df = pd.DataFrame({'avg_assignment_score': np.random.uniform(min_val,max_val,sample_size),'avg_quiz_score': np.random.uniform(min_val,max_val,sample_size)})
df

# To determine whether a student passed or failed the course we are giving 60% weightage to average assignment score and 40% weightage to average quiz score and setting the threshold value of passing as 65. If sum of 60% of average assignment score and 40% of average quiz score is greater than 65 then that student has passed the course or else failed

# Creating output field i.e., result which holds two values 0,1, 0:fail and 1: pass. Using numpy function where to assign those values to result based on the designed formula.

df['result'] = (0.6*df['avg_assignment_score'] +0.4*df['avg_quiz_score']> 65).astype(int)

df

# Visualizing the generated data using scattered plot 

fig = plt.figure()
ax = fig.add_subplot()

colors = np.array(["red", "green"])
ax.scatter(df['avg_assignment_score'], df['avg_quiz_score'], c=colors[np.ravel(df['result'])], s=50)

ax.set_xlabel('Average Assignment score')
ax.set_ylabel('Average Quiz score')

plt.show()



# Adding random noise to 'avg_assignment_score' and 'avg_quiz_score' columns

df['avg_assignment_score'] += np.random.randint(-10, 10, size=len(df))
df['avg_quiz_score'] += np.random.randint(-10, 10, size=len(df))

df


fig = plt.figure()
ax = fig.add_subplot()

colors = np.array(["red", "green"])
ax.scatter(df['avg_assignment_score'], df['avg_quiz_score'], c=colors[np.ravel(df['result'])], s=50)

ax.set_xlabel('Average Assignment score')
ax.set_ylabel('Average Quiz score')

plt.show()



# Creating the csv file which can be used in next analysis notebook where we do the model fitting

df.to_csv('./data/course_result.csv', index=False)

# # WE03-SVM-Model-Fitting
# 
# 
# 
# ### Data Science Programming

# # Introduction
# 
# In this notebook we will use hyper-parameter tuning to test and fit 3 model forms. SVC/SVR with the three kernels  linear, rbf, and poly.
# 
#  we will use The synthetic dataset we generated in  WE03-SVM-Data-Generation notebook. 
# 
# This is a simple dataset that predicts if student has passed the course or not based on average assignment and quiz scores.
# 
# This dataset, therefore, has two features and a binary target variable (P is Pass, F is Fail).
# 
# We will use GridSearchCV to find the best hyper-parameters for the model - and we will test rbf, linear and polynomial kernels. We choose F1_score and accuracy as our metrics as the costs of false positives and false negatives is same. 

# ## 1. Setup

# Import modules

import pandas as pd
from sklearn.svm import SVC
from matplotlib import pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

np.random.seed(42)

# ## 2. Load data

# Load data ( since it is synthesised it's already cleaned and preprocessed)

df = pd.read_csv('./data/course_result.csv') # let's use the same data as we did in the logistic regression example
df.head(3)

# Use sklearn to split df into a training set and a test set

X = df[['avg_assignment_score','avg_quiz_score']]
y = df['result']

# since it's a small dataset we are splitting in 70:30 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)


# ## 3. Model the data

# First, let's create a dataframe to load the model performance metrics into.

performance = pd.DataFrame({"model": [], "Accuracy": [], "F1": [], "Parameters": []})

# ### 3.1 Fit a SVM classification model using linear kernal

# defining parameter range 
param_grid = {'C': [0.01, 0.1, 0.5, 1, 5, 10, 50, 100],  
              'kernel': ['linear']}
  
#grid = GridSearchCV(SVC(), param_grid, scoring='f1', refit = True, verbose = 3, n_jobs=-1) 

grid = GridSearchCV(SVC(), param_grid, scoring='f1', refit = True, verbose = 3, n_jobs=-1) 
  
# fitting the model for grid search 
_ = grid.fit(X_train, y_train)

# print best parameter after tuning 
print(grid.best_params_) 
  
# print how our model looks after hyper-parameter tuning 
print(grid.best_estimator_)

y_pred = grid.predict(X_test) 

accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)


performance = pd.concat([performance, pd.DataFrame({"model": ["SVM Linear"], "Accuracy": [accuracy],  "F1": [f1],  "Parameters": [grid.best_params_]})])


performance

# ### 3.2 Fit a SVM classification model using rbf kernal

# defining parameter range 
param_grid = {'C': [0.1, 1, 10, 100],  
              'gamma': [1, 0.1, 0.01, 0.001, 0.0001], 
              'kernel': ['rbf']}
  
grid = GridSearchCV(SVC(), param_grid, scoring='f1', refit = True, verbose = 3, n_jobs=-1) 
  
# fitting the model for grid search 
_ = grid.fit(X_train, y_train)

# print best parameter after tuning 
print(grid.best_params_) 
  
# print how our model looks after hyper-parameter tuning 
print(grid.best_estimator_)

y_pred = grid.predict(X_test) 

accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)


performance = pd.concat([performance, pd.DataFrame({"model": ["SVM rbf"], "Accuracy": [accuracy],  "F1": [f1],  "Parameters": [grid.best_params_]})])


performance

# ### 3.3 Fit a SVM classification model using polynomial kernal

# defining parameter range 
param_grid = {'C': [0.01, 0.1, 0.5, 1, 5, 10, 50, 100],  
              'coef0': [0.01, 0.1, 0.5, 1, 5, 10, 50, 100],
              'kernel': ['poly']}
  
grid = GridSearchCV(SVC(), param_grid, scoring='f1', refit = True, verbose = 3, n_jobs=-1) 
  
# fitting the model for grid search 
_ = grid.fit(X_train, y_train)

# print best parameter after tuning 
print(grid.best_params_) 
  
# print how our model looks after hyper-parameter tuning 
print(grid.best_estimator_)

y_pred = grid.predict(X_test) 

accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)


performance = pd.concat([performance, pd.DataFrame({"model": ["SVM poly"], "Accuracy": [accuracy],  "F1": [f1],  "Parameters": [grid.best_params_]})])


performance

# ## 4.0 Conclusion

performance.sort_values(by="F1", ascending=False)

# 
# 
# 1. **SVM Linear**:
#    - Accuracy: 93.33%
#    - F1 Score: 85.71%
#    - Parameters: {'C': 0.1, 'kernel': 'linear'}
#    - This model achieved high accuracy and F1 score using a linear kernel with a regularization parameter (C) of 0.1. The linear kernel is suitable for linearly separable data.
# 
# 2. **SVM Poly**:
#    - Accuracy: 93.33%
#    - F1 Score: 85.71%
#    - Parameters: {'C': 0.1, 'coef0': 10, 'kernel': 'poly'}
#    - This model also achieved high accuracy and F1 score, utilizing a polynomial kernel with a regularization parameter (C) of 0.1 and a coefficient (coef0) of 10. since this is linearly separable data The polynomial kernel performs similar to linear kernel.
# 
# 3. **SVM RBF**:
#    - Accuracy: 86.67%
#    - F1 Score: 66.67%
#    - Parameters: {'C': 100, 'gamma': 0.001, 'kernel': 'rbf'}
#    - This model achieved relatively lower accuracy and F1 score compared to the linear and polynomial kernels. It used a radial basis function (RBF) kernel with a regularization parameter (C) of 100 and a gamma value of 0.001. The RBF kernel is versatile and can capture complex relationships but may be sensitive to parameter tuning.
# 
# Overall, the SVM models with linear and polynomial kernels performed better than the RBF kernel-based model on this dataset, as indicated by higher accuracy and F1 scores. The choice of kernel and associated parameters significantly influences the model's performance, and further optimization may be warranted to improve results further.


```
**Feedback:**
Data Generation (30%) - 0.45/0.45
• Creativity and relevance of the chosen scenario - 0.15/0.15
• Correct implementation of data generation techniques - 0.15/0.15
• Clarity and structure of the data (including labeling and documentation) - 0.15/0.15

Model Implementation and Tuning (40%) - 0.6/0.6
• Correct implementation of SVC/SVR models with the three specified kernels - 0.2/0.2
• Effective use of hyper-parameter tuning, with a clear rationale for chosen parameters - 0.2/0.2
• Documentation of model performance metrics, with comparisons to justify the best-performing model - 0.2/0.2

Analysis and Discussion (30%)- 0.45/0.45
• Clarity and depth of the introduction and explanation of the chosen problem and dataset - 0.225/0.225
• Critical analysis of model results, including insights into performance variations across different kernels - 0.225/0.225

---

# Student 35
```python
# In this notebook we will generate a synthetic dataset and fit with two features and one target. This data set has been used to test and fit the 3 forms of Support vector Classification(SVC) using Hyperparameter tuning.

import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, f1_score
from sklearn.model_selection import train_test_split
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
from sklearn.metrics import fbeta_score
np.random.seed(1)

# Imported the essential libraries for data handling, model building, and evaluation. 

# ### Step 1 : Generation of Data set

Temperature  = np.random.randint(low=0, high=50, size=1000)
Humidity = np.random.randint(low=0, high=100, size=1000)

# Initially a Synthetic data set with two features Temperature and Humidity has been generated using  numpy.random.randint function to generate random integers within a specified range as mentioned above.

df = pd.DataFrame({'Temperature': Temperature, 'Humidity': Humidity})

# Dataset has been loaded into the dataframe. 

# As we can see the dataset is a collection of both temperatures and humidity rates for each day with  1000 instances.

df['Rainfall'] = np.where((df['Temperature'] > 25) & (df['Humidity'] > 70), 1, 0)


# Target variable has been defined based on the relation of both features based on whether there will be rainfall on that particular day or not. Rainfall is said to occure when the temperature rate is more than 25 degrees and humidity rate is more than 70%. So the target rainfall variable is defined as in a mathematical relation as shown above to define whether it will rain on that particular day or not.
# 
# Now on an overall the dataset is a collection of temperatures and humidity values for each day to define whether it will rain or not on a particular day.

# ### Step 2 : Data preprocessing

df.head()

df.shape

df.describe()

df.isna().sum()

# Since this a generated synthetic dataset  hence it does not contain null values.

df.info()

df['Rainfall'].value_counts()

# Here we can see that the data is heavily imbalanced and will not give yield proper results when fit without data balancing.

# ### Step 3 : Splitting the data into train data and test data

features=df.drop(columns=['Rainfall'])
Target= df['Rainfall']

X_train,X_test,y_train,y_test=train_test_split(features,Target,test_size=0.2,random_state=42)

# Here the df is divided into features and target and using train test split function the df has been split into train data and test data with a split size of 0.2.

# ### Step 4: Data balancing 

# Random Under sampler is a type of balancing technique which is used when there is a heavy imbalance in the data and majority class samples have to be removed to balance the dataframe. Here the case is similar so choosing Random Under sampler is seemingly the best option. 

rus = RandomUnderSampler(random_state=42)
X_res, y_res = rus.fit_resample(X_train, y_train)

performance = pd.DataFrame({"model": [], "Accuracy": [], "Precision": [], "Recall": [], "F1": [], "F2": [], "Parameters": []})

# Performance is a seperate dataframe used created to store the performace metrics of all the three forms SVC into one single space for easy inference.

# create a fbeta 2 scorer
f2_scorer = make_scorer(fbeta_score, beta=2)

# In this scenario, the challenge is to construct a custom scorer for the F2 score, a version of the F1 score that prioritises recall over precision. This is beneficial in unbalanced classification issues when one class is more significant than the others.The parameter beta in the F-beta score allows you to assign different weights to precision and recall when calculating the harmonic mean. When f beta is 1 precision and recal is given equal weightage and when fbeta is 2 recall is given more weightage than precision.

# ### Step 5.1: Fit a SVM classification model using linear kernal

# The method for the SVM model is to determine the ideal hyperplane that divides the classes in the feature space in a linear manner. This hyperplane has the greatest margin, which is the distance between the hyperplane and the closest data points in each class, also known as support vectors.

# defining parameter range 
param_grid = {'C': [0.001,0.01, 0.1, 0.5, 1, 5, 10, 50, 100],  
              'kernel': ['linear']}
  
#grid = GridSearchCV(SVC(), param_grid, scoring='f1', refit = True, verbose = 3, n_jobs=-1) 

grid = GridSearchCV(SVC(), param_grid, scoring=f2_scorer, refit = True, verbose = 3, n_jobs=-1) 
  
# fitting the model for grid search 
_ = grid.fit(X_res, y_res)

# Data is undergoing cross validation to find the best hyperparameters for the model. The data is being split into 5 parts of 9 candidates each and a total of 45 model fits will be performed to find the best hyperparameters for the SVM linear kernal model.

# print best parameter after tuning 
print(grid.best_params_) 
  
# print how our model looks after hyper-parameter tuning 
print(grid.best_estimator_)

y_pred = grid.predict(X_test) 

recall = recall_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
f2 = fbeta_score(y_test, y_pred, beta=2)

performance = pd.concat([performance, pd.DataFrame({"model": ["SVM Linear"], "Accuracy": [accuracy], "Precision": [precision], "Recall": [recall], "F1": [f1], "F2": [f2], "Parameters": [grid.best_params_]})])


# Here for the case of SVM Linear kernel the ideal Hyperparameters are C = 0.01 and kernel = linear

# ### Step 5.2 : Fit a SVM classification model using rbf kernal

# SVM model with a radial basis function (RBF) kernel aims to find the optimal non-linear decision boundary that separates the classes in the feature space.The RBF kernel is frequently used in SVMs for non-linear classification applications because it can capture complicated relationships between features.

# defining parameter range 
param_grid = {'C': [0.1, 1, 10, 100],  
              'gamma': [1, 0.1, 0.01, 0.001, 0.0001], 
              'kernel': ['rbf']}
  
grid = GridSearchCV(SVC(), param_grid, scoring=f2_scorer, refit = True, verbose = 3, n_jobs=-1) 
  
# fitting the model for grid search 
_ = grid.fit(X_res, y_res)

# The data is being split into 5 parts of 20 candidates each and a total of 100 model fits will be performed to find the best hyperparameters for the SVM rbf kernal model

# print best parameter after tuning 
print(grid.best_params_) 
  
# print how our model looks after hyper-parameter tuning 
print(grid.best_estimator_)

y_pred = grid.predict(X_test) 

recall = recall_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
f2 = fbeta_score(y_test, y_pred, beta=2)

performance = pd.concat([performance, pd.DataFrame({"model": ["SVM rbf"], "Accuracy": [accuracy], "Precision": [precision], "Recall": [recall], "F1": [f1], "F2": [f2],"Parameters": [grid.best_params_]})])


# The ideal hyperparameters for SVM rbf kernel model are C = 1 , gamma = 0.01.

# ### Step 5.3 : Fit a SVM classification model using polynomial kernal

# SVM model with a polynomial kernel seeks to determine the best decision boundary that divides the classes by mapping the input features into a higher-dimensional space using polynomial functions. 

# defining parameter range 
param_grid = {'C': [0.01, 0.1, 0.5, 1, 5, 10, 50, 100],  
              'coef0': [0.01, 0.1, 0.5, 1, 5, 10, 50, 100],
              'kernel': ['poly']}
  
grid = GridSearchCV(SVC(), param_grid, scoring=f2_scorer, refit = True, verbose = 3, n_jobs=-1) 
  
# fitting the model for grid search 
_ = grid.fit(X_res, y_res)

# The data is being split into 5 parts of 64 candidates each and a total of 320 model fits will be performed to find the best hyperparameters for the SVM polynomial kernal model.

# print best parameter after tuning 
print(grid.best_params_) 
  
# print how our model looks after hyper-parameter tuning 
print(grid.best_estimator_)

y_pred = grid.predict(X_test) 

recall = recall_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
f2 = fbeta_score(y_test, y_pred, beta=2)

performance = pd.concat([performance, pd.DataFrame({"model": ["SVM Poly"], "Accuracy": [accuracy], "Precision": [precision], "Recall": [recall], "F1": [f1], "F2": [f2], "Parameters": [grid.best_params_]})])

# The ideal Hyperparameters for the SVM polynomial kernel are  C = 0.5, coef0 = 100, kernal = poly.

# ### Summary

# I have generated a synthetic dataset using numpy.random.randint with features temperature and Humidity to predict whether it will rain or not on a particular day. Data preprocessing is done and balancing technique random under sampler has been applied to remove the data imbalance. The forms of SVC have been fit to the dataset while also tuning for best hyperparameters.

performance

# The performances of all the three forms of SVM have been noted and shown  in the performance dataframe which we have created initially. 
# - The linear SVM model has high recall (1.0) but lower precision (0.6739), resulting in an F1 score of 0.8052 and an F2 score of 0.9118.
# - The SVM with an RBF kernel has high precision (0.9091) and relatively high recall (0.9677), resulting in a high F1 score of 0.9375 and an F2 score of 0.9554.
# - The SVM with a polynomial kernel also has high precision (0.8824) and recall (0.9677), resulting in an F1 score of 0.9231 and an F2 score of 0.9494.
# 
# 
# Based on these considerations, if we're looking for the best overall performance across different metrics, the SVM with the RBF kernel seems to be the best choice with an overall accuracy of 0.98.The best hyper parameters for the rbf kernel are C = 1 and gamma = 0.01.Hence we can tell that this model algorithm is a better fit when compared to other kernel SVMs in defining the the occurance of rainfall on a particular day.


```
**Feedback:**
Data Generation (30%) - 0.4/0.45
• Creativity and relevance of the chosen scenario - 0.15/0.15
• Correct implementation of data generation techniques - 0.15/0.15
• Clarity and structure of the data (including labeling and documentation) - 0.1/0.15

Model Implementation and Tuning (40%) - 0.6/0.6
• Correct implementation of SVC/SVR models with the three specified kernels - 0.2/0.2
• Effective use of hyper-parameter tuning, with a clear rationale for chosen parameters - 0.2/0.2
• Documentation of model performance metrics, with comparisons to justify the best-performing model - 0.2/0.2

Analysis and Discussion (30%)- 0.45/0.45
• Clarity and depth of the introduction and explanation of the chosen problem and dataset - 0.225/0.225
• Critical analysis of model results, including insights into performance variations across different kernels - 0.225/0.225

Need to create a dedicated notebook for data generation as this keeps your code more organized. So, 0.05 marks were deducted

---

# Student 36
```python
# # Synthetic Data Generation Notebook for WE03-SVM Regression

# ## Introduction:
# 
# The assignment requested I genrate a dataset with the following specifications: two input column features, and one output column feature.
# 
# **Assumed Context:**
# 
# I am generating a dataset for E-commerce Shipping Time Prediction with the following features:
# -Input columns - Distance to destination, Package weight
# -Output column - Delivery time (hours).

# ## Importing Libraries and setting a seed for consistensy and replication

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(1)


# # Generating the actual data

# 
# **Assumed Realtionship**: I will assume a linear realtionship between the time taking the delivery and the ditance and weight of the package.  
# **Equation:** Delivery time (hours)=c + a × Distance to destination + b × Package weight
# where 
# **Coefficients:**
# - **c** is the base number of hours it takes after the ordering to begin shipping **(48 hours in this case)**
# - **b** is the time incerase or decrease to ship a package with a unit change in weight, I will assume for every pound it takes 15 more mins to reach its destination **(value: 0.25 hours/ pound)**
# - **a** is the realtionship between how fast we can cover the distance, in this case I am assuming, we have a constant speed and are covering the distance by a car assuming no traffic and no latency with a standard speed limit of 20 miles/ hour. (**0.05 hours /mile** that is the number of hours it takes to cover one mile of distance).
# **Range of input values:**
# - **distance:**(to destination) mean: 650, standard deviation = 200
# - (Package) **weight:** max: 200 pounds, min: 1 pound, mean: 80 pounds
# 

sample_size = 1000
stdev_distance = 200
mean_distance = 650
min_val_weight = 1
max_val_weight = 200

# Using the previously defined values I am trying a to simulate  a randomly shuffeld weight array to simulate differnt sizedm packages and no visible increasing trend in the weight data, and a normally distributed distribution of distance data

weight = np.ndarray.astype(np.linspace(min_val_weight,max_val_weight,sample_size),int)
np.random.shuffle(weight)
df = pd.DataFrame({'distance': np.round(np.random.normal(mean_distance, stdev_distance, sample_size), 1),'weight':weight })
df

# ## Calculating the deilvery_time 
# **Assumed equation:** [delivery_time] = 48 hours + 0.05 hour/mile * [distance] + 0.25 hour/pound * [weight]

df['delivery_time'] = (48 + (0.05 * df['distance']) + (0.25 * df['weight']))

df['delivery_time'].describe()

# ## Adding noise into the data

df['delivery_time'] = np.round(df['delivery_time'] + np.random.uniform(-10,10, sample_size),0).astype('int')
df

#pip install ipympl

%matplotlib widget 

fig = plt.figure()
ax = fig.add_subplot(projection='3d')

ax.scatter(df['distance'], df['weight'], df['delivery_time'])

ax.set_xlabel('distance')
ax.set_ylabel('weight')
ax.set_zlabel('delivery_time')
plt.tight_layout()
plt.show()

# **Analysis:**
# as we can observe in the graph almost all the points appear to be in a plane when looking from the top. Now looking at the data perpendicular to the assumed plane rather than just seeing a singlr thin line of dotd we see that they are uniformly distributed in a thin strip this is because of the uniform +10, -10 hours we applied earlier.

df.to_csv('./data/delivery_time.csv', index=False)

# # SVM Application for E-commerce Shipping Time Prediction

# ## Introduction
# 
# Following the generation of our synthetic dataset for E-commerce Shipping Time Prediction, this notebook is dedicated to applying and analyzing Support Vector Machine (SVM) models to our dataset. Our focus will be on predicting the delivery time of packages, a regression problem, utilizing the SVM regression variant, Support Vector Regression (SVR).
# 
# 
# - **Objective:** Apply SVM models to predict delivery times for an e-commerce shipping dataset.
# - **Dataset Features:** Distance to destination and package weight.
# - **Target Variable:** Delivery time in hours.
# - **Analysis Focus:**
#     - Examining the effect of different SVM kernels (linear, RBF, poly) on prediction accuracy.
#     - Utilizing hyper-parameter tuning to enhance model performance.

# ## Importing modules

import pandas as pd
from sklearn.svm import SVR
from matplotlib import pyplot as plt
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV

np.random.seed(1)

# ## 2. Load data

# as it is synthetically generated there in need to furture clean and process the data. So we can import and move on with our application.

df = pd.read_csv('./data/delivery_time.csv') # let's use the same data as we did in the logistic regression example
df.head(3)

# ## Train Test Split:
# I am choosing a train test split of 20% for the following reasons
# - Balanced Dataset Size: Using a test size of 0.2 provides a good balance between training and testing datasets, ensuring enough data for model training while still having a substantial amount to validate model performance.
# - Sufficient Testing Data: With 1000 observations, a 0.2 split ensures 200 observations for testing, which is adequate to assess model accuracy and generalizability without significantly reducing the training set size.
# - Avoid Overfitting: A larger training set (80%) helps in building a more accurate model while the testing set (20%) is sufficient to evaluate overfitting.

# Use sklearn to split df into a training set and a test set

X = df[['distance','weight']]
y = df['delivery_time']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


# ## Choosing Performance metrics for the data
# 
# To evaluate and compare the performance of our tuned SVM models, we can  will consider several metrics: 
# - Mean Absolute Error (MAE), 
# - Mean Squared Error (MSE), and the 
# - R-squared score.
# 
# but I will be optimizing on MSE as it optimizes on the overall accuracy on delivery times.

performance = pd.DataFrame({"model": [], "MSE": [], "MAE": [], "R2": [], "Parameters": []})

# # Modelling and Hyperparameter tuning

# ##  SVM Regression model using linear kernal 

# defining parameter range 
param_grid = {'C': [0.01, 0.1, 0.5, 1, 5, 10, 50, 100],  
              'kernel': ['linear']}
  

grid = GridSearchCV(SVR(), param_grid, scoring='neg_mean_squared_error', refit = True, verbose = 3, n_jobs=-1) 
  
# fitting the model for grid search 
_ = grid.fit(X_train, y_train)

# print best parameter after tuning 
print(grid.best_params_) 
  
# print how our model looks after hyper-parameter tuning 
print(grid.best_estimator_)

y_pred = grid.predict(X_test) 

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)


performance = pd.concat([performance, pd.DataFrame({"model": ["SVM Linear"], "MSE": [mse], "MAE": [mae], "R2": [r2], "Parameters": [grid.best_params_]})])


performance

# ##   SVM regression model using rbf kernal

# defining parameter range 
param_grid = {'C': [0.1, 1, 10, 100],  
              'gamma': [1, 0.1, 0.01, 0.001, 0.0001], 
              'kernel': ['rbf']}
  
grid = GridSearchCV(SVR(), param_grid, scoring='neg_mean_squared_error', refit = True, verbose = 3, n_jobs=-1) 
  
# fitting the model for grid search 
_ = grid.fit(X_train, y_train)

# print best parameter after tuning 
print(grid.best_params_) 
  
# print how our model looks after hyper-parameter tuning 
print(grid.best_estimator_)

y_pred = grid.predict(X_test) 

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)



performance = pd.concat([performance, pd.DataFrame({"model": ["SVM rbf"], "MSE": [mse], "MAE": [mae], "R2": [r2],"Parameters": [grid.best_params_]})])

# ## SVM classification model using polynomial kernal

# defining parameter range 
param_grid = {'C': [0.01, 0.1, 0.5, 1, 5, 10, 50, 100],  
              'coef0': [0.01, 0.1, 0.5, 1, 5, 10, 50, 100],
              'kernel': ['poly']}
  
grid = GridSearchCV(SVR(), param_grid, scoring='neg_mean_squared_error', refit = True, verbose = 3, n_jobs=-1) 
  
# fitting the model for grid search 
_ = grid.fit(X_train, y_train)

# print best parameter after tuning 
print(grid.best_params_) 
  
# print how our model looks after hyper-parameter tuning 
print(grid.best_estimator_)

y_pred = grid.predict(X_test) 

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)



performance = pd.concat([performance, pd.DataFrame({"model": ["SVM Poly"], "MSE": [mse], "MAE": [mae], "R2": [r2], "Parameters": [grid.best_params_]})])

# # Performances of each model

performance.sort_values(by="MSE", ascending=True)

# # Summary and Conclusion
# 
# - **Model Performance:** Linear and Polynomial kernels showed the best performance, optimized for Mean Squared Error (MSE) to enhance accuracy.
# 
# ### Pros and Cons of Each Model Choice:
# - **Linear Kernel:** Chosen for its simplicity and ease of interpretation. It works well for linearly separable data.
# - **Polynomial Kernel:** Selected for its ability to handle non-linear relationships, offering flexibility in modeling complex patterns.
# - **RBF Kernel:** Though it is a powerful kernel capable of complex modelings, it did not perform as expected in our case, likely due to overfitting or the specific characteristics of our data.
# 
# ### Pros and Cons of Each Metric Choice:
# - **MSE (Mean Squared Error):** Focuses on penalizing larger errors more heavily, making it suitable for our regression problem where accuracy in predicting delivery times is critical. However, it can be sensitive to outliers.
# - **MAE (Mean Absolute Error):** Provides a straightforward measure of error magnitude without heavily penalizing larger errors, offering a more robust metric against outliers compared to MSE. Its downside is that it might not reflect the performance on datasets with large errors well.
# - **R2 (R-Squared):** Indicates the proportion of variance in the dependent variable that is predictable from the independent variables. While it gives a good indication of fit quality, it doesn't specify the error magnitude.
# 
# ### Our Result:
# Upon comparing the linear and polynomial models, both have their advantages. The linear model's simplicity and interpretability make it highly valuable for straightforward problems or when explaining the model to stakeholders is necessary. The polynomial model's flexibility is advantageous for capturing more complex relationships in the data, although at the risk of overfitting.
# 
# ### Linear vs. Polynomial - Pros, Cons, and Output Comparison:
# - **Linear Kernel:** Its main advantage lies in simplicity and lower risk of overfitting, making it highly efficient for datasets where the relationship between the variables is approximately linear.
# - **Polynomial Kernel:** Offers the ability to capture complex relationships but requires careful tuning of parameters to avoid overfitting.
# 
# ### Why Choosing Linear is Better than Any Other Model in the Final Output:
# The linear model's simplicity, efficiency, and ease of interpretation often make it the preferred choice, especially in a business context where decisions need to be explained to non-technical stakeholders. It strikes a balance between accuracy and model complexity, ensuring that the model is both practical and reliable.
# 
# ### What These advantages mean in the context of an e-commerce platform:
# In the context of e-commerce shipping time prediction, choosing the right model impacts not only the accuracy of predictions but also the operational efficiency and customer satisfaction. A linear model, with its balance of simplicity and effectiveness, supports timely and reliable delivery predictions. This reliability is crucial for planning, resource allocation, and enhancing the overall customer experience by setting realistic expectations for delivery times.




```
**Feedback:**
Data Generation (30%) - 0.45/0.45
• Creativity and relevance of the chosen scenario - 0.15/0.15
• Correct implementation of data generation techniques - 0.15/0.15
• Clarity and structure of the data (including labeling and documentation) - 0.15/0.15

Model Implementation and Tuning (40%) - 0.6/0.6
• Correct implementation of SVC/SVR models with the three specified kernels - 0.2/0.2
• Effective use of hyper-parameter tuning, with a clear rationale for chosen parameters - 0.2/0.2
• Documentation of model performance metrics, with comparisons to justify the best-performing model - 0.2/0.2

Analysis and Discussion (30%)- 0.45/0.45
• Clarity and depth of the introduction and explanation of the chosen problem and dataset - 0.225/0.225
• Critical analysis of model results, including insights into performance variations across different kernels - 0.225/0.225

---

# Student 37
```python
# # WE03-SVM

# In this notebook, we'll generate a synthetic dataset regarding credit card approvals based on income and credit scores. The classification model created will help determine customers if they will be approved for a credit card. We'll utilize various versions of SVM classifiers to train our data and compare their respective results.

# Importing the required libraries

import pandas as pd
from sklearn.svm import SVC
from matplotlib import pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
from sklearn.metrics import fbeta_score

np.random.seed(42)

# # Generating synthetic data

# Now, let's generate a synthetic dataset comprising two input columns. We'll create a dataset that encapsulates information about individuals' credit scores and annual income. Typically, credit scores range between 350 to 850, however for our analysis, we'll narrow it down to 500 to 800, as this range encompasses the majority of credit scores. Regarding annual income, we'll consider the range to be 20,000 to 180,000.

df = pd.DataFrame({'credit_score': np.random.randint(500,800,1000),
                   'annual_income': np.random.randint(20000,180000,1000)})
df

# For the output column, we'll establish that individuals with an annual income exceeding 100,000 and a credit score above 650 will be approved for the credit card.

df['credit_card_approved'] = ((df['annual_income'] > 100000) 
                              &
                              (df['credit_score'] > 650 )).astype(int)

# Lets visualize this generated dataset

fig = plt.figure()
ax = fig.add_subplot()

colors = np.array(["red", "blue"])

labels = ['Not Approved', 'Approved']

for i, label in enumerate(labels):
    ax.scatter(df.loc[df['credit_card_approved'] == i, 'annual_income'], 
               df.loc[df['credit_card_approved'] == i, 'credit_score'], 
               c=colors[i], label=label)
    
ax.set_title('Credit Card Approvals')
ax.set_xlabel('Credit Score')
ax.set_ylabel('Annual Income')

ax.legend(loc='lower right')

plt.tight_layout()
plt.show()

# Now, let's split the data into testing and training sets.

X = df[['annual_income', 'credit_score']]
y = df['credit_card_approved']

X_train_, X_test_, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Lets normalise the data so that they are on the same scale.

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train_)
X_test = scaler.fit_transform(X_test_)

# # Modeling the Data

# Because false positives and false negatives are equally important in this scenario, let us consider accuracy as our metric

accuracy_scorer = make_scorer(accuracy_score)
performance = pd.DataFrame({"model": [], "Accuracy": [], "Precision": [], "Recall": [], "F1": [], "F2": [], "Parameters": []}) # creating a dataframe to store our results

# We will be using three versions of SVM - linear,rbf,poly

# We will use grid search to systematically explore a range of hyperparameters and select the optimal combination for our model, we'll use this for all our models to ensure better performance.

# ## SVM classifier using linear kernal

# Linear SVM, a variant of Support Vector Machines (SVM), employs a hyperplane as the decision boundary to separate classes in the original feature space. It excels when data is linearly separable, offering computational efficiency particularly useful for high-dimensional datasets. However, its simplicity limits its effectiveness in handling non-linear relationships between features.

param_grid = {'C': [0.01, 0.1, 0.5, 1, 5, 10, 50],  
              'kernel': ['linear']}

grid = GridSearchCV(SVC(), param_grid, scoring=accuracy_scorer, refit = True, verbose = 3, n_jobs=-1) 
  
_ = grid.fit(X_train, y_train)

print(grid.best_params_) 
  
print(grid.best_estimator_)

y_pred = grid.predict(X_test) 

recall = recall_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
f2 = fbeta_score(y_test, y_pred, beta=2)
print('SVM Linear F1 Score =',f1)
performance = pd.concat([performance, pd.DataFrame({"model": ["SVM Linear"], "Accuracy": [accuracy], "Precision": [precision], "Recall": [recall], "F1": [f1], "F2": [f2], "Parameters": [grid.best_params_]})])


# ## SVM classifier using rbf kernal

# RBF kernel SVM, leveraging Gaussian radial basis functions, offers superior flexibility in capturing intricate decision boundaries, making it suitable for highly non-linear datasets. By mapping data into an infinite-dimensional space, it accommodates complex relationships between features, making it robust across a wide range of scenarios. However, its flexibility comes with computational cost, and selecting appropriate hyperparameters is crucial for optimal performance.

param_grid = {'C': [0.1, 1, 10, 100],  
              'gamma': [1, 0.1, 0.01, 0.001, 0.0001], 
              'kernel': ['rbf']}
  
grid = GridSearchCV(SVC(), param_grid, scoring=accuracy_scorer, refit = True, verbose = 3, n_jobs=-1) 

_ = grid.fit(X_train, y_train)

print(grid.best_params_) 
  
print(grid.best_estimator_)

y_pred = grid.predict(X_test) 

recall = recall_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
f2 = fbeta_score(y_test, y_pred, beta=2)
print('SVM rbf F1 Score =',f1)
performance = pd.concat([performance, pd.DataFrame({"model": ["SVM rbf"], "Accuracy": [accuracy], "Precision": [precision], "Recall": [recall], "F1": [f1], "F2": [f2],"Parameters": [grid.best_params_]})])

# ## SVM classifier using polynomial kernal

# Polynomial kernel SVM extends SVM's capabilities by mapping data into a higher-dimensional space using polynomial functions, allowing for more complex decision boundaries. This variant suits situations where data exhibits moderate non-linearity, permitting the adjustment of decision boundary complexity through the choice of polynomial degree. However, it may struggle with very high-dimensional data or excessively complex relationships.

param_grid = {'C': [0.01, 0.1, 0.5, 1, 5, 10, 50],  
              'coef0': [0.01, 0.1, 0.5, 1, 5, 10, 50],
              'kernel': ['poly']}
  
grid = GridSearchCV(SVC(), param_grid, scoring=accuracy_scorer, refit = True, verbose = 3, n_jobs=-1) 
  
_ = grid.fit(X_train, y_train)

print(grid.best_params_) 
print(grid.best_estimator_)

y_pred = grid.predict(X_test) 

recall = recall_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
f2 = fbeta_score(y_test, y_pred, beta=2)
print('SVM poly F1 Score =',f1)
performance = pd.concat([performance, pd.DataFrame({"model": ["SVM Poly"], "Accuracy": [accuracy], "Precision": [precision], "Recall": [recall], "F1": [f1], "F2": [f2], "Parameters": [grid.best_params_]})])

# # Results and Discussion

performance.sort_values(by="Accuracy", ascending=False)

# The results demonstrate the performance of each SVM variant on the classification task. Firstly, the SVM with RBF kernel achieved the highest accuracy of 96.00% and outperformed the other models in terms of precision, recall, F1 score, and F2 score. This indicates its effectiveness in capturing complex relationships in the data, as evidenced by its higher recall and precision compared to the polynomial and linear SVM. The choice of hyperparameters significantly influences the performance of each SVM model. In the case of SVM with RBF kernel, the 'C' parameter controls the trade-off between a smooth decision boundary and correctly classifying training points. A higher 'C' value indicates a more stringent penalty for misclassification, which can lead to overfitting if not carefully tuned. In this instance, a 'C' value of 100 was selected, indicating a relatively high tolerance for misclassification. Additionally, the 'gamma' parameter determines the influence of each training example; a low 'gamma' value implies a far-reaching effect, potentially leading to oversmoothing of the decision boundary, while a high 'gamma' value can result in overfitting. Here, a 'gamma' value of 1 was chosen, balancing the need for generalization with capturing intricate patterns in the data.
# 
# 
# The SVM with polynomial kernel also performed well, with an accuracy of 95.67%. While its precision and recall were slightly lower than the RBF kernel SVM, it still managed to achieve competitive results. However, the polynomial kernel SVM tends to be sensitive to the choice of hyperparameters, as indicated by the influence of the 'C' and 'coef0' parameters on its performance. The 'C' parameter governs the trade-off between margin width and training error. A higher 'C' value imposes a stronger penalty for misclassification, potentially leading to a narrower margin and overfitting. In this case, a 'C' value of 50 was selected. Additionally, the 'coef0' parameter determines the influence of higher-degree polynomial terms relative to lower-degree terms. A higher 'coef0' value emphasizes the importance of higher-degree terms, potentially making the model more sensitive to fluctuations in the data. Here, a 'coef0' value of 50 was chosen, indicating a moderate emphasis on higher-degree terms.
# 
# On the other hand, the linear SVM exhibited lower accuracy and performance metrics compared to the kernelized SVMs. While it achieved a respectable accuracy of 91.67%, its precision, recall, and F1 scores were notably lower. This suggests that the linear SVM struggled to capture the non-linear relationships present in the data, resulting in lower precision and recall, particularly for more complex patterns. The lower performance of the linear SVM underscores the likelihood that the data was not linearly separable, thus limiting the efficacy of a linear decision boundary in this context. The 'C' parameter controls the regularization strength, determining the trade-off between maximizing the margin and minimizing the training error. A higher 'C' value allows for a smaller margin and potentially higher training error, while a lower 'C' value encourages a wider margin and lower training error. Here, a 'C' value of 10 was selected, indicating a relatively low tolerance for misclassification.


```
**Feedback:**
Data Generation (30%) - 0.4/0.45
• Creativity and relevance of the chosen scenario - 0.15/0.15
• Correct implementation of data generation techniques - 0.15/0.15
• Clarity and structure of the data (including labeling and documentation) - 0.1/0.15

Model Implementation and Tuning (40%) - 0.6/0.6
• Correct implementation of SVC/SVR models with the three specified kernels - 0.2/0.2
• Effective use of hyper-parameter tuning, with a clear rationale for chosen parameters - 0.2/0.2
• Documentation of model performance metrics, with comparisons to justify the best-performing model - 0.2/0.2

Analysis and Discussion (30%)- 0.45/0.45
• Clarity and depth of the introduction and explanation of the chosen problem and dataset - 0.225/0.225
• Critical analysis of model results, including insights into performance variations across different kernels - 0.225/0.225

Need to create a dedicated notebook for data generation as this keeps your code more organized. So, 0.05 marks were deducted

---

# Student 38
```python
# #####  I am importing the required libraries for data manipulation, data visualization, & machine learning with scikit-learn for support vector machine (SVM) 
# ##### I am considering a random seed so tha we can get consistent results even the randomness is included 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler

np.random.seed(1)

# ###### Here i am generating a sample data with 25 rows, It contains two columns one is "BMI", and other one is 'fasting blood sugar'. 

sample_size = 30
minimum_BMI = 18
maximum_BMI = 35
minimum_fasting_blood_sugar = 80
maximum_fasting_blood_sugar = 180

df = pd.DataFrame({'BMI': np.linspace(minimum_BMI,maximum_BMI,sample_size).astype(float),'fasting blood sugar': np.linspace(minimum_fasting_blood_sugar,maximum_fasting_blood_sugar,sample_size).astype(int)})
df

df['diabetes risk'] = ((df['BMI'] > 28) |  (df['fasting blood sugar'] > 100)).astype(int)
df

# ##### now i am involving some random noise to the fasting blood sugar column in the data frame, this will make sure that the data shows some randomness, so that  the practicality in dataset is enhanced.

df['fasting blood sugar'] = df['fasting blood sugar'] + np.random.uniform(-30,30, sample_size)

# ##### scatter plot between BMI and fasting blood sugar, where the color-coded points on label diabetic risk'.

plt.figure(figsize=(6, 5))
plt.scatter(df['BMI'], df['fasting blood sugar'], c=df['diabetes risk'], cmap='cool', edgecolors='k', alpha=0.8, s=100)
plt.scatter(df['BMI'], df['fasting blood sugar'], c='green', marker='o', label='Data Points')
plt.colorbar(label='diabetes risk')
plt.title('Scatter Plot of BMI vs fasting blood sugar')
plt.xlabel('BMI')
plt.ylabel('fasting blood sugar')
plt.legend()
plt.grid(True)
plt.show()

# ##### now we are saving the above created data into a csf file named 'diabetes_risk.csv'. Here we did not include the index.

df.to_csv('diabetes_risk.csv', index=False)

# ### Train and Split Data

# Use sklearn to split df into a training set and a test set

X = df.iloc[:,:2]
y = df['diabetes risk']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)


# ### Modeling the Data

performance = pd.DataFrame({"model": [], "Accuracy": [], "Precision": [], "Recall": [], "F1": [], "F2": [], "Parameters": []})

# generate a fbeta 2 scorer
from sklearn.metrics import make_scorer
from sklearn.metrics import fbeta_score
f2_scorer = make_scorer(fbeta_score, beta=2)

# ### 3.1 Fit a SVM classification model using linear kernal

# defining parameter range 
param_grid = {'C': [0.01, 0.1, 0.5, 1, 5, 10, 50, 100],  
              'kernel': ['linear']}
  

grid = GridSearchCV(SVC(), param_grid, scoring=f2_scorer, refit = True, verbose = 3, n_jobs=-1) 
  
# fitting the model for grid search 
_ = grid.fit(X_train, y_train)

# ##### The SVM model is tuned and then the test datais forecasted. Also, calculating the performance metrics for evaluation

# print best parameter after tuning 
print(grid.best_params_) 
  
# print how our model looks after hyper-parameter tuning 
print(grid.best_estimator_)

y_pred = grid.predict(X_test) 

recall = recall_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
f2 = fbeta_score(y_test, y_pred, beta=2)

performance = pd.concat([performance, pd.DataFrame({"model": ["SVM Linear"], "Accuracy": [accuracy], "Precision": [precision], "Recall": [recall], "F1": [f1], "F2": [f2], "Parameters": [grid.best_params_]})])


# ### 3.2 Fit a SVM classification model using rbf kernal

# defining parameter range 
param_grid = {'C': [0.1, 1, 10, 100],  
              'gamma': [1, 0.1, 0.01, 0.001, 0.0001], 
              'kernel': ['rbf']}
  
grid = GridSearchCV(SVC(), param_grid, scoring=f2_scorer, refit = True, verbose = 3, n_jobs=-1) 
  
_ = grid.fit(X_train, y_train)

# ##### Printing the after-tuning with the goal of seeing the optimal parameters. After adjusting the hyperparameters, assess and record the SVM's performance metrics

# print best parameter after tuning 
print(grid.best_params_) 
  
# print how our model looks after hyper-parameter tuning 
print(grid.best_estimator_)

y_pred = grid.predict(X_test) 

recall = recall_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
f2 = fbeta_score(y_test, y_pred, beta=2)

performance = pd.concat([performance, pd.DataFrame({"model": ["SVM rbf"], "Accuracy": [accuracy], "Precision": [precision], "Recall": [recall], "F1": [f1], "F2": [f2],"Parameters": [grid.best_params_]})])

# ### 3.3 Fit a SVM classification model using polynomial kernal

# defining parameter range 
param_grid = {'C': [0.01, 0.1, 0.5, 1, 5, 10, 50, 100],  
              'coef0': [0.01, 0.1, 0.5, 1, 5, 10, 50, 100],
              'kernel': ['poly']}
  
grid = GridSearchCV(SVC(), param_grid, scoring=f2_scorer, refit = True, verbose = 3, n_jobs=-1) 
  
# fitting the model for grid search 
_ = grid.fit(X_train, y_train)

# #### printing the best paramters, and tuned the model, evaluating the performance metrics and then added results to dataframe.

# print best parameter after tuning 
print(grid.best_params_) 
  
# print how our model looks after hyper-parameter tuning 
print(grid.best_estimator_)

y_pred = grid.predict(X_test) 

recall = recall_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
f2 = fbeta_score(y_test, y_pred, beta=2)

performance = pd.concat([performance, pd.DataFrame({"model": ["SVM Poly"], "Accuracy": [accuracy], "Precision": [precision], "Recall": [recall], "F1": [f1], "F2": [f2], "Parameters": [grid.best_params_]})])

# ## 4.0 Summary

# #### Here the 'performance' dataframe is been sorted by the F2 column in descending order

performance.sort_values(by="F2", ascending=False)

# #####  The analysis of three SVM models on a dataset reveals perfect scores in Accuracy, Precision, Recall, F1, and F2 metrics for each, indicating 100% classification effectiveness. These results are demonstrating  exceptional model performance but necessitate caution in interpretation regarding their applicability to unseen data.




```
**Feedback:**
Data Generation (30%) - 0.4/0.45
• Creativity and relevance of the chosen scenario - 0.15/0.15
• Correct implementation of data generation techniques - 0.15/0.15
• Clarity and structure of the data (including labeling and documentation) - 0.1/0.15

Model Implementation and Tuning (40%) - 0.6/0.6
• Correct implementation of SVC/SVR models with the three specified kernels - 0.2/0.2
• Effective use of hyper-parameter tuning, with a clear rationale for chosen parameters - 0.2/0.2
• Documentation of model performance metrics, with comparisons to justify the best-performing model - 0.2/0.2

Analysis and Discussion (30%)- 0.275/0.45
• Clarity and depth of the introduction and explanation of the chosen problem and dataset - 0.05/0.225
• Critical analysis of model results, including insights into performance variations across different kernels - 0.225/0.225

Detailed explanation about the dataset(input and output variables) is necessary and need to represent whether you are choosing regression or classification task and dedicated notebook for data generation should be provided as this keeps your code more organized. So, 0.275 marks were deducted.

---

# Student 39
```python
# # WE03-SVM

# ## Objective 

# ### Importing the libraries

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error

# ### Data Generation

np.random.seed(1)
# Generating synthetic data for energy consumption prediction
num_samples = 1000

# Features: Number of occupants and Number of rooms in the house
X = np.random.randint(1, 6, size=(num_samples, 2))  # Random number of occupants (1-5) and Number of rooms

print(X)

# Output: Energy consumption (randomly generated based on occupants and Number of rooms)
y = 50 * X[:, 0] + 100 * X[:, 1] + np.random.normal(0, 50, size=num_samples)

print(y)

# ### Creating DataFrame 

df = pd.DataFrame(data=np.column_stack([X, y]), columns=['No.of.Occupants', 'No.of.Rooms', 'Energy_Consumption'])

df

# ### Splitting data into training and testing sets

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ### SVR 

# #### Linear model

# SVR with Linear Kernel

# Define the model
svr_linear = SVR(kernel='linear')

# Define hyperparameters for tuning
params_linear = {'C': [0.1, 1, 10]}

# Perform hyperparameter tuning
svr_linear_cv = GridSearchCV(svr_linear, params_linear, cv=5)
svr_linear_cv.fit(X_train, y_train)

# Best parameters
best_params_linear = svr_linear_cv.best_params_

# Model evaluation
y_pred_linear = svr_linear_cv.predict(X_test)
mse_linear = mean_squared_error(y_test, y_pred_linear)
r2_linear = svr_linear_cv.score(X_test, y_test)

print("SVR with Linear Kernel - Best Parameters:", best_params_linear)
print("SVR with Linear Kernel - Mean Squared Error:", mse_linear)
print("SVR with Linear Kernel - R-squared Score:", r2_linear)


# #### RBF model

# SVR with RBF Kernel

# Define the model
svr_rbf = SVR(kernel='rbf')

# Define hyperparameters for tuning
params_rbf = {'C': [0.1, 1, 10], 'gamma': ['scale', 'auto']}

# Perform hyperparameter tuning
svr_rbf_cv = GridSearchCV(svr_rbf, params_rbf, cv=5)
svr_rbf_cv.fit(X_train, y_train)

# Best parameters
best_params_rbf = svr_rbf_cv.best_params_

# Model evaluation
y_pred_rbf = svr_rbf_cv.predict(X_test)
mse_rbf = mean_squared_error(y_test, y_pred_rbf)
r2_rbf = svr_rbf_cv.score(X_test, y_test)

print("SVR with RBF Kernel - Best Parameters:", best_params_rbf)
print("SVR with RBF Kernel - Mean Squared Error:", mse_rbf)
print("SVR with RBF Kernel - R-squared Score:", r2_rbf)


# #### Poly model

# SVR with Polynomial Kernel

# Define the model
svr_poly = SVR(kernel='poly')

# Define hyperparameters for tuning
params_poly = {'C': [0.1, 1, 10], 'degree': [2, 3, 4]}

# Perform hyperparameter tuning
svr_poly_cv = GridSearchCV(svr_poly, params_poly, cv=5)
svr_poly_cv.fit(X_train, y_train)

# Best parameters
best_params_poly = svr_poly_cv.best_params_

# Model evaluation
y_pred_poly = svr_poly_cv.predict(X_test)
mse_poly = mean_squared_error(y_test, y_pred_poly)
r2_poly = svr_poly_cv.score(X_test, y_test)

print("SVR with Polynomial Kernel - Best Parameters:", best_params_poly)
print("SVR with Polynomial Kernel - Mean Squared Error:", mse_poly)
print("SVR with Polynomial Kernel - R-squared Score:", r2_poly)


# ### Discussion


```
**Feedback:**
Data Generation (30%) - 0.4/0.45
• Creativity and relevance of the chosen scenario - 0.15/0.15
• Correct implementation of data generation techniques - 0.15/0.15
• Clarity and structure of the data (including labeling and documentation) - 0.1/0.15

Model Implementation and Tuning (40%) - 0.6/0.6
• Correct implementation of SVC/SVR models with the three specified kernels - 0.2/0.2
• Effective use of hyper-parameter tuning, with a clear rationale for chosen parameters - 0.2/0.2
• Documentation of model performance metrics, with comparisons to justify the best-performing model - 0.2/0.2

Analysis and Discussion (30%)- 0.35/0.45
• Clarity and depth of the introduction and explanation of the chosen problem and dataset - 0.125/0.225
• Critical analysis of model results, including insights into performance variations across different kernels - 0.225/0.225

Need to create a dedicated notebook for data generation as this keeps your code more organized and should mention whether you choose classification or regression task in introduction or data generation notebook. So, 0.15 marks were deducted.

---

# Student 40
```python
# Assignment : Week03- SVM

# This document outlines the procedure for creating a synthetic dataset. The dataset is designed to simulate the prediction of manufacturing situations where a variety of factors influence the end product's quality. we can investigate and evaluate the correlations between the input factors production time and defects and the outcome variable product quality. This exercise aims to showcase the application of binary classification models. The Python programming language, along with libraries such as pandas, numpy, and matplotlib, will be utilized for dataset generation, introducing randomness to reflect real-world data and saving the dataset for future analysis.

# Our goal is to comprehend how various model forms might represent the underlying patterns found in the data from synthetic manufacturing. This procedure will reveal which Support Vector Machine (SVM) model is most suited for forecasting product quality depending on defects and production time. We'll use metrics like Accuracy, Precision, Recall, F1 Score, and F2 Score to measure their performance, giving extra importance to Recall.

# Step 1: Importing all the necessary Libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVC, SVR
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, f1_score
from sklearn.metrics import make_scorer
from sklearn.metrics import fbeta_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler

# Step 2: Set Random Seed
# 
# Setting a random seed ensures the reproducibility of our results.

np.random.seed(1) # set this to ensure the results are repeatable.

# Step 3: Generate Sample Data
# 
# We define our sample size and generate random distributions for the features: applicant_age and monthly_income

# Define sample size
sample_size = 100

#setting up a age and income limit to open a bank account
min_age = 18 
max_age = 65
income_min = 10000 
income_max = 80000

# Create DataFrame
df = pd.DataFrame({'Applicant_age': np.linspace(min_age,max_age,sample_size).astype(int),'monthly_income': np.linspace(income_min,income_max,sample_size).astype(int)})
df.head(10)

# The DataFrame has two columns: 'Applicant_age' and 'monthly_income'. The 'Applicant_age' column contains 99 evenly spaced values between 18 and 65, and the 'income' column contains 99 evenly spaced values between 10000 and 80000. The DataFrame has 99 rows, with each row representing a unique combination of Applicant_age and monthly_income. The 'Applicant_age' and 'monthly_income' columns are of type integer.

# Generating Account opening based on simplistic assumptions
df['Account_opens'] = ((df['Applicant_age'] > 27 ) |  (df['monthly_income'] > 42000)).astype(int)
df

# Step 4: Save the DataFrame to a CSV file
# 
# we can save our dataset to a CSV file for further analysis or to serve as input for machine learning models.

# Save to CSV
df.to_csv('C:/Users/sanga/Downloads/Bankaccount.csv', index=False)

# Step 5: Dependent and Independent Variables

# Load the dataset
df = pd.read_csv('C:/Users/sanga/Downloads/Bankaccount.csv')

# Separate features and target variable
X = df[['Applicant_age', 'monthly_income']]
y = df['Account_opens']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=1)

# The code snippet provided encompasses loading the dataset, separating features and the target variable, splitting the data into training and test sets

# Our data will be divided into two sets: one set will be used to train the model, and the other will be used to evaluate its output. To set aside 40% of the data for testing, we use test_size=0.4. In order to maintain consistency and reproducibility in our results, we additionally set random_state=1 to guarantee that the data split stays the same during each code run.

# Step 6:Model the Data
# 
# Initialize Performance Tracking

performance = pd.DataFrame({"model": [], "Accuracy": [], "Precision": [], "Recall": [], "F1": [], "F2": [], "Parameters": []})
#This code initializes a DataFrame named performance to store performance metrics for different models.

# create a fbeta 2 scorer
f2_scorer = make_scorer(fbeta_score, beta=2)

# This code snippet shows how to write a custom scorer for the F-beta 2 score, a valuable metric for recall-focused model evaluation in binary classification. The scorer is created using the make_scorer function from sklearn.metrics, with the beta value set to 2.

# Step 7: Genrating a SVM classification
#     
# a. Fit a SVM classification model using linear kernal

# defining parameter range 
param_grid = {'C': [0.01, 0.1, 0.5, 1, 5, 10, 50, 100],  
              'kernel': ['linear']}
  
#grid = GridSearchCV(SVC(), param_grid, scoring='f1', refit = True, verbose = 3, n_jobs=-1) 

grid = GridSearchCV(SVC(), param_grid, scoring=f2_scorer, refit = True, verbose = 3, n_jobs=-1) 
  
# fitting the model for grid search 
_ = grid.fit(X_train, y_train)

# In this code snippet, a grid search is performed to find the optimal hyperparameters for a Support Vector Machine (SVM) model. The parameter grid consists of different values for the regularization parameter C and the kernel function kernel, with 'linear' specified as the kernel. The f2_scorer scoring function is utilized for evaluation during the grid search, aiming to optimize for a specific F2 score. The model is trained using the training data (X_train and y_train) with parallel processing (n_jobs=-1). The best parameters are selected and used to refit the model (refit=True). The process is executed with verbosity level set to 3 for detailed output during the search.

# print best parameter after tuning 
print(grid.best_params_) 
  
# print how our model looks after hyper-parameter tuning 
print(grid.best_estimator_)

y_pred = grid.predict(X_test) 

# Calculate metrics
recall = recall_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
f2 = fbeta_score(y_test, y_pred, beta=2)

# Update performance dataframe
performance = pd.concat([performance, pd.DataFrame({"model": ["SVM Linear"], "Accuracy": [accuracy], "Precision": [precision], "Recall": [recall], "F1": [f1], "F2": [f2], "Parameters": [grid.best_params_]})])


# in above code snippet the optimized parameters and the revised model after adjusting the hyperparameters. Next, we use the test data to generate predictions, and we compute a number of performance metrics including recall, accuracy, precision, F1 and F2 scores. We may assess the performance of our Support Vector Machine (SVM) model with the use of these measures. Ultimately, we refresh a datframe with the performance measurements so that it may be compared to alternative models.

# b. Fit a SVM classification model using rbf kernal

# defining parameter range 
param_grid = {'C': [0.1, 1, 10, 100],  
              'gamma': [1, 0.1, 0.01, 0.001, 0.0001], 
              'kernel': ['rbf']}
  
grid = GridSearchCV(SVC(), param_grid, scoring=f2_scorer, refit = True, verbose = 3, n_jobs=-1) 
  
# fitting the model for grid search 
_ = grid.fit(X_train, y_train)

# We are trying with various settings for our Support Vector Machine model to determine which performs well. We are experimenting with different sets of 'C' and 'gamma', employing the 'rbf' kernel and the 'f2_scorer' scoring technique. Then, for maximum efficiency, we will train the optimal model on all CPU cores that are available.

# print best parameter after tuning 
print(grid.best_params_) 
  
# print how our model looks after hyper-parameter tuning 
print(grid.best_estimator_)

y_pred = grid.predict(X_test) 

# Calculate metrics
recall = recall_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
f2 = fbeta_score(y_test, y_pred, beta=2)

# Update performance dataframe
performance = pd.concat([performance, pd.DataFrame({"model": ["SVM rbf"], "Accuracy": [accuracy], "Precision": [precision], "Recall": [recall], "F1": [f1], "F2": [f2],"Parameters": [grid.best_params_]})])

# The model's new configuration and the optimal parameter are printed when the hyperparameters have been adjusted. Next, we derive metrics such as accuracy, precision, recall, F1, and F2 scores by predicting test data. Our SVM model's performance with the RBF kernel is evaluated by these metrics. The optimal parameters and performance measures of the model are then updated in a dataframe.

# c. Fit a SVM classification model using polynomial kernal

# defining parameter range 
param_grid = {'C': [0.01, 0.1, 0.5, 1, 5, 10, 50, 100],  
              'coef0': [0.01, 0.1, 0.5, 1, 5, 10, 50, 100],
              'kernel': ['poly']}
  
grid = GridSearchCV(SVC(), param_grid, scoring=f2_scorer, refit = True, verbose = 3, n_jobs=-1) 
  
# fitting the model for grid search 
_ = grid.fit(X_train, y_train)

# To determine which support vector machine (SVM) model parameters are optimal, we employ a grid search. We are using a polynomial kernel to test the parameters "C" and "coef0," which have a fixed range of values. The GridSearchCV approach assesses each model's performance using a unique scoring formula, assisting us in methodically determining the ideal set of parameters. By following this procedure, we may be sure that the SVM model we choose has the best parameters for our situation.

# print best parameter after tuning 
print(grid.best_params_) 
  
# print how our model looks after hyper-parameter tuning 
print(grid.best_estimator_)

y_pred = grid.predict(X_test) 

recall = recall_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
f2 = fbeta_score(y_test, y_pred, beta=2)

performance = pd.concat([performance, pd.DataFrame({"model": ["SVM Poly"], "Accuracy": [accuracy], "Precision": [precision], "Recall": [recall], "F1": [f1], "F2": [f2], "Parameters": [grid.best_params_]})])

# We find the best model settings, test the model with new data, and measure how well it performs using various metrics like accuracy, precision, and recall. Then, we update a table with these results and the best settings.

# Step 8: Sorting Performance Data by F2 Column

performance.sort_values(by="F2", ascending=False)

# This code snippet tells the computer to arrange the rows of the performance data based on the values in the "F2" column, with the highest values appearing first. It's like organizing a list from biggest to smallest according to a specific criterion.

# Analysis
# 
# 
# In order to forecast whether or not someone would open a bank account, we examined various models. Our attention was drawn to a score known as F2, which tells us how well the algorithms identify instances in which users actually open an account.
# 

# Strong performance is shown by all three models, which have high F1 scores, recall, accuracy, and precision.
# 
# SVM Polynomial and SVM Linear perform flawlessly on the provided dataset, earning perfect scores (100%) across all measures.
# 
# SVM RBF has a slightly lower precision and F1 score than the other two models, but it still performs well, with 90% accuracy. This implies that it may encounter difficulties accurately categorizing specific cases, resulting in a reduced level of precision.

# Recommendations:
# 
# Either SVM Linear or SVM Polynomial could be taken into consideration, depending on the particular objectives and specifications, as both exhibit flawless performance on the given metrics.
# 
# SVM Linear may be better if interpretability and processing economy are crucial.
# 
# In situations when non-linearity is important and SVM Linear is unable to adequately represent the data, SVM Polynomial may be a better option.
# 
# Even though SVM RBF performs a little bit worse, it could still be a decent choice, particularly if processing power is an issue.


```
**Feedback:**
Data Generation (30%) - 0.4/0.45
• Creativity and relevance of the chosen scenario - 0.15/0.15
• Correct implementation of data generation techniques - 0.15/0.15
• Clarity and structure of the data (including labeling and documentation) - 0.1/0.15

Model Implementation and Tuning (40%) - 0.6/0.6
• Correct implementation of SVC/SVR models with the three specified kernels - 0.2/0.2
• Effective use of hyper-parameter tuning, with a clear rationale for chosen parameters - 0.2/0.2
• Documentation of model performance metrics, with comparisons to justify the best-performing model - 0.2/0.2

Analysis and Discussion (30%)- 0.45/0.45
• Clarity and depth of the introduction and explanation of the chosen problem and dataset - 0.225/0.225
• Critical analysis of model results, including insights into performance variations across different kernels - 0.225/0.225

Need to create a dedicated notebook for data generation as this keeps your code more organized. So, 0.05 marks were deducted

---

# Student 41
```python
# 
# ## Assignment - 3

# # Data Generation

# ### 1. Import libraries that we expect to use

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification

# ### 2. Generate synthetic data with two input features (income and age) and one output feature (owned)

# Function to generate composite numbers within a range
def generate_composite_numbers(start, end):
    composite_numbers = []
    for num in range(start, end + 1):
        if num > 1:
            for i in range(2, num):
                if (num % i) == 0:
                    composite_numbers.append(num)
                    break
    return composite_numbers

# Generating synthetic data with composite numbers
def generate_data(n_samples):
    composite_income = generate_composite_numbers(10, 100)  # Composite numbers between 10 and 100
    composite_age = generate_composite_numbers(20, 80)  # Composite numbers between 20 and 80
    
    X = np.zeros((n_samples, 2))
    X[:, 0] = np.random.choice(composite_income, size=n_samples)
    X[:, 1] = np.random.choice(composite_age, size=n_samples)
    
    y = np.random.randint(2, size=n_samples)  # Random ownership status
    
    return X, y

X, y = generate_data(100)

# ### 3. Visualise the data


# Visualizing the synthetic data
plt.figure(figsize=(10, 6))
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='bwr', marker='o')
plt.xlabel('Income')
plt.ylabel('Age')
plt.title('Synthetic Data - Predicting Ownership')
plt.colorbar(label='Owned (1) / Not Owned (0)')
plt.grid(True)
plt.show()

# ### 4. Create a dataframe

# Transfer data to DataFrame
data = pd.DataFrame(X, columns=['income', 'age'])
data['owned'] = y
data

# ### 5. Save the data to a CSV file

data.to_csv('ownership_data.csv', index=False)

# 
# ## Assignment - 3

# ## SVC Model Generation:

# ### 1. Importing important libraries

import pandas as pd
from sklearn.svm import SVC
from matplotlib import pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV

np.random.seed(1)

# ### 2. Load the data

df = pd.read_csv('ownership_data.csv') # let's use the same data as we did in the logistic regression example
df.head(3)

# Use sklearn to split df into a training set and a test set

X = df[['income', 'age']]
y = df['owned']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)


# ### 3. Model the data

performance = pd.DataFrame({"model": [], "Accuracy": [], "Precision": [], "Recall": [], "F1": [], "F2": [], "Parameters": []})

# create a fbeta 2 scorer
from sklearn.metrics import make_scorer
from sklearn.metrics import fbeta_score
f2_scorer = make_scorer(fbeta_score, beta=2)

# ### 3.1 Fit a SVM classification model using linear kernal

# defining parameter range 
param_grid = {'C': [0.01, 0.1, 0.5, 1, 5, 10, 50, 100],  
              'kernel': ['linear']}
  
#grid = GridSearchCV(SVC(), param_grid, scoring='f1', refit = True, verbose = 3, n_jobs=-1) 

grid = GridSearchCV(SVC(), param_grid, cv=5,scoring=f2_scorer, refit = True, verbose = 3, n_jobs=-1) 
  
# fitting the model for grid search 
_ = grid.fit(X_train, y_train)

# print best parameter after tuning 
print(grid.best_params_) 
  
# print how our model looks after hyper-parameter tuning 
print(grid.best_estimator_)

y_pred = grid.predict(X_test) 

recall = recall_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
f2 = fbeta_score(y_test, y_pred, beta=2)

performance = pd.concat([performance, pd.DataFrame({"model": ["SVM Linear"], "Accuracy": [accuracy], "Precision": [precision], "Recall": [recall], "F1": [f1], "F2": [f2], "Parameters": [grid.best_params_]})])


performance

# ### 3.2 Fit a SVM classification model using rbf kernal

# defining parameter range 
param_grid = {'C': [0.1, 1, 10, 100],  
              'gamma': [1, 0.1, 0.01, 0.001, 0.0001], 
              'kernel': ['rbf']}
  
grid = GridSearchCV(SVC(), param_grid, scoring=f2_scorer, refit = True, verbose = 3, n_jobs=-1) 
  
# fitting the model for grid search 
_ = grid.fit(X_train, y_train)

# print best parameter after tuning 
print(grid.best_params_) 
  
# print how our model looks after hyper-parameter tuning 
print(grid.best_estimator_)

y_pred = grid.predict(X_test) 

recall = recall_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
f2 = fbeta_score(y_test, y_pred, beta=2)

performance = pd.concat([performance, pd.DataFrame({"model": ["SVM rbf"], "Accuracy": [accuracy], "Precision": [precision], "Recall": [recall], "F1": [f1], "F2": [f2],"Parameters": [grid.best_params_]})])

# ### 3.3 Fit a SVM classification model using polynomial kernal

# defining parameter range 
param_grid = {'C': [0.01, 0.1, 0.5, 1, 5, 10, 50, 100],  
              'coef0': [0.01, 0.1, 0.5, 1, 5, 10, 50, 100],
              'kernel': ['poly']}
  
grid = GridSearchCV(SVC(), param_grid, scoring=f2_scorer, refit = True, verbose = 3, n_jobs=-1) 
  
# fitting the model for grid search 
_ = grid.fit(X_train, y_train)

# print best parameter after tuning 
print(grid.best_params_) 
  
# print how our model looks after hyper-parameter tuning 
print(grid.best_estimator_)

y_pred = grid.predict(X_test) 

recall = recall_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
f2 = fbeta_score(y_test, y_pred, beta=2)

performance = pd.concat([performance, pd.DataFrame({"model": ["SVM Poly"], "Accuracy": [accuracy], "Precision": [precision], "Recall": [recall], "F1": [f1], "F2": [f2], "Parameters": [grid.best_params_]})])

performance.sort_values(by="F2", ascending=False)

# ### Discussion

# - The Linear SVM model shows consistent and high performance across various metrics, including accuracy (56.7%), precision (56.7%), recall (100%), F1 score (72.3%), and AUC-ROC (86.7%).
# - The model performs well, achieving perfect recall, indicating that it correctly identifies all positive instances. The regularization strength (C = 0.01) suggests a relatively low penalty for misclassification.

# - The RBF SVM model displays identical performance metrics to the Linear SVM, with accuracy, precision, recall, F1 score, and AUC-ROC all at 56.7%.
# - Despite sharing metrics with the Linear SVM, the RBF kernel might capture more complex patterns in the data, as it considers non-linear relationships through the radial basis function.

# - Similar to Linear and RBF SVM, the Polynomial SVM also shows consistent performance with all metrics at 56.7%.
# - The polynomial kernel introduces non-linearity, yet the model performance is on par with the linear and RBF kernels in this case. The hyperparameters (C, coef0) may need further exploration for potential improvement.

# ### Summary

# - The RBF SVM model is the most promising among the three, demonstrating a balanced performance across various metrics.
# - The Poly SVM model shows mixed results and may benefit from further hyperparameter tuning.
# - The Linear SVM model performs poorly, suggesting it might not be suitable for the given dataset.
# - Hyperparameter details provide insights for potential model refinement, and additional experimentation could enhance overall predictive capabilities.


```
**Feedback:**
Data Generation (30%) - 0.45/0.45
• Creativity and relevance of the chosen scenario - 0.15/0.15
• Correct implementation of data generation techniques - 0.15/0.15
• Clarity and structure of the data (including labeling and documentation) - 0.15/0.15

Model Implementation and Tuning (40%) - 0.6/0.6
• Correct implementation of SVC/SVR models with the three specified kernels - 0.2/0.2
• Effective use of hyper-parameter tuning, with a clear rationale for chosen parameters - 0.2/0.2
• Documentation of model performance metrics, with comparisons to justify the best-performing model - 0.2/0.2

Analysis and Discussion (30%)- 0.225/0.45
• Clarity and depth of the introduction and explanation of the chosen problem and dataset - 0/0.225
• Critical analysis of model results, including insights into performance variations across different kernels - 0.225/0.225

No introduction section provided. So, 0.225 marks were deducted

---

# Student 42
```python
# # WE03

# In this Notebook we are generating a synthetic dataset which has two features and one target where we look at the data and do preprocessing if possible and save it to a csv file and this csv file will be further used in another notebook where we  use the data set for fitting it to the  Support vector Classification(SVC) by implementing the Hyper Parameter tuning.
# 
# We are trying to predict if there will be crop yield or not by considering the Temperature and Humidity as the factors that are influencing the crop yield which is a case of classification.

# # Importing the required libraries that we will be using in this note book.

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from imblearn.under_sampling import RandomUnderSampler
np.random.seed(1)
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, f1_score
from sklearn.metrics import fbeta_score,make_scorer
from sklearn.metrics import confusion_matrix


# ### Step 1: Generation of data set and storing it into data frame df 

df = pd.DataFrame({'TEMPERATURES_DEVIATION': np.random.randint(low=0,high=30,size=1000)})

# I have generated the synthetic data of Temperature deviations where the minimum temperature deviation is 0 and the Maximum temperatire deviation of 30  by using the np.random.randint to generate random integers  and storing them in the column TEMPERATURES_DEVIATION of the data frame.

df['SOIL_MOISTURE'] = pd.DataFrame({'SOIL_MOISTURE': np.random.randint(low=0,high=100,size=1000).round().astype(int)})

# I have generated the synthetic data of Soil Moisture Percentages  where the minimum Soil Moisture Percentages  is 0 and the Maximum temperatire deviation of 100 by using the np.random.randint to generate random integers and storing them in the column SOIL_MOISTURE of the data frame.

# Now we are looking at the top 5 observations of the dataframe which contain the dat aof Soil Moisture and the temperture difference 

df.head(5)

# Here we are creating a new column CROP_YIELD based on the condition that the Values in the data frame of the column TEMPERATURES_DEVIATION is lesser than 15 and also the SOIL_MOISTURE percentage should be greater than 50 then we are considering it as yield and considering it as 1 if the condition fails we are considering as no yield and assigning 0 which is a boolean condition

df['CROP_YIELD'] = ((df['TEMPERATURES_DEVIATION'] <15) & (df['SOIL_MOISTURE'] > 50)).astype(int)


# Now we are looking at the top 5 observations of the dataframe which contain the data of Soil Moisture and the temperture difference along with the CROP_YIELD information as well.

# ### Step 2: Preprocessing the Data

df.head()

# We are looking at the number of rows and columns in the data frame 

df.shape

# Checking if there are any null values in the data frame we have generated and additing them to get more clear understanding on data if there are any nulls in the data frame or not

df.isna().sum()

# We are looking at the statstics of the data in the dataframe, from the below we can see the minimum values, maximum values and the mean and standard deviadtion in each column of the data frame 

df.describe()

# Here we are checking if the generated dataset is balanced or not where Value_cpunts() will give the unique values in the CROP_YIELD column of the dataframe. we are only concerened about the CROP_YIELD as it it will be Target Column.

df['CROP_YIELD'].value_counts()

# We can see that 730 observations are 0 which is no yield and 270 which is yield.

# ### Step 3: Splitting the data into Train set and the Test set

# We are assigning the features to X dataframe by dropping the column CROP_YIELD and assigning the target CROP_YIELD to Y 

X=df.drop('CROP_YIELD',axis=1)
y=df['CROP_YIELD']


# The 'train_test_split'  is to divide the dataset into training and testing sets. It separates the features ('X_train' and 'X_test') and the target variable ('y_train' and 'y_test') with a 20% test size, i have considered the 80% of training set as the total number of observations are less

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=1)


# ### Step 4: Balancing the data 

# Here we are balancing the dataset and we are using the RandomUnderSampler for attaining the goal, The random undersampler is used as the distribution of the classes are skewed as the data is looking biased,the randomunders sampler randomly deletes the  the majority class in the train set data

rus = RandomUnderSampler()
X_resampled, y_resampled = rus.fit_resample(X_train, y_train)



# Here we are Creating a data frame named performance to store the metrices that are required to evaluate the performance and efficieny of all the Models we are going to fit. We are using the fbeta_score to calculate the f2 score, where f2 score is calculated when we want to pioritize the recall rather than the precision as the minority class is quite smaller than the majority class.

performance = pd.DataFrame({"model": [], "Accuracy": [], "Precision": [], "Recall": [], "F1": [], "F2": [], "Parameters": []})
f2_scorer = make_scorer(fbeta_score, beta=2)

# Fitting the models

# The Support Vector Machine(SVM) is more effective both in linear and non linear decision boundaries, where the svm always looks for a hyperplane that seperates the data points into different classes where the data points that are close to the hyperplane are the support vectors and the line that separates is the hyper plane.We do have three Kernels in the SVM, these kernels will help the SVM to handle the non linear classification without having to explicitly perform the data transformation which can be expensive computationally.

# We are implementing the Hyper parameter tuning where we find the optimal hyperparameters as we know that the paramters can impact the performace of the model. We are using the Grid search where we define the set of parameters and the we check the performance of the model by trying these combinations.
# 

# ### Step 5: Fitting the models

# #### Implementing Support Vector Machine with Linear as Kernel

# We are consider the below C values, where C is the factor that controls the regularization strength.

# defining parameter range 
param_grid = {'C': [0.001,0.01, 0.1, 0.5, 1, 5, 10, 50, 100],  
              'kernel': ['linear']}
  
#grid = GridSearchCV(SVC(), param_grid, scoring='f1', refit = True, verbose = 3, n_jobs=-1) 

grid = GridSearchCV(SVC(), param_grid, scoring=f2_scorer, refit = True, verbose = 3, n_jobs=-1) 
  
# fitting the model for grid search 
_ = grid.fit(X_resampled, y_resampled)

# The total fits are 45 as we can see that 5 folds are being performed by testing the 9 different combinations.

# print best parameter after tuning 
print(grid.best_params_) 
  
# print how our model looks after hyper-parameter tuning 
print(grid.best_estimator_)

y_pred = grid.predict(X_test) 

recall = recall_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
f2 = fbeta_score(y_test, y_pred, beta=2)


performance = pd.concat([performance, pd.DataFrame({"model": ["SVM Linear"], "Accuracy": [accuracy], "Precision": [precision], "Recall": [recall], "F1": [f1], "F2": [f2], "Parameters": [grid.best_params_]})])


# Here we are capturng the best C value which is 0.1 for the kernel linear 

# Generating The Confusion Matrix

CM_L= confusion_matrix(y_test,y_pred)
CM_L

# ####  Implementing Support Vector Machine with rbf as Kernel

# The radial basis function which uses the Gaussian radial basis function to transform the data points into high dimensional space without computing the transformed feature and the gama value controls the widdth of the Gaussian function where when it is small the decision boundaries are much more smoother and when it is large the detailing of the data will be more. 

# defining parameter range 
param_grid = {'C': [0.1, 1, 10, 100],  
              'gamma': [1, 0.1, 0.01, 0.001, 0.0001], 
              'kernel': ['rbf']}
  
grid = GridSearchCV(SVC(), param_grid, scoring=f2_scorer, refit = True, verbose = 3, n_jobs=-1) 
  
# fitting the model for grid search 
_ = grid.fit(X_resampled, y_resampled)

# The total fits are 100 as we can see that 5 folds are being performed by testing the 20 different combinations.

# print best parameter after tuning 
print(grid.best_params_) 
  
# print how our model looks after hyper-parameter tuning 
print(grid.best_estimator_)

y_pred = grid.predict(X_test) 

recall = recall_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
f2 = fbeta_score(y_test, y_pred, beta=2)


performance = pd.concat([performance, pd.DataFrame({"model": ["SVM rbf"], "Accuracy": [accuracy], "Precision": [precision], "Recall": [recall], "F1": [f1], "F2": [f2],"Parameters": [grid.best_params_]})])


# Here we are capturing the best C value which is 100 and the Gama value is 0.01 for the kernel rbf

# Generating The Confusion Matrix

CM_RBF= confusion_matrix(y_test,y_pred)
CM_RBF

# ###  Implementing Support Vector Machine with Poly as Kernel

# The Coef0 will effect the flexibilty of the model so we are trying down a range of values

# defining parameter range 
param_grid = {'C': [0.01, 0.1, 0.5, 1, 5, 10, 50, 100],  
              'coef0': [0.01, 0.1, 0.5, 1, 5, 10, 50, 100],
              'kernel': ['poly']}
  
grid = GridSearchCV(SVC(), param_grid, scoring=f2_scorer, refit = True, verbose = 3, n_jobs=-1) 
  
# fitting the model for grid search 
_ = grid.fit(X_resampled, y_resampled)

# The total fits are 320 as we can see that 5 folds are being performed by testing the 64 different combinations.

# print best parameter after tuning 
print(grid.best_params_) 
  
# print how our model looks after hyper-parameter tuning 
print(grid.best_estimator_)

y_pred = grid.predict(X_test) 

recall = recall_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
f2 = fbeta_score(y_test, y_pred, beta=2)



performance = pd.concat([performance, pd.DataFrame({"model": ["SVM Poly"], "Accuracy": [accuracy], "Precision": [precision], "Recall": [recall], "F1": [f1], "F2": [f2], "Parameters": [grid.best_params_]})])

# Here we are capturing the best C value which is 100 and the Coef value is 5 for the kernel Poly 

# Generating The Confusion Matrix

CM_P= confusion_matrix(y_test,y_pred)
CM_P

# We are printing the metrices of all the SVM models we have implemented 

performance

# # Summary

# I have Genereated  a synthetic dataset with two inputs and one output where my inputs describes the  TEMPERATURES_DEVIATION
#  and the SOIL_MOISTURE and the output is the CROP_YIELD which is dependednt on the input variables and done the preprocessing of the data and splitted the data by train test split method.The dataset looks imbalanced so i have used the RandomUndersampler technique to balnce the data set and fitted the  Support Vector Machine model by using hyper parameter tuning to test different combinations of parameters by using Grid search abnd got the below results.
#  #### ACCURACY
# - The accuracy of the SVM model with Kernel as Linear is 0.895
# - The accuracy of the SVM model with Kernel as rbf is 0.985
# - The accuracy of the SVM model with Kernel as poly is 0.990
# 
#  #### Precision
# - The Precision of the SVM model with Kernel as Linear is 0.826087
# - The Precision of the SVM model with Kernel as rbf is 0.956522	
# - The Precision of the SVM model with Kernel as poly is 0.970588
#  
#  #### Recall
# - The Recall of the SVM model with Kernel as Linear is 0.863636
# - The Recall of the SVM model with Kernel as rbf is 1.00
# - The Recall of the SVM model with Kernel as poly is 1.00
#  
#  #### F1 Score
# - The F1 Score of the SVM model with Kernel as Linear is 0.844444	
# - The F1 Score of the SVM model with Kernel as rbf is 0.977778
# - The F1 Score of the SVM model with Kernel as poly is 0.993976
#  
#  #### F2 Score
# - The F2 Score of the SVM model with Kernel as Linear is 0.855856	
# - The F2 Score of the SVM model with Kernel as rbf is 0.990991
# - The F2 Score of the SVM model with Kernel as poly is 	0.993976
# 
# 
# From the Performance dataframe we can observe that the C value is high for both the SVM kernel- rbf and the kernel- Poly
#  where C is the regularization parameter that determines the width of the margin and the no of violations. 
#  In our case C is valued as 100 for both Kernels RBF and Poly that means the margin will be small  and the no of violations will be less which will be sign of overfitting and we can see that the boost in the accuracy from 0.895 when the kernel is Linear  to 0.985(Kernel- rbf) and 0.990(Kernel-Poly) and the recall is also equivalent to 1 for these both kernels(Poly,rbf) which can also be a sign of overfitting.However we should also consider the fact that the dataset is quite small.The SVM Linear model is the best model for predicting the target Crop_yield as the C value is minimum which is 0.1 which means the more regularization has been performed and the margin will be quite wider hyperplane seperating the classses.The decision boundary is also smoother which may generalize and prevents over fitting of the model.When we look at the confusion matrix of the SVM linear model the True Positives are 122 which means the model have predicted that there will be crop yield when there is actually crop yield and the True negatives are also 57 which means there is no crop yield then the model have predicted it truly and classified as Neative.We can cosider that the SVM Linear Kernel is better model for predicting the Crop Yield.




```
**Feedback:**
Data Generation (30%) - 0.4/0.45
• Creativity and relevance of the chosen scenario - 0.15/0.15
• Correct implementation of data generation techniques - 0.15/0.15
• Clarity and structure of the data (including labeling and documentation) - 0.1/0.15

Model Implementation and Tuning (40%) - 0.6/0.6
• Correct implementation of SVC/SVR models with the three specified kernels - 0.2/0.2
• Effective use of hyper-parameter tuning, with a clear rationale for chosen parameters - 0.2/0.2
• Documentation of model performance metrics, with comparisons to justify the best-performing model - 0.2/0.2

Analysis and Discussion (30%)- 0.45/0.45
• Clarity and depth of the introduction and explanation of the chosen problem and dataset - 0.225/0.225
• Critical analysis of model results, including insights into performance variations across different kernels - 0.225/0.225

Need to create a dedicated notebook for data generation as this keeps your code more organized. So, 0.05 marks were deducted

---

# Student 43
```python
# # Week03 - SVM models - Linear, RBF, Polynomial
# 
# In this notebook, we have a synthetic dataset representing loan approval predictions based on applicants' credit scores and annual incomes. The dataset consists of 1000 samples, with two input features: Credit Score and Annual Income, and a binary output variable indicating Loan Approval (0 for rejection and 1 for approval). The loan approval prediction function uses the applicant's credit score and annual income as input.If both values meet predefined thresholds, the function predicts approval; otherwise, it predicts rejection. We'll explore how the model makes predictions by analyzing the relationship between credit scores, annual incomes, and loan approval status through visualization and discussion. 

# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, f1_score

# ## Sample Size and Range Parameters definition

# Set input conditions
credit_score_threshold = 600  # Example threshold for credit score
annual_income_threshold = 50000  # Example threshold for annual income


# ## Generating Target Variable by applying conditions on input columns

# Define function to predict loan approval based on input conditions
def predict_loan_approval(credit_score, annual_income):
    if credit_score >= credit_score_threshold and annual_income >= annual_income_threshold:
        return 1  # Loan Approved
    else:
        return 0  # Loan Rejected

# Generate synthetic classification data
np.random.seed(1)
X, y = make_classification(n_samples=1000, n_features=2, n_classes=2, n_clusters_per_class=2, 
                           n_redundant=0, n_informative=2, class_sep=0.8, flip_y=0.1)


# Create a DataFrame
data = pd.DataFrame({'Credit Score': X[:, 0], 'Annual Income': X[:, 1], 'Loan Approval': y})

data['Loan Approval'].value_counts()

# ## Visualize the data

# Visualize the data
plt.figure(figsize=(8, 6))
plt.scatter(data[data['Loan Approval'] == 0]['Credit Score'], data[data['Loan Approval'] == 0]['Annual Income'], color='blue', label='Rejected', alpha=0.5)
plt.scatter(data[data['Loan Approval'] == 1]['Credit Score'], data[data['Loan Approval'] == 1]['Annual Income'], color='red', label='Approved', alpha=0.5)
plt.xlabel('Credit Score')
plt.ylabel('Annual Income')
plt.title('Synthetic Classification Data: Loan Approval (More Variability)')
plt.legend()
plt.show()


data.head(10)

# Save the dataset to a CSV file
data.to_csv('synthetic_loan_approval_data.csv', index=False)

# Load the dataset
data = pd.read_csv('synthetic_loan_approval_data.csv')

# Split the data into features (X) and target variable (y)
X = data[['Credit Score', 'Annual Income']]
y = data['Loan Approval']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# ## Model the data
# 
# First, let's create a dataframe to load the model performance metrics into.

performance = pd.DataFrame({"model": [], "Accuracy": [], "Precision": [], "Recall": [], "F1": [], "F2": [], "Parameters": []})

# create a fbeta 2 scorer
from sklearn.metrics import make_scorer
from sklearn.metrics import fbeta_score
f2_scorer = make_scorer(fbeta_score, beta=2)


# ## Fit a SVM classification model using linear kernal

# defining parameter range 
param_grid = {'C': [0.01, 0.1, 0.5, 1, 5, 10, 50, 100],  
              'kernel': ['linear']}
  
#grid = GridSearchCV(SVC(), param_grid, scoring='f1', refit = True, verbose = 3, n_jobs=-1) 

grid = GridSearchCV(SVC(), param_grid, scoring=f2_scorer, refit = True, verbose = 3, n_jobs=-1) 
  
# fitting the model for grid search 
_ = grid.fit(X_train, y_train)

# print best parameter after tuning 
print(grid.best_params_) 
  
# print how our model looks after hyper-parameter tuning 
print(grid.best_estimator_)

y_pred = grid.predict(X_test) 

recall = recall_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
f2 = fbeta_score(y_test, y_pred, beta=2)

performance = pd.concat([performance, pd.DataFrame({"model": ["SVM Linear"], "Accuracy": [accuracy], "Precision": [precision], "Recall": [recall], "F1": [f1], "F2": [f2], "Parameters": [grid.best_params_]})])


performance

# Calculate confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Create a heatmap for the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Spectral_r', xticklabels=['Predicted Negative', 'Predicted Positive'], yticklabels=['Actual Negative', 'Actual Positive'])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()

# ## SVC with Linear Kernel result analysis:
# 
# ### Introduction:
# 
# In this analysis, we explore the performance of the Support Vector Classifier (SVC) with a linear kernel for loan approval prediction. The dataset consists of synthetic data representing various features related to loan applicants, such as credit score and annual income to predict whether a loan application will be approved or not.
# 
# ### Evaluation Metrics:
# 
# We evaluate the model using the following metrics:
# 
# Accuracy: Measures the proportion of correctly classified instances out of all instances.
# 
# Precision: Measures the proportion of true positive predictions out of all positive predictions.
# 
# Recall: Measures the proportion of true positive predictions out of all actual positive instances.
# 
# F1 Score: The harmonic mean of precision and recall, providing a balanced measure between the two.
# 
# ### Results:
# 
# SVC with Linear Kernel:
# 
# Accuracy: 78%
# 
# Precision: 77%
# 
# Recall: 78.5%
# 
# F1 Score: 77.7%
# 
# F2 Score: 78.25%
# 
# Parameters: {'C': 0.01, 'kernel': 'linear'}
# 
# ### Discussion:
# 
# The SVC with a linear kernel assumes that the data is linearly separable in the input space. It operates effectively when the underlying data can be distinctly separated by a linear boundary. In this case, the decision boundary is linear, signifying that it separates the data into classes using a straight line.
# 
# Analyzing the performance metrics:
# 
# Accuracy: The model achieves an accuracy of 78%, indicating that approximately 78% of the instances are correctly classified.
# 
# Precision: With a precision of 77%, the model correctly identifies around 77% of the positive predictions out of all positive predictions made.
# 
# Recall: The recall score of 78.57% implies that the model captures roughly 78.57% of the actual positive instances.
# 
# F1 Score: The F1 score, which considers both precision and recall, is computed as 77.78%. This harmonic mean provides a balanced measure of the model's performance.
# 
# Parameters: The model is trained with a regularization parameter (C) set to 0.01 and a linear kernel.
# 
# Overall, the SVC with a linear kernel demonstrates moderate performance in predicting loan approvals based on the provided features. While the model achieves reasonable accuracy, precision, recall, and F1-score, further exploration with different kernels or fine-tuning of parameters may be necessary to improve its performance.

# ## Fit a SVM classification model using rbf kernal

# defining parameter range 
param_grid = {'C': [0.1, 1, 10, 100],  
              'gamma': [1, 0.1, 0.01, 0.001, 0.0001], 
              'kernel': ['rbf']}
  
grid = GridSearchCV(SVC(), param_grid, scoring=f2_scorer, refit = True, verbose = 3, n_jobs=-1) 
  
# fitting the model for grid search 
_ = grid.fit(X_train, y_train)

# print best parameter after tuning 
print(grid.best_params_) 
  
# print how our model looks after hyper-parameter tuning 
print(grid.best_estimator_)

y_pred = grid.predict(X_test) 

recall = recall_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
f2 = fbeta_score(y_test, y_pred, beta=2)

performance = pd.concat([performance, pd.DataFrame({"model": ["SVM rbf"], "Accuracy": [accuracy], "Precision": [precision], "Recall": [recall], "F1": [f1], "F2": [f2],"Parameters": [grid.best_params_]})])

performance

# Calculate confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Create a heatmap for the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Reds', xticklabels=['Predicted Negative', 'Predicted Positive'], yticklabels=['Actual Negative', 'Actual Positive'])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()

# ## SVC with RBF Kernel result analysis:
# 
# ### Introduction:
# 
# In this analysis, we explore the performance of the Support Vector Classifier (SVC) with a radial basis function (RBF) kernel for loan approval prediction. The dataset consists of synthetic data representing various features related to loan applicants, such as credit score and annual income, to predict whether a loan application will be approved or not.
# 
# The RBF kernel is chosen for its ability to capture non-linear relationships in the data. Unlike the linear kernel, which assumes linear separability in the input space, the RBF kernel computes similarity between data points in a high-dimensional space, allowing for more complex patterns to be captured.
# 
# 
# ### Evaluation Metrics:
# 
# We evaluate the model based on the following metrics:
# 
# Accuracy: Measures the proportion of correctly classified instances out of all instances.
# 
# Precision: Measures the proportion of true positive predictions out of all positive predictions.
# 
# Recall: Measures the proportion of true positive predictions out of all actual positive instances.
# 
# F1 Score: The harmonic mean of precision and recall, providing a balanced measure between the two.
# 
# ### Results - SVC with RBF Kernel:
# 
# Accuracy: 77.5%
# 
# Precision: 76.77%
# 
# Recall: 77.55%
# 
# F1 Score: 77.16%
# 
# F2 Score: 77.39%
# 
# Parameters: {'C': 100, 'gamma': 0.01, 'kernel': 'rbf'}
# 
# ### Discussion:
# 
# The SVC with a radial basis function (RBF) kernel differs from the linear kernel by its ability to capture non-linear relationships in the data. It computes similarity between data points in a high-dimensional space, allowing for more complex patterns to be captured.
# 
# Analyzing the performance metrics:
# 
# Accuracy: The model achieves an accuracy of 77.5%, indicating that approximately 77.5% of the instances are correctly classified.
# 
# Precision: With a precision of 76.77%, the model correctly identifies around 76.77% of the positive predictions out of all positive predictions made.
# 
# Recall: The recall score of 77.55% implies that the model captures roughly 77.55% of the actual positive instances.
# 
# F1 Score: The F1 score, which considers both precision and recall, is computed as 77.16%. This harmonic mean provides a balanced measure of the model's performance.
# 
# F2 Score: The F2 score, which places more emphasis on recall, is slightly higher at 77.39%.
# 
# Overall, the SVC with an RBF kernel demonstrates moderate performance in predicting loan approvals. It leverages the non-linear nature of the RBF kernel to capture more complex relationships in the data, resulting in comparable performance to the linear kernel. Further experimentation with different parameter settings or kernel types may be necessary to improve performance further.

# ## Fit a SVM classification model using polynomial kernal

# defining parameter range 
param_grid = {'C': [0.01, 0.1, 0.5, 1, 5, 10, 50, 100],  
              'coef0': [0.01, 0.1, 0.5, 1, 5, 10, 50, 100],
              'kernel': ['poly']}
  
grid = GridSearchCV(SVC(), param_grid, scoring=f2_scorer, refit = True, verbose = 3, n_jobs=-1) 
  
# fitting the model for grid search 
_ = grid.fit(X_train, y_train)

# print best parameter after tuning 
print(grid.best_params_) 
  
# print how our model looks after hyper-parameter tuning 
print(grid.best_estimator_)

y_pred = grid.predict(X_test) 

recall = recall_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
f2 = fbeta_score(y_test, y_pred, beta=2)

performance = pd.concat([performance, pd.DataFrame({"model": ["SVM Poly"], "Accuracy": [accuracy], "Precision": [precision], "Recall": [recall], "F1": [f1], "F2": [f2], "Parameters": [grid.best_params_]})])

performance

# Calculate confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Create a heatmap for the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges', xticklabels=['Predicted Negative', 'Predicted Positive'], yticklabels=['Actual Negative', 'Actual Positive'])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()

# ## SVC with Polynomial Kernel result analysis:
# 
# ### Introduction:
# 
# In this analysis, we assess the performance of the Support Vector Regressor (SVR) with a polynomial kernel for predicting housing prices. The dataset comprises synthetic data reflecting various housing attributes such as size and number of bedrooms.
# 
# The polynomial kernel is selected for its ability to capture non-linear relationships in the data. Unlike the linear kernel, which assumes linear separability in the input space, the polynomial kernel computes similarity between data points using polynomial functions, allowing for more complex patterns to be captured.
# 
# ### Evaluation Metrics:
# 
# We evaluate the model using the following metrics:
# 
# Accuracy: Measures the proportion of correctly classified instances out of all instances.
# Precision: Measures the proportion of true positive predictions out of all positive predictions.
# Recall: Measures the proportion of true positive predictions out of all actual positive instances.
# F1 Score: The harmonic mean of precision and recall, providing a balanced measure between the two.
# 
# ### Results: SVC with Polynomial Kernel:
# 
# Accuracy: 71.5%
# 
# Precision: 65.41%
# 
# Recall: 88.78%
# 
# F1 Score: 75.32%
# 
# F2 Score: 82.86%
# 
# Parameters: {'C': 0.5, 'coef0': 0.01, 'kernel': 'poly'}
# 
# ### Discussion:
# 
# The SVC with a polynomial kernel is adept at capturing non-linear relationships in the data, making it suitable for complex datasets like housing price prediction. By using polynomial functions to compute similarity between data points, the model can capture intricate patterns that may not be discernible with a linear kernel.
# 
# Analyzing the performance metrics:
# 
# Accuracy: The model achieves an accuracy of 71.5%, indicating that approximately 71.5% of the instances are correctly classified.
# 
# Precision: With a precision of 65.41%, the model correctly identifies around 65.41% of the positive predictions out of all positive predictions made.
# 
# Recall: The recall score of 88.78% implies that the model captures roughly 88.78% of the actual positive instances.
# 
# F1 Score: The F1 score, which considers both precision and recall, is computed as 75.32%. This harmonic mean provides a balanced measure of the model's performance.
# 
# F2 Score: The F2 score, which places more emphasis on recall, is slightly higher at 82.86%.
# 
# Overall, the SVC with a polynomial kernel demonstrates moderate performance in predicting housing prices. Leveraging the polynomial kernel's ability to capture non-linear relationships, the model achieves reasonable accuracy, precision, recall, and F1-score. Further fine-tuning of parameters or exploration of different kernel types may be necessary to enhance performance further.

# ## Summary
# 
# The provided SVM models, including Linear, RBF, and Polynomial kernels, were evaluated based on performance metrics and parameters.Linear and RBF SVMs show nearly identical results, suggesting that the data might be linearly separable or the RBF parameters had minimal impact.Polynomial SVM performed notably worse, indicating unsuitability for the dataset or suboptimal parameter selection.Precision was perfect for all models, but recall and F1-score were significantly lower for Polynomial SVM.

performance.sort_values(by="F2", ascending=False)


```
**Feedback:**
Data Generation (30%) - 0.4/0.45
• Creativity and relevance of the chosen scenario - 0.15/0.15
• Correct implementation of data generation techniques - 0.15/0.15
• Clarity and structure of the data (including labeling and documentation) - 0.1/0.15

Model Implementation and Tuning (40%) - 0.6/0.6
• Correct implementation of SVC/SVR models with the three specified kernels - 0.2/0.2
• Effective use of hyper-parameter tuning, with a clear rationale for chosen parameters - 0.2/0.2
• Documentation of model performance metrics, with comparisons to justify the best-performing model - 0.2/0.2

Analysis and Discussion (30%)- 0.35/0.45
• Clarity and depth of the introduction and explanation of the chosen problem and dataset - 0.125/0.225
• Critical analysis of model results, including insights into performance variations across different kernels - 0.225/0.225

Need to create a dedicated notebook for data generation as this keeps your code more organized and should mention whether you choose classification or regression task in introduction or data generation notebook. So, 0.15 marks were deducted.

---

# Student 44
```python
# 
# # Generating Synthetic Data
# ## Overview:
# The notebook generates synthetic credit card approval data with two features ('Income' and 'CreditScore') and a binary target variable ('Approval'). It simulates a scenario where approval depends on both income and credit score, visualizes the data, and exports it to a CSV file.

# #### Importing Modules

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#Library to generate synthetic classification data
from sklearn.datasets import make_classification

np.random.seed(42)

# #### Generating Synthetic Data 

# Generate synthetic classification data with two features and 600 Samples
X, y = make_classification(n_samples=600, n_features=2, n_informative=2, n_redundant=0,
                           n_clusters_per_class=1, flip_y=0.1, random_state=42)
# Add more realistic features
X[:, 0] = X[:, 0] * 30 + 50  # Scale and shift the first feature (Income)
X[:, 1] = X[:, 1] * 20 + 650  # Scale and shift the second feature (Credit Score)

# #### Creating a Dataframe

data = pd.DataFrame({'Income': X[:, 0], 'CreditScore': X[:, 1], 'Approval': y})
data.head()

# #### Creating Realistic condition for Credit card approval

# Realistic Scenario where approval depends on both income and credit score
data['Approval'] = np.where((data['Income'] > 70) & (data['CreditScore'] > 665), 1, 0)
data.head()

# #### Visualizing the Data

plt.figure(figsize=(8, 6))
plt.scatter(data[data['Approval'] == 0]['Income'], data[data['Approval'] == 0]['CreditScore'], label='Denied', marker='o')
plt.scatter(data[data['Approval'] == 1]['Income'], data[data['Approval'] == 1]['CreditScore'], label='Approved', marker='x')

plt.xlabel('Income')
plt.ylabel('Credit Score')
plt.title('Synthetic Data for Credit Card Approval')
plt.legend()
plt.show()

# #### Saving the Data

# Save the synthetic dataset to a CSV file
data.to_csv('credit_approval_data.csv', index=False)

# # SVM Demonstration with Hyperparameter tuning

# ## Overview:
# 
#    
# This notebook explores the performance of Support Vector Machine (SVM) models with various kernels (Linear, RBF, and Polynomial) on a synthetic credit approval dataset. The dataset comprises 'Income' and 'CreditScore' as input features and 'Approval' as the output label. The synthetic data simulates a scenario where credit approval hinges on income and credit score. Hyperparameter tuning is employed to optimize model parameters, enhancing their predictive capabilities. 

# ### 1. Importing Modules

import pandas as pd
from sklearn.svm import SVC
from matplotlib import pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV

np.random.seed(1)

# ### 2. Loading the Data and Performing Train-Test Split 

df = pd.read_csv('credit_approval_data.csv') 
df.head(5)

# Use sklearn to split df into a training set and a test set
features_columns = ['Income', 'CreditScore']

X = df[features_columns]
y = df['Approval']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

# ## 3. Model the Data

performance = pd.DataFrame({"model": [], "Accuracy": [], "Precision": [], "Recall": [], "F1": [], "Parameters": []})

# ### 3.1 Fit a SVM classification model using linear kernel with Hyper-parameter Tuning

# defining parameter range 
param_grid = {'C': [0.01, 0.1, 0.5, 1, 5, 10, 50, 100],  
              'kernel': ['linear']}
  
grid = GridSearchCV(SVC(), param_grid, scoring='f1', refit = True, verbose = 3, n_jobs=-1) 
  
# fitting the model for grid search 
_ = grid.fit(X_train, y_train)

# ### 3.2  Printing Best Parameters

# print best parameter after tuning 
print(grid.best_params_) 
  
# print how our model looks after hyper-parameter tuning 
print(grid.best_estimator_)

y_pred = grid.predict(X_test) 

recall = recall_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)


performance = pd.concat([performance, pd.DataFrame({"model": ["SVM Linear"], "Accuracy": [accuracy], "Precision": [precision], "Recall": [recall], "F1": [f1], "Parameters": [grid.best_params_]})])


performance

# ### 3.3 Fit a SVM classification model using rbf kernel with Hyper-parameter Tuning

# defining parameter range 
param_grid = {'C': [0.1, 1, 10, 100],  
              'gamma': [1, 0.1, 0.01, 0.001, 0.0001], 
              'kernel': ['rbf']}
  
grid = GridSearchCV(SVC(), param_grid, scoring='f1', refit = True, verbose = 3, n_jobs=-1) 
  
# fitting the model for grid search 
_ = grid.fit(X_train, y_train)

# ### 3.4 Printing Best Parameters

# print best parameter after tuning 
print(grid.best_params_) 
 
# print how our model looks after hyper-parameter tuning 
print(grid.best_estimator_)

y_pred = grid.predict(X_test) 

recall = recall_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

performance = pd.concat([performance, pd.DataFrame({"model": ["SVM rbf"], "Accuracy": [accuracy], "Precision": [precision], "Recall": [recall], "F1": [f1],"Parameters": [grid.best_params_]})])

# ### 3.5 Fit a SVM classification model using Poly kernel with Hyper-parameter Tuning

# defining parameter range 
param_grid = {'C': [0.01, 0.1, 0.5, 1, 5, 10, 50, 100],  
              'coef0': [0.01, 0.1, 0.5, 1, 5, 10, 50, 100],
              'kernel': ['poly']}
  
grid = GridSearchCV(SVC(), param_grid, scoring='f1', refit = True, verbose = 3, n_jobs=-1) 
  
# fitting the model for grid search 
_ = grid.fit(X_train, y_train)

# ### 3.6 Printing Best Parameters

# print best parameter after tuning 
print(grid.best_params_) 
 
# print how our model looks after hyper-parameter tuning 
print(grid.best_estimator_)

y_pred = grid.predict(X_test) 

recall = recall_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

performance = pd.concat([performance, pd.DataFrame({"model": ["SVM Poly"], "Accuracy": [accuracy], "Precision": [precision], "Recall": [recall], "F1": [f1],"Parameters": [grid.best_params_]})])

# ## 4. Summary

# The SVM models with different kernels demonstrated strong performance across various metrics. The SVM model with an RBF (Radial Basis Function) kernel emerged as the top-performing model, achieving an impressive accuracy of 98.89%. This model exhibited excellent precision (96.43%), recall (100%), and an F1 score of 98.18%. The optimal hyperparameters for this model were a regularization parameter (C) of 100 and a gamma value of 0.01.

performance.sort_values(by="F1", ascending=False)

# ## 5. Discussion and Summary

# **Discussion:**
# 
# From the obtained results, it is evident that Support Vector Machine (SVM) models with different kernels performed exceptionally well in predicting credit approval based on synthetic data. Here are the key observations:
# 
# 1. **RBF Kernel (SVM rbf):**
#    - Accuracy: 98.89%
#    - Precision: 96.43%
#    - Recall: 100%
#    - F1 Score: 98.18%
#    - Optimal Parameters: {'C': 100, 'gamma': 0.01}
#    
#    The RBF kernel model achieved the highest accuracy among the three, showcasing excellent balance between precision and recall. The optimal combination of hyperparameters indicates a well-tuned model.
# 
# 2. **Linear Kernel (SVM Linear):**
#    - Accuracy: 97.78%
#    - Precision: 94.64%
#    - Recall: 98.15%
#    - F1 Score: 96.36%
#    - Optimal Parameters: {'C': 0.5}
#    
#    The linear kernel model performed admirably, providing a good trade-off between precision and recall. It is notable for its simplicity and efficiency.
# 
# 3. **Polynomial Kernel (SVM Poly):**
#    - Accuracy: 97.22%
#    - Precision: 92.98%
#    - Recall: 98.15%
#    - F1 Score: 95.50%
#    - Optimal Parameters: {'C': 50, 'coef0': 10}
#    
#    While the polynomial kernel model exhibited strong performance, it slightly lagged behind the RBF and linear models in terms of accuracy and F1 score.
# 
# **Summary:**
# 
# In summary, all three SVM models demonstrated robust predictive capabilities for credit approval. The RBF kernel model stood out as the top performer, achieving the highest accuracy and F1 score. The linear kernel model offered an excellent balance between precision and recall, making it a practical choice. The polynomial kernel model, though slightly less accurate, still provided competitive results.
# 
# Fine-tuning hyperparameters significantly contributed to the overall success of these models in predicting credit approval.


```
**Feedback:**
Data Generation (30%) - 0.45/0.45
• Creativity and relevance of the chosen scenario - 0.15/0.15
• Correct implementation of data generation techniques - 0.15/0.15
• Clarity and structure of the data (including labeling and documentation) - 0.15/0.15

Model Implementation and Tuning (40%) - 0.6/0.6
• Correct implementation of SVC/SVR models with the three specified kernels - 0.2/0.2
• Effective use of hyper-parameter tuning, with a clear rationale for chosen parameters - 0.2/0.2
• Documentation of model performance metrics, with comparisons to justify the best-performing model - 0.2/0.2

Analysis and Discussion (30%)- 0.35/0.45
• Clarity and depth of the introduction and explanation of the chosen problem and dataset - 0.125/0.225
• Critical analysis of model results, including insights into performance variations across different kernels - 0.225/0.225

Need to mention whether you choose regression or classification task in introduction or data generation notebook. So, 0.1 marks were deducted.

---

# Student 45
```python
# ## Assignment- 3
# 
# The notebook presents an exploration of synthetic plant growth data generation, processing, visualization, and saving. Synthetic data generation is a crucial aspect of machine learning and data science research, allowing researchers to simulate datasets that mimic real-world scenarios. This notebook focuses on generating synthetic data for plant growth based on temperature and humidity factors. The generated data is then processed to transform the growth rate into binary values and visualized to understand the relationship between temperature, humidity, and growth rate. Finally, the synthetic dataset is saved to a CSV file for future use.

# This section imports necessary libraries such as numpy, pandas, sklearn.datasets, and matplotlib.pyplot to facilitate data generation, processing, visualization, and saving.

import numpy as np
import pandas as pd
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

np.random.seed(1)

# ### Generating Synthetic Data
# In this section, a function generate_plant_growth_data() is defined to create synthetic data for temperature, humidity, and growth rate. The function utilizes make_regression() from scikit-learn to generate synthetic features for temperature and humidity, and then computes the growth rate based on a simple formula with random noise.

def generate_plant_growth_data(n_samples=1000, random_state=42):
    # Generate synthetic data for temperature and humidity
    X, _ = make_regression(n_samples=n_samples, n_features=2, noise=0.2, random_state=random_state)
    # Transform the synthetic features to represent temperature and humidity
    temperature = 25 + 5 * X[:, 0]  # mean=25, std=5
    humidity = 50 + 10 * X[:, 1]     # mean=50, std=10
    
    # Generate synthetic data for growth rate using a simple formula
    growth_rate = 0.5 * temperature + 0.3 * humidity + np.random.normal(loc=0, scale=2, size=n_samples)
    
    return temperature, humidity, growth_rate

# Generate synthetic data
temperature, humidity, growth_rate = generate_plant_growth_data()

# ### Create a DataFrame
# This part converts the generated synthetic data into a pandas DataFrame with columns 'Temperature', 'Humidity', and 'GrowthRate'. It also displays the first few rows of the DataFrame to provide a glimpse of the data.

data = pd.DataFrame({'Temperature': temperature, 'Humidity': humidity, 'GrowthRate': growth_rate})
print(data.head())

# Calculate the mean of the 'GrowthRate' column
growth_rate_mean = data['GrowthRate'].mean()

# Convert 'GrowthRate' column to 1 if more than the mean, else 0
data['GrowthRate'] = data['GrowthRate'].apply(lambda x: 1 if x > growth_rate_mean else 0)

# ### Save the synthetic data to a CSV file
# In this section, the processed synthetic data is saved to a CSV file named 'synthetic_plant_growth_data.csv' for further analysis or usage in machine learning models.

data.to_csv('synthetic_plant_growth_data.csv', index=False)
print("Synthetic data saved to synthetic_plant_growth_data.csv")

# ## Assignment- 3
# 
# The notebook aims to explore the application of Support Vector Machines (SVMs) in classifying plant growth rates based on temperature and humidity data. SVMs are powerful supervised learning models capable of performing classification tasks by finding the hyperplane that best separates the classes in the feature space. The data used for this analysis has been preprocessed and cleaned, and it consists of synthetic plant growth data with temperature, humidity, and corresponding growth rate labels. The process involves splitting the data into training and test sets, fitting SVM classification models with different kernels (linear, rbf, and polynomial), tuning hyperparameters using grid search.

# ## 1. Setup

# This section imports necessary modules and sets a random seed.

import pandas as pd
from sklearn.svm import SVC
from matplotlib import pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV

np.random.seed(1)

# ## 2. Load data

# Data loading process involves reading the preprocessed synthetic plant growth data from a CSV file into a pandas DataFrame. The data includes features such as temperature and humidity, as well as the target variable, growth rate.

df = pd.read_csv('./synthetic_plant_growth_data.csv') # let's use the same data as we did in the logistic regression example
df.head(3)

# Use sklearn to split df into a training set and a test set

X = df[['Temperature', 'Humidity']]
y = df['GrowthRate']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)


# ## 3. Model the data

performance = pd.DataFrame({"model": [], "Accuracy": [], "Precision": [], "Recall": [], "F1": [], "F2": [], "Parameters": []})

# create a fbeta 2 scorer
from sklearn.metrics import make_scorer
from sklearn.metrics import fbeta_score
f2_scorer = make_scorer(fbeta_score, beta=2)


# ### 3.1 Metric
# 
# We choose F1 score as our main metric because the cost of both false positive and false negative is similar in my case.

# ### 3.2 Fit a SVM classification model using linear kernal

# defining parameter range 
param_grid = {'C': [0.01, 0.1, 0.5, 1, 5, 10, 50, 100],  
              'kernel': ['linear']}
  
grid = GridSearchCV(SVC(), param_grid, scoring='f1', refit = True, verbose = 3, n_jobs=-1) 
  
# fitting the model for grid search 
_ = grid.fit(X_train, y_train)

# print best parameter after tuning 
print(grid.best_params_) 
  
# print how our model looks after hyper-parameter tuning 
print(grid.best_estimator_)

y_pred = grid.predict(X_test) 

recall = recall_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
f2 = fbeta_score(y_test, y_pred, beta=2)

performance = pd.concat([performance, pd.DataFrame({"model": ["SVM Linear"], "Accuracy": [accuracy], "Precision": [precision], "Recall": [recall], "F1": [f1], "F2": [f2], "Parameters": [grid.best_params_]})])


# ### 3.3 Fit a SVM classification model using rbf kernal

# defining parameter range 
param_grid = {'C': [0.1, 1, 10, 100],  
              'gamma': [1, 0.1, 0.01, 0.001, 0.0001], 
              'kernel': ['rbf']}
  
grid = GridSearchCV(SVC(), param_grid, scoring='f1', refit = True, verbose = 3, n_jobs=-1) 
  
# fitting the model for grid search 
_ = grid.fit(X_train, y_train)

# print best parameter after tuning 
print(grid.best_params_) 
  
# print how our model looks after hyper-parameter tuning 
print(grid.best_estimator_)

y_pred = grid.predict(X_test) 

recall = recall_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
f2 = fbeta_score(y_test, y_pred, beta=2)

performance = pd.concat([performance, pd.DataFrame({"model": ["SVM rbf"], "Accuracy": [accuracy], "Precision": [precision], "Recall": [recall], "F1": [f1], "F2": [f2],"Parameters": [grid.best_params_]})])

# ### 3.4 Fit a SVM classification model using polynomial kernal

# defining parameter range 
param_grid = {'C': [0.01, 0.1, 0.5, 1, 5, 10, 50, 100],  
              'coef0': [0.01, 0.1, 0.5, 1, 5, 10, 50, 100],
              'kernel': ['poly']}
  
grid = GridSearchCV(SVC(), param_grid, scoring='f1', refit = True, verbose = 3, n_jobs=-1) 
  
# fitting the model for grid search 
_ = grid.fit(X_train, y_train)

# print best parameter after tuning 
print(grid.best_params_) 
  
# print how our model looks after hyper-parameter tuning 
print(grid.best_estimator_)

y_pred = grid.predict(X_test) 

recall = recall_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
f2 = fbeta_score(y_test, y_pred, beta=2)

performance = pd.concat([performance, pd.DataFrame({"model": ["SVM Poly"], "Accuracy": [accuracy], "Precision": [precision], "Recall": [recall], "F1": [f1], "F2": [f2], "Parameters": [grid.best_params_]})])

# ## 4.0 Summary

# In conclusion, the analysis demonstrates the effectiveness of SVMs in classifying plant growth rates using temperature and humidity features. 

performance.sort_values(by="F1", ascending=False)

# Based on the evaluation of SVM classification models using different kernels on the synthetic plant growth data, several key conclusions can be drawn:
# 
# 1. **Performance Metrics**: 
#    - The SVM models achieved overall good performance across various metrics including accuracy, precision, recall, F1 score, and F2 score.
#    - The SVM models with linear and polynomial kernels exhibited similar performance, achieving an accuracy of approximately 87.7% and F2 scores around 0.873.
#    - The SVM model with an rbf kernel performed slightly lower in terms of accuracy (86.3%) and F2 score (0.864), but still demonstrated acceptable performance.
# 
# 2. **Model Selection**:
#    - Both the SVM models with linear and polynomial kernels showed comparable results, indicating that they are suitable for classifying plant growth rates based on temperature and humidity features.
#    - The SVM model with an rbf kernel, while slightly less accurate, still provides a viable alternative for classification tasks.
# 
# 3. **Optimal Hyperparameters**:
#    - For the SVM model with a linear kernel, the best-performing model was achieved with a regularization parameter (C) of 0.01.
#    - For the SVM model with a polynomial kernel, the optimal C value was also 0.01, with a coefficient of 100.
#    - The SVM model with an rbf kernel demonstrated optimal performance with a C value of 10 and a gamma value of 0.001.
# 
# 4. **Practical Implications**:
#    - These findings provide valuable insights for practitioners in agriculture or related fields who seek to predict plant growth rates based on environmental factors.
#    - The choice of SVM kernel should be based on specific requirements such as interpretability, computational efficiency, and performance metrics of interest.
# 
# In summary, the evaluation of SVM classification models on synthetic plant growth data suggests that both linear and polynomial kernels offer robust performance for classifying plant growth rates. However, the choice between these kernels should consider factors such as interpretability and computational efficiency. Additionally, the rbf kernel presents a viable alternative, albeit with slightly lower performance in this specific context.

# ## 5.0 Analysis

# - The Linear kernel might have achieved a better F1 score compared to the Polynomial (Poly) and Radial Basis Function (RBF) kernels due to the nature of the underlying data and the characteristics of each kernel. 
# 
# - The Linear kernel assumes a linear decision boundary, making it well-suited for scenarios where the relationships between features and the target variable are more straightforward. 
# 
# - In cases where the true patterns in the data are predominantly linear, the Linear kernel can outperform more complex kernels, such as Poly and RBF, by avoiding overfitting to intricate structures that might not exist. 
# 
# - Additionally, with a lower regularization parameter (C=0.01), the Linear kernel may prioritize a simpler model, reducing the risk of fitting noise in the data. 
# 
# - The Linear kernel's success could be attributed to its ability to capture the inherent linearity in the dataset without introducing unnecessary complexity, resulting in a better balance between precision and recall and ultimately yielding a higher F1 score.


```
**Feedback:**
Data Generation (30%) - 0.45/0.45
• Creativity and relevance of the chosen scenario - 0.15/0.15
• Correct implementation of data generation techniques - 0.15/0.15
• Clarity and structure of the data (including labeling and documentation) - 0.15/0.15

Model Implementation and Tuning (40%) - 0.6/0.6
• Correct implementation of SVC/SVR models with the three specified kernels - 0.2/0.2
• Effective use of hyper-parameter tuning, with a clear rationale for chosen parameters - 0.2/0.2
• Documentation of model performance metrics, with comparisons to justify the best-performing model - 0.2/0.2

Analysis and Discussion (30%)- 0.45/0.45
• Clarity and depth of the introduction and explanation of the chosen problem and dataset - 0.225/0.225
• Critical analysis of model results, including insights into performance variations across different kernels - 0.225/0.225

---

# Student 46
```python
# ### Import necessary libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification

np.random.seed(1)

# ### Generating synthetic data

def generate_weather_data(n_samples):
    X, y = make_classification(
        n_samples=n_samples, 
        n_features=2, 
        n_informative=2, 
        n_redundant=0, 
        n_clusters_per_class=1, 
        random_state=42
    )
    # Convert to meaningful weather ranges
    X[:, 0] = 30 + 20 * X[:, 0]  # Temperature range: 30-50°C
    X[:, 1] = 40 + 30 * np.abs(X[:, 1])  # Humidity range: 40-70%
    
    # Introduce some randomness for rain occurrence
    y = np.random.randint(2, size=n_samples)
    
    return X, y

X, y = generate_weather_data(100)

# ### Discussion:
# 
# #### Temperature:
# Higher Temperatures: Warm temperatures may indicate clear skies and lower chances of rain in some climates. However, in others, they can lead to increased evaporation and atmospheric instability, potentially causing thunderstorms and rain.
# 
# Lower Temperatures: Cooler temperatures might signal stable weather with reduced rain likelihood in some regions. However, in others, they could indicate the presence of cold fronts and potential precipitation.
# 
# #### Humidity:
# Higher Humidity: High humidity levels suggest moisture-saturated air, often associated with increased rain chances, especially when combined with other atmospheric factors like low pressure systems.
# 
# Lower Humidity: Lower humidity indicates drier air, which may lead to reduced chances of rain. However, extremely low humidity could also indicate arid conditions with minimal precipitation likelihood.
# 
# In summary, while higher humidity typically increases rain probability, the relationship between humidity, temperature, and rainfall is complex and varies based on climate and atmospheric dynamics.

# ### Visualizing the synthetic data

plt.figure(figsize=(10, 6))
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='bwr', marker='o')
plt.xlabel('Temperature (°C)')
plt.ylabel('Humidity (%)')
plt.title('Synthetic Data - Predicting Rain')
plt.colorbar(label='Rain (1) / No Rain (0)')
plt.grid(True)
plt.show()

# ### Transfer data to DataFrame

data = pd.DataFrame(X, columns=['temperature', 'humidity'])
data['rain'] = y

data

# ### Save data to CSV

data.to_csv('weather_data.csv', index=False)

# ### Import necessary libraries

import pandas as pd
from sklearn.svm import SVC
from matplotlib import pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV

np.random.seed(1)

# ### Loading the data

df = pd.read_csv('weather_data.csv') # let's use the same data as we did in the logistic regression example
df.head(3)

# Use sklearn to split df into a training set and a test set

X = df[['temperature', 'humidity']]
y = df['rain']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)


# ### Model the data

performance = pd.DataFrame({"model": [], "Accuracy": [], "Precision": [], "Recall": [], "F1": [], "F2": [], "Parameters": []})

# create a fbeta 2 scorer
from sklearn.metrics import make_scorer
from sklearn.metrics import fbeta_score
f2_scorer = make_scorer(fbeta_score, beta=2)

# #### 1. Fit a SVM classification model using linear kernal

# defining parameter range 
param_grid = {'C': [0.01, 0.1, 0.5, 1, 5, 10, 50, 100],  
              'kernel': ['linear']}
  
#grid = GridSearchCV(SVC(), param_grid, scoring='f1', refit = True, verbose = 3, n_jobs=-1) 

grid = GridSearchCV(SVC(), param_grid, cv=5,scoring=f2_scorer, refit = True, verbose = 3, n_jobs=-1) 
  
# fitting the model for grid search 
_ = grid.fit(X_train, y_train)

# print best parameter after tuning 
print(grid.best_params_) 
  
# print how our model looks after hyper-parameter tuning 
print(grid.best_estimator_)

y_pred = grid.predict(X_test) 

recall = recall_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
f2 = fbeta_score(y_test, y_pred, beta=2)

performance = pd.concat([performance, pd.DataFrame({"model": ["SVM Linear"], "Accuracy": [accuracy], "Precision": [precision], "Recall": [recall], "F1": [f1], "F2": [f2], "Parameters": [grid.best_params_]})])


performance

# #### 2 Fit a SVM classification model using rbf kernal

# defining parameter range 
param_grid = {'C': [0.1, 1, 10, 100],  
              'gamma': [1, 0.1, 0.01, 0.001, 0.0001], 
              'kernel': ['rbf']}
  
grid = GridSearchCV(SVC(), param_grid, scoring=f2_scorer, refit = True, verbose = 3, n_jobs=-1) 
  
# fitting the model for grid search 
_ = grid.fit(X_train, y_train)

# print best parameter after tuning 
print(grid.best_params_) 
  
# print how our model looks after hyper-parameter tuning 
print(grid.best_estimator_)

y_pred = grid.predict(X_test) 

recall = recall_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
f2 = fbeta_score(y_test, y_pred, beta=2)

performance = pd.concat([performance, pd.DataFrame({"model": ["SVM rbf"], "Accuracy": [accuracy], "Precision": [precision], "Recall": [recall], "F1": [f1], "F2": [f2],"Parameters": [grid.best_params_]})])

# #### 3 Fit a SVM classification model using polynomial kernal

# defining parameter range 
param_grid = {'C': [0.01, 0.1, 0.5, 1, 5, 10, 50, 100],  
              'coef0': [0.01, 0.1, 0.5, 1, 5, 10, 50, 100],
              'kernel': ['poly']}
  
grid = GridSearchCV(SVC(), param_grid, scoring=f2_scorer, refit = True, verbose = 3, n_jobs=-1) 
  
# fitting the model for grid search 
_ = grid.fit(X_train, y_train)

# print best parameter after tuning 
print(grid.best_params_) 
  
# print how our model looks after hyper-parameter tuning 
print(grid.best_estimator_)

y_pred = grid.predict(X_test) 

recall = recall_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
f2 = fbeta_score(y_test, y_pred, beta=2)

performance = pd.concat([performance, pd.DataFrame({"model": ["SVM Poly"], "Accuracy": [accuracy], "Precision": [precision], "Recall": [recall], "F1": [f1], "F2": [f2], "Parameters": [grid.best_params_]})])

performance.sort_values(by="F2", ascending=False)

# ### Discussion:

# - The SVM Linear model achieved an accuracy of 53.33%, indicating that it correctly classified 53.33% of the instances in the dataset.
# 
# - Precision, which measures the proportion of correctly predicted positive cases among all predicted positive cases, is 53.33%. This indicates that when the model predicts rain, it is correct 53.33% of the time.
# 
# - Recall, also known as sensitivity, is 100%. This implies that the model correctly identifies all actual positive cases.
# 
# - The F1 score, which balances precision and recall, is 69.57%.
# 
# - The F2 score, which places more emphasis on recall, is 85.11%.
# 
# - The model's parameters indicate a regularization parameter (C) of 0.01 and a linear kernel.

# - The SVM rbf (radial basis function) model shows identical performance metrics compared to the SVM Linear model.
# - Both models achieve perfect recall, indicating they correctly identify all actual positive cases.
# - The parameters for the SVM rbf model include a regularization parameter (C) of 0.1, a gamma value of 1, and a radial basis function kernel.

# - Similar to the previous models, the SVM Poly (polynomial) model achieves consistent performance metrics with perfect recall.
# - The parameters for the SVM Poly model include a regularization parameter (C) of 0.01, a coefficient (coef0) of 10, and a polynomial kernel.

# ### Summary

# - All three SVM models exhibit identical performance across various metrics, indicating that they have the same predictive power on the given dataset.
# 
# - Despite achieving perfect recall, the models' precision is relatively low, suggesting a high false positive rate.
# The choice of hyperparameters (C, gamma, coef0) can significantly influence model performance and may require further optimization.
# 
# - Further analysis, such as cross-validation and feature engineering, may be necessary to improve model accuracy and address potential overfitting or underfitting issues.


```
**Feedback:**
Data Generation (30%) - 0.45/0.45
• Creativity and relevance of the chosen scenario - 0.15/0.15
• Correct implementation of data generation techniques - 0.15/0.15
• Clarity and structure of the data (including labeling and documentation) - 0.15/0.15

Model Implementation and Tuning (40%) - 0.4/0.6
• Correct implementation of SVC/SVR models with the three specified kernels - 0.2/0.2
• Effective use of hyper-parameter tuning, with a clear rationale for chosen parameters - 0.2/0.2
• Documentation of model performance metrics, with comparisons to justify the best-performing model - 0/0.2

Analysis and Discussion (30%)- 0.35/0.45
• Clarity and depth of the introduction and explanation of the chosen problem and dataset - 0.125/0.225
• Critical analysis of model results, including insights into performance variations across different kernels - 0.225/0.225

Need to mention whether you choose regression or classification task in introduction or in data generation notebook. The results table values doesn't match with the values and insights provided in the discussion. So, 0.3 marks were deducted.

---

# Student 47
```python
# ### Data Generation

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification

# ### Generating synthetic data

def generate_happiness_data(n_samples):
    X, y = make_classification(
        n_samples=n_samples, 
        n_features=2, 
        n_informative=2, 
        n_redundant=0, 
        n_clusters_per_class=1, 
        random_state=42
    )
    # Convert to meaningful ranges
    X[:, 0] = 30 + 20 * X[:, 0]  # Social interaction range: 30-50 units
    X[:, 1] = 30 + 20 * X[:, 1]  # Work-life balance range: 30-50 units
    
    # Introduce some randomness for happiness level
    y = np.random.randint(2, size=n_samples)
    
    return X, y

X, y = generate_happiness_data(100)

# ### Visualizing the synthetic data

plt.figure(figsize=(10, 6))
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='bwr', marker='o')
plt.xlabel('Social Interaction')
plt.ylabel('Work-Life Balance')
plt.title('Synthetic Data - Predicting Happiness')
plt.colorbar(label='Happy (1) / Not Happy (0)')
plt.grid(True)
plt.show()

# Transfer data to DataFrame
data = pd.DataFrame(X, columns=['social_interaction', 'work_life_balance'])
data['happy'] = y

data

# Save data to CSV
data.to_csv('happiness_data.csv', index=False)

# #### 1. Importing important libraries

import pandas as pd
from sklearn.svm import SVC
from matplotlib import pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV

np.random.seed(1)

# #### 2. Load data

df = pd.read_csv('happiness_data.csv') # let's use the same data as we did in the logistic regression example
df.head(3)

# Use sklearn to split df into a training set and a test set

X = df[['social_interaction', 'work_life_balance']]
y = df['happy']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)


# #### 3. Model the data

performance = pd.DataFrame({"model": [], "Accuracy": [], "Precision": [], "Recall": [], "F1": [], "F2": [], "Parameters": []})

# create a fbeta 2 scorer
from sklearn.metrics import make_scorer
from sklearn.metrics import fbeta_score
f2_scorer = make_scorer(fbeta_score, beta=2)

# #### 3.1 Fit a SVM classification model using linear kernal

# defining parameter range 
param_grid = {'C': [0.01, 0.1, 0.5, 1, 5, 10, 50, 100],  
              'kernel': ['linear']}
  
#grid = GridSearchCV(SVC(), param_grid, scoring='f1', refit = True, verbose = 3, n_jobs=-1) 

grid = GridSearchCV(SVC(), param_grid, cv=5,scoring=f2_scorer, refit = True, verbose = 3, n_jobs=-1) 
  
# fitting the model for grid search 
_ = grid.fit(X_train, y_train)

# print best parameter after tuning 
print(grid.best_params_) 
  
# print how our model looks after hyper-parameter tuning 
print(grid.best_estimator_)

y_pred = grid.predict(X_test) 

recall = recall_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
f2 = fbeta_score(y_test, y_pred, beta=2)

performance = pd.concat([performance, pd.DataFrame({"model": ["SVM Linear"], "Accuracy": [accuracy], "Precision": [precision], "Recall": [recall], "F1": [f1], "F2": [f2], "Parameters": [grid.best_params_]})])


performance

# #### 3.2 Fit a SVM classification model using rbf kernal

# defining parameter range 
param_grid = {'C': [0.1, 1, 10, 100],  
              'gamma': [1, 0.1, 0.01, 0.001, 0.0001], 
              'kernel': ['rbf']}
  
grid = GridSearchCV(SVC(), param_grid, scoring=f2_scorer, refit = True, verbose = 3, n_jobs=-1) 
  
# fitting the model for grid search 
_ = grid.fit(X_train, y_train)

# print best parameter after tuning 
print(grid.best_params_) 
  
# print how our model looks after hyper-parameter tuning 
print(grid.best_estimator_)

y_pred = grid.predict(X_test) 

recall = recall_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
f2 = fbeta_score(y_test, y_pred, beta=2)

performance = pd.concat([performance, pd.DataFrame({"model": ["SVM rbf"], "Accuracy": [accuracy], "Precision": [precision], "Recall": [recall], "F1": [f1], "F2": [f2],"Parameters": [grid.best_params_]})])

# #### 3.3 Fit a SVM classification model using polynomial kernal

# defining parameter range 
param_grid = {'C': [0.01, 0.1, 0.5, 1, 5, 10, 50, 100],  
              'coef0': [0.01, 0.1, 0.5, 1, 5, 10, 50, 100],
              'kernel': ['poly']}
  
grid = GridSearchCV(SVC(), param_grid, scoring=f2_scorer, refit = True, verbose = 3, n_jobs=-1) 
  
# fitting the model for grid search 
_ = grid.fit(X_train, y_train)

# print best parameter after tuning 
print(grid.best_params_) 
  
# print how our model looks after hyper-parameter tuning 
print(grid.best_estimator_)

y_pred = grid.predict(X_test) 

recall = recall_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
f2 = fbeta_score(y_test, y_pred, beta=2)

performance = pd.concat([performance, pd.DataFrame({"model": ["SVM Poly"], "Accuracy": [accuracy], "Precision": [precision], "Recall": [recall], "F1": [f1], "F2": [f2], "Parameters": [grid.best_params_]})])

performance.sort_values(by="F2", ascending=False)

# ### Summary:

# - The SVM model with the Radial Basis Function (RBF) kernel achieved an accuracy of 50%, indicating that it correctly classified half of the instances. 
# - The precision of 55% suggests that when it predicted positive, it was correct 55% of the time. The recall of 64.71% indicates the model captured about two-thirds of the positive instances. 
# - The F1 score, which balances precision and recall, is 59.46%. The F2 score, which gives more weight to recall, is 62.50%. The chosen hyperparameters for this model are C=100 and gamma=0.0001.

# - The SVM model with the Polynomial kernel also achieved similar performance as the RBF kernel. The accuracy, precision, recall, F1, and F2 scores are identical. 
# - The hyperparameters chosen for this model are C=1 and coef0=100.

# - The SVM model with the Linear kernel achieved a slightly higher accuracy of 56.67% compared to the models with RBF and Polynomial kernels. 
# - The precision of 62.50% indicates that the model correctly predicted positive cases 62.50% of the time. The recall of 58.82% suggests that it captured nearly 60% of the positive instances. 
# - The F1 score and F2 score are 60.61% and 59.52%, respectively. The chosen hyperparameters for this model are C=0.01.

# ### Overall Discussion:

# - The models with RBF and Polynomial kernels show similar performance, indicating that they may be capturing similar patterns in the data.
# - The Linear kernel outperforms the other two in terms of accuracy but still faces challenges in achieving a higher accuracy, likely due to the complexity of the underlying data.
# - The choice of hyperparameters plays a crucial role, and further fine-tuning might be necessary to improve the model performance.
# - The interpretation of these results should consider the specific context of the problem and the trade-offs between precision and recall based on the application requirements.


```
**Feedback:**
Data Generation (30%) - 0.45/0.45
• Creativity and relevance of the chosen scenario - 0.15/0.15
• Correct implementation of data generation techniques - 0.15/0.15
• Clarity and structure of the data (including labeling and documentation) - 0.15/0.15

Model Implementation and Tuning (40%) - 0.4/0.6
• Correct implementation of SVC/SVR models with the three specified kernels - 0.2/0.2
• Effective use of hyper-parameter tuning, with a clear rationale for chosen parameters - 0.2/0.2
• Documentation of model performance metrics, with comparisons to justify the best-performing model - 0/0.2

Analysis and Discussion (30%)- 0.225/0.45
• Clarity and depth of the introduction and explanation of the chosen problem and dataset - 0/0.225
• Critical analysis of model results, including insights into performance variations across different kernels - 0.225/0.225

No introduction section provided and the results table values doesn't match with the values in the discussion. So, 0.3 marks were deducted.

---

# Student 48
```python
# # Introduction
# 
# In this analysis, we utilized the Default dataset to predict whether individuals would default on their loans based on their credit score and income level. This dataset predicts whether someone will default based on their credit score, which ranges from 350 to 800, and their income level, which ranges from $20000 to $120000. It has two input variables and a binary target variable, where 1 signifies default and 0 represents non-default. If an individual's credit score is less than 650 and their income level is less than $60000, they will be considered as a defaulter. Extra noises have been added to the dataset to simulate a real-world scenario.
# 
# We employed Support Vector Machines (SVM) with different kernels (linear, polynomial, and radial basis function (rbf)) to fit models to the data and predict default likelihood. Additionally, we used GridSearchCV to optimize hyperparameters, focusing on a custom beta score with a beta of 2 to prioritize recall over precision due to the higher cost associated with false negatives
# 
# The reason for using this metric is that there is a significant difference between the cost of a false positive and a false negative in this case. A false negative is much more expensive than a false positive because it means that someone who is likely to default is predicted as not defaulting. However, we cannot entirely ignore precision because we do not want to deny loans to eligible candidates

# ## 1. Setup
# Import modules

import pandas as pd
from sklearn.svm import SVC
from matplotlib import pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV

np.random.seed(1)

# ## 2. Load data

# Load data (it's already cleaned and preprocessed)

# Uncomment the following snippet of code to debug problems with finding the .csv file path
# This snippet of code will exit the program and print the current working directory.
#import os
#print(os.getcwd())

df = pd.read_csv('Default.csv') 
df.head(5)

# Use sklearn to split df into a training set and a test set

X = df[['Credit Score','Income']]
y = df['Default']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)


# ## 3. Model the data

# First, let's create a dataframe to load the model performance metrics into.

performance = pd.DataFrame({"model": [], "Accuracy": [], "Precision": [], "Recall": [], "F1": [], "F2": [], "Parameters": []})

# create a fbeta 2 scorer
from sklearn.metrics import make_scorer
from sklearn.metrics import fbeta_score
f2_scorer = make_scorer(fbeta_score, beta=2)


# ### 3.1 Fit a SVM classification model using linear kernal

# defining parameter range 
param_grid = {'C': [0.01, 0.1, 0.5, 1, 5, 10, 50, 100],  
              'kernel': ['linear']}
  
#grid = GridSearchCV(SVC(), param_grid, scoring='f1', refit = True, verbose = 3, n_jobs=-1) 

grid = GridSearchCV(SVC(), param_grid, scoring=f2_scorer, refit = True, verbose = 3, n_jobs=-1) 
  
# fitting the model for grid search 
_ = grid.fit(X_train, y_train)

# print best parameter after tuning 
print(grid.best_params_) 
  
# print how our model looks after hyper-parameter tuning 
print(grid.best_estimator_)

y_pred = grid.predict(X_test) 

recall = recall_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
f2 = fbeta_score(y_test, y_pred, beta=2)

performance = pd.concat([performance, pd.DataFrame({"model": ["SVM Linear"], "Accuracy": [accuracy], "Precision": [precision], "Recall": [recall], "F1": [f1], "F2": [f2], "Parameters": [grid.best_params_]})])


performance

# ### 3.2 Fit a SVM classification model using rbf kernal

# defining parameter range 
param_grid = {'C': [0.1, 1, 10, 100],  
              'gamma': [1, 0.1, 0.01, 0.001, 0.0001], 
              'kernel': ['rbf']}
  
grid = GridSearchCV(SVC(), param_grid, scoring=f2_scorer, refit = True, verbose = 3, n_jobs=-1) 
  
# fitting the model for grid search 
_ = grid.fit(X_train, y_train)

# print best parameter after tuning 
print(grid.best_params_) 
  
# print how our model looks after hyper-parameter tuning 
print(grid.best_estimator_)

y_pred = grid.predict(X_test) 

recall = recall_score(y_test, y_pred)
precision = precision_score(y_test, y_pred,zero_division=0)
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
f2 = fbeta_score(y_test, y_pred, beta=2)

performance = pd.concat([performance, pd.DataFrame({"model": ["SVM rbf"], "Accuracy": [accuracy], "Precision": [precision], "Recall": [recall], "F1": [f1], "F2": [f2],"Parameters": [grid.best_params_]})])

performance

# ### 3.3 Fit a SVM classification model using polynomial kernal

# defining parameter range 
param_grid = {'C': [0.01, 0.1, 0.5, 1, 5, 10, 50, 100],  
              'coef0': [0.01, 0.1, 0.5, 1, 5, 10, 50, 100],
              'kernel': ['poly']}
  
grid = GridSearchCV(SVC(), param_grid, scoring=f2_scorer, refit = True, verbose = 3, n_jobs=-1) 
  
# fitting the model for grid search 
_ = grid.fit(X_train, y_train)

# print best parameter after tuning 
print(grid.best_params_) 
  
# print how our model looks after hyper-parameter tuning 
print(grid.best_estimator_)

y_pred = grid.predict(X_test) 

recall = recall_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
f2 = fbeta_score(y_test, y_pred, beta=2)

performance = pd.concat([performance, pd.DataFrame({"model": ["SVM Poly"], "Accuracy": [accuracy], "Precision": [precision], "Recall": [recall], "F1": [f1], "F2": [f2], "Parameters": [grid.best_params_]})])

# ## 4.0 Summary

# From out results, we can see that the linear kernel and rbf models perform the best. For the SVC model with a linear kernel, the best C value is 0.1. For the SVC model with a rbf kernel, the best C value is 0.1 and the best gamma value is 0.1. The polynomial kernel model did not perform as well as the other two models and therefore we will not use it.

performance.sort_values(by="F2", ascending=False)

# # Discussion

# Analysis of Results:
# 
# SVM Polynomial Kernel: This model achieved the highest accuracy (93.33%) and F2 score (0.961538), indicating that it has the best balance between recall and precision. With a recall of 1.0, it correctly identifies all defaulters, which is crucial for this task. The parameters chosen (C=100, coef0=50) suggest a relatively high penalty for misclassifications and a significant influence of higher-degree polynomial terms.
# 
# SVM Linear Kernel: While having a lower accuracy (83.33%) and F2 score (0.784314) compared to the polynomial kernel, it still performs reasonably well. However, it has lower recall (0.8) compared to the polynomial kernel, indicating it might miss some default cases. The chosen parameter (C=0.1) suggests a relatively low penalty for misclassifications.
# 
# SVM RBF Kernel: This model performed the worst among the three, with an accuracy of 66.67% and F2 score of 0.0. It failed to correctly identify any default cases (recall of 0.0), resulting in a complete lack of meaningful predictions. The chosen parameters (C=0.1, gamma=1) indicate a relatively loose decision boundary with a high gamma value, possibly leading to overfitting.
# 
# According to the analysis, the polynomial kernel SVM model was the most effective in predicting the chance of default, with the highest recall, which is a crucial aspect of this task. This suggests that the polynomial kernel was able to capture the non-linear relationships between the input variables and the target variable better than the linear and rbf kernels. Moreover, the significant difference in performance between the polynomial kernel and the rbf kernel emphasizes the importance of selecting an appropriate kernel for SVM models. Finally, the choice of hyperparameters had a significant impact on the model's performance, which highlights the importance of hyperparameter tuning using techniques like GridSearchCV.




```
**Feedback:**
Data Generation (30%) - 0/0.45
• Creativity and relevance of the chosen scenario - 0/0.15
• Correct implementation of data generation techniques - 0/0.15
• Clarity and structure of the data (including labeling and documentation) - 0/0.15

Model Implementation and Tuning (40%) - 0.6/0.6
• Correct implementation of SVC/SVR models with the three specified kernels - 0.2/0.2
• Effective use of hyper-parameter tuning, with a clear rationale for chosen parameters - 0.2/0.2
• Documentation of model performance metrics, with comparisons to justify the best-performing model - 0.2/0.2

Analysis and Discussion (30%)- 0.45/0.45
• Clarity and depth of the introduction and explanation of the chosen problem and dataset - 0.225/0.225
• Critical analysis of model results, including insights into performance variations across different kernels - 0.225/0.225

The data generation file is not submitted. So, 0.45 marks were deducted.

---

# Student 49
```python
# 
# # Synthetic Data Generation and Model Training
# This notebook demonstrates the process of generating a synthetic dataset, exploring the data, and then training a model using Support Vector Machine (SVC) with hyper-parameter tuning to predict whether a customer will buy a product based on their age and estimated salary.


import pandas as pd
import numpy as np
import random

np.random.seed(0)  # For reproducibility

# Generating synthetic data with noise
age = np.random.randint(18, 65, 1000)  # Age range from 18 to 65
estimated_salary = np.random.randint(10000, 100000, 1000)  # Estimated salary range


# Noise addition
noise = np.random.normal(-10000, 10000, 1000)  # Adding some noise to the salary to make the boundary less perfect
noise_age = np.random.randint(0, 6, 1000)



# Assuming a simplistic decision boundary: older and higher-earning individuals are more likely to buy, with some noise
buy_decision = ((age ) > 40) & ((estimated_salary ) > 50000)  # Simplistic decision boundary with noise
#buy_decision = buy_decision.astype(int)  # Convert to integer (0 or 1)

# Creating a DataFrame with noisy data
data_noisy = pd.DataFrame({
    'Age': age + noise_age,
    'EstimatedSalary': estimated_salary + noise,  # Adding noise to the salary
    'Buy': buy_decision
})

data_noisy.head()



import matplotlib.pyplot as plt
import seaborn as sns

# Data exploration
fig, ax = plt.subplots(1, 2, figsize=(16, 6))

sns.scatterplot(data=data_noisy, x='Age', y='EstimatedSalary', hue='Buy', style='Buy', ax=ax[0])
ax[0].set_title('Scatter Plot of Age vs. Estimated Salary')

sns.countplot(x='Buy', data=data_noisy, ax=ax[1])
ax[1].set_title('Count Plot of Buy Decision')

plt.tight_layout()



from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

# Splitting data into features and target
X = data_noisy[['Age', 'EstimatedSalary']]
y = data_noisy['Buy']

# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# Feature Scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Grid search for hyper-parameter tuning on three SVC kernels
parameters = {
    'kernel': ['linear', 'rbf', 'poly'],
    'C': [0.1, 1, 10],
    'degree': [2, 3, 4],  # Only used for poly kernel
}

svc = SVC()
clf = GridSearchCV(svc, parameters, scoring='accuracy')
clf.fit(X_train, y_train)

# Best parameters and best score
best_parameters = clf.best_params_
best_score = clf.best_score_

print(f"Best Parameters: {best_parameters}")
print(f"Best Score: {best_score}")


# 
# ## Discussion and Analysis
# The results demonstrate the effectiveness of SVC and the importance of hyper-parameter tuning in achieving high model performance. The choice of kernel significantly impacts the model's ability to capture complex patterns in the data. In this case, the `rbf` kernel provided the best results, likely due to its ability to handle non-linear decision boundaries more effectively than the `linear` kernel and with better generalization than the `poly` kernel for this particular dataset.
# 
# This analysis underscores the necessity of experimenting with different model configurations and the potential of SVC for classification problems, especially when dealing with non-linear relationships between features and the target variable.




```
**Feedback:**
Data Generation (30%) - 0.4/0.45
• Creativity and relevance of the chosen scenario - 0.15/0.15
• Correct implementation of data generation techniques - 0.15/0.15
• Clarity and structure of the data (including labeling and documentation) - 0.1/0.15

Model Implementation and Tuning (40%) - 0.6/0.6
• Correct implementation of SVC/SVR models with the three specified kernels - 0.2/0.2
• Effective use of hyper-parameter tuning, with a clear rationale for chosen parameters - 0.2/0.2
• Documentation of model performance metrics, with comparisons to justify the best-performing model - 0.2/0.2

Analysis and Discussion (30%)- 0.45/0.45
• Clarity and depth of the introduction and explanation of the chosen problem and dataset - 0.225/0.225
• Critical analysis of model results, including insights into performance variations across different kernels - 0.225/0.225

Need to create a dedicated notebook for data generation as this keeps your code more organized. So, 0.05 marks were deducted

---

