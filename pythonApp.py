import pandas as pd
import numpy as np
from scipy.stats import zscore
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
import matplotlib.pyplot as plt
#import seaborn as sns
import matplotlib
matplotlib.use('Agg')

# Step 1: Load the dataset
df = pd.read_csv('/tmp/dataset02.csv')
#print("Initial dataset columns:", df.columns)
#print("Dataset shape after dropping NaN values:", df_cleaned.shape)
#print("Columns after removing non-numeric data:", df_cleaned.columns)

# Convert all columns to numeric, forcing errors to NaN
df = df.apply(pd.to_numeric, errors='coerce')

# Check the columns after conversion
print("Columns after conversion to numeric:", df.columns)

#print("Initial dataset columns:", df.columns)
#print("Dataset shape before cleaning:", df.shape)

df_cleaned = df.dropna()
print("Columns after removing non-numeric columns:", df_cleaned.columns)
# Step 3: Keep only numeric columns
#df_cleaned = df_cleaned.select_dtypes(include=[np.number])
#print("Columns after removing non-numeric columns:", df_cleaned.columns)
# Step 2: Data Cleaning - Drop non-numerical and NaN values
#df = df.select_dtypes(include=['float64', 'int64']).dropna()

# Step 3: Remove outliers using Z-score method
z_scores = np.abs(zscore(df_cleaned))
df_cleaned = df_cleaned[(z_scores < 3).all(axis=1)]
print("Dataset shape after cleaning:", df_cleaned.shape)
print("Columns after cleaning:", df_cleaned.columns)


#z_scores = np.abs(zscore(df))
#df_cleaned = df[(z_scores < 3).all(axis=1)]  # Keep rows where all z-scores are below 3 (common threshold)

# Step 4: Outlier Removal using IQR method
Q1 = df_cleaned.quantile(0.25)
Q3 = df_cleaned.quantile(0.75)
IQR = Q3 - Q1

# Step 9: Apply IQR-based filtering (Ensure 'x' and 'y' are present)
if 'x' in df_cleaned.columns and 'y' in df_cleaned.columns:
    df_cleaned = df_cleaned[(df_cleaned['x'] >= Q1['x'] - 1.5 * IQR['x']) & (df_cleaned['x'] <= Q3['x'] + 1.5 * IQR['x']) &
                            (df_cleaned['y'] >= Q1['y'] - 1.5 * IQR['y']) & (df_cleaned['y'] <= Q3['y'] + 1.5 * IQR['y'])]
else:
    print("Error: 'x' or 'y' column is missing after cleaning.")
    exit()

# Step 10: Print the resulting dataset shape and a preview
print("Dataset shape after cleaning and outlier removal:", df_cleaned.shape)
print(df_cleaned.head())

# Remove rows where any feature is an outlier based on IQR
#df_cleaned = df_cleaned[~((df_cleaned < (Q1 - 1.5 * IQR)) | (df_cleaned > (Q3 + 1.5 * IQR))).any(axis=1)]
# Step 2: Apply IQR-based filtering

#df_cleaned = df_cleaned[(df_cleaned['x'] >= Q1['x'] - 1.5 * IQR['x']) & (df_cleaned['x'] <= Q3['x'] + 1.5 * IQR['x']) &
 #                       (df_cleaned['y'] >= Q1['y'] - 1.5 * IQR['y']) & (df_cleaned['y'] <= Q3['y'] + 1.5 * IQR['y'])]

# Print the resulting dataset shape and a preview to check the output
#print("Dataset shape after IQR-based outlier removal:", df_cleaned.shape)
#print(df_cleaned.head())

# Step 5: Normalize the data using Min-Max scaling
scaler = MinMaxScaler()
df_normalized = pd.DataFrame(scaler.fit_transform(df_cleaned), columns=df_cleaned.columns)

print("Normalized dataset:")
print(df_normalized.head())

# Step 6: Split dataset into training (80%) and testing (20%)
#train_df, test_df = train_test_split(df_normalized, test_size=0.2, random_state=42)
# Splitting the dataset
#train_df, test_df = train_test_split(df_cleaned, test_size=0.2, random_state=42)

# Determine training size (80%)
train_size = int(0.8 * len(df_cleaned))

# Split into training and testing sets
training_data = df_cleaned.iloc[:train_size]
testing_data = df_cleaned.iloc[train_size:]

print(f"Training data shape: {training_data.shape}")
print(f"Testing data shape: {testing_data.shape}")

training_data.to_csv('/tmp/dataset02_training.csv', index=False)
testing_data.to_csv('/tmp/dataset02_testing.csv', index=False)

# Separate features (x) and target variable (y)
X_train = training_data['x']
y_train = training_data['y']
X_test = testing_data['x']
y_test = testing_data['y']



# Step 7: Save the training and testing datasets as CSV files
#train_df.to_csv('/tmp/dataset02_training.csv', index=False)
#test_df.to_csv('/tmp/dataset02_testing.csv', index=False)
#print("Training dataset saved to /tmp/dataset02_training.csv")
#print("Testing dataset saved to /tmp/dataset02_testing.csv")
# Step 8: Inspect the columns to debug and replace 'x' and 'y' with actual column names
#print("Columns in dataset:", df.columns)
#print("Columns in training data:", train_df.columns)

# Modify these based on the correct column names, for example, 'feature1' and 'target'
#X_train = train_df['x']  # Replace with actual column name for x
#y_train = train_df['y']    # Replace with actual column name for y

# Add constant to independent variable (for the intercept)
#X_train = sm.add_constant(X_train)

# Step 9: Create and fit the OLS model
#ols_model = sm.OLS(y_train, X_train).fit()

# OLS Model Training
X_train = training_data['x']  # Independent variable (training)
y_train = training_data['y']  # Dependent variable (training)

X_test = testing_data['x']    # Independent variable (testing)
y_test = testing_data['y']    # Dependent variable (testing)


# Add a constant to the X_train (required for OLS)
#X_train_const = sm.add_constant(X_train)

# Fit the OLS model
#ols_model = sm.OLS(y_train, X_train_const).fit()

X_train_with_const = sm.add_constant(X_train)  # Add constant for the intercept
model = sm.OLS(y_train, X_train_with_const).fit()


# Summary of the OLS model
print("OLS Model Summary (Training Data Only):")
#print(ols_model.summary())
print(model.summary())
# Step 10: Print the summary of the model
#print(ols_model.summary())

# Step 11: Save the OLS model results to a file
#ols_model.save('/tmp/UE_05_App1_OLS_model')

# Step 12: Create scatter plot for training and testing data
#plt.scatter(train_df['x'], train_df['y'], color='orange', label='Training Data')  # Adjust column names here
#plt.scatter(test_df['x'], test_df['y'], color='blue', label='Testing Data')      # Adjust column names here


# Step 3: Create the scatter plot
plt.figure(figsize=(10, 6))

# Plot the training data (orange color)
plt.scatter(X_train, y_train, color='orange', label='Training Data')

# Plot the testing data (blue color)
plt.scatter(X_test, y_test, color='blue', label='Testing Data')

# Step 4: Create the red line plot of the OLS regression model
# Calculate the predicted values based on the model for the entire dataset
X_combined = np.concatenate([X_train, X_test])
X_combined_with_const = sm.add_constant(X_combined)
y_pred = model.predict(X_combined_with_const)

# Sort X_combined for smooth line plotting
sorted_idx = np.argsort(X_combined)
plt.plot(X_combined[sorted_idx], y_pred[sorted_idx], color='red', label='OLS Regression Line')

# Step 5: Customize the plot
plt.title("Scatter Plot with OLS Regression Line")
plt.xlabel("Influence Data (X)")
plt.ylabel("Target Variable (Y)")
plt.legend()
plt.grid(True)

# Step 6: Save the plot to a PDF file
plt.savefig("/tmp/UE_04_App2_ScatterVisualizationAndOlsModel.pdf")

# Show the plot
plt.show()



# Step 2: Generate Box Plot
plt.figure(figsize=(10, 6))
df_cleaned.boxplot()

# Step 3: Customize the plot
plt.title('Box Plot of All Dimensions')
plt.ylabel('Values')

# Step 4: Save the figure as a PDF
plt.savefig('/tmp/UE_04_App2_BoxPlot.pdf')
print("Box plot saved as 'UE_04_App2_BoxPlot.pdf' in the /tmp directory.")





# Scatter Plot: Training data (orange) and Testing data (blue)
#plt.figure(figsize=(10, 6))
#plt.scatter(X_train, y_train, color='orange', label='Training Data')
#plt.scatter(X_test, y_test, color='blue', label='Testing Data')

# Red Line Plot: OLS regression line (using the OLS model coefficients)
# Use the coefficient from the OLS model
# y = coef_x * x + intercept
#intercept = model.params['const']
#coef_x = model.params['x']

# Create a line using the model's coefficients
#x_line = np.linspace(min(df_cleaned['x']), max(df_cleaned['x']), 100)
#y_line = coef_x * x_line + intercept
#plt.plot(x_line, y_line, color='red', label='OLS Regression Line')

# Add labels, title, and legend
#plt.xlabel('Influence Data (x)')
#plt.ylabel('Target Variable (y)')
#plt.title('Scatter Plot of Training and Testing Data with OLS Regression Line')
#plt.legend()

# Save the plot to a PDF file
#plt.savefig('/tmp/UE_04_App2_ScatterVisualizationAndOlsModel.pdf')

# Display the plot
#plt.show()







# Scatter plot for testing data
#plt.scatter(X_train, y_train, color='orange', label='Training Data')

# Scatter plot for testing data
#plt.scatter(X_test, y_test, color='blue', label='Testing Data')

# Regression line
#x_range = np.linspace(df_cleaned['x'].min(), df_cleaned['x'].max(), 100)
#y_line = model.params['const'] + model.params['x'] * x_range
#plt.plot(x_range, y_line, color='red', label='Regression Line')



# Add a red line for the regression line
#plt.plot(train_df['x'], ols_model.predict(sm.add_constant(train_df['x'])), color='red', label='OLS Line')

# Add labels, legend, and title
#plt.xlabel('x (Independent Variable)')
#plt.ylabel('y (Target Variable)')
#plt.title('Scatter Plot and Regression Line')
#plt.legend()


# Save figure as PDF
#plt.savefig('/tmp/UE_04_App2_ScatterVisualizationAndOlsModel.pdf', format='pdf')

# Show plot (optional)
#plt.show()


# Save the scatter plot to a PDF
#plt.savefig('/tmp/UE_04_App2_ScatterVisualizationAndOlsModel.pdf')

# Step 13: Create boxplot of normalized data
#plt.boxplot(df_normalized.values, labels=df_normalized.columns)
#plt.title('Boxplot of Normalized Data')

# Save the boxplot to a PDF
#plt.savefig('/tmp/UE_04_App2_BoxPlot.pdf')

# Step 14: Create Diagnostic Plots

# Residuals vs Fitted values plot (for heteroscedasticity check)
#plt.figure(figsize=(8, 6))
#plt.scatter(ols_model.fittedvalues, ols_model.resid, color='blue', edgecolor='black', alpha=0.5)
#plt.axhline(y=0, color='red', linestyle='--')
#plt.xlabel('Fitted values')
#plt.ylabel('Residuals')
#plt.title('Residuals vs Fitted')
#plt.savefig('/tmp/diagnostic_residuals_vs_fitted.pdf')

# Normal Q-Q plot (for normality check of residuals)
#sm.qqplot(ols_model.resid, line='45', fit=True)
#plt.title('Q-Q Plot')
#plt.savefig('/tmp/diagnostic_qq_plot.pdf')

# Scale-Location plot (for checking homoscedasticity)
#plt.figure(figsize=(8, 6))
#plt.scatter(ols_model.fittedvalues, np.sqrt(np.abs(ols_model.resid)), color='blue', edgecolor='black', alpha=0.5)
#plt.axhline(y=0, color='red', linestyle='--')
#plt.xlabel('Fitted values')
#plt.ylabel('Sqrt(|Residuals|)')
#plt.title('Scale-Location')
#plt.savefig('/tmp/diagnostic_scale_location.pdf')

# Leverage vs Standardized Residuals (for influential points)
#from statsmodels.graphics.regressionplots import influence_plot
#plt.figure(figsize=(8, 6))
#influence_plot(ols_model, criterion="cooks")
#plt.title('Leverage vs Standardized Residuals')
#plt.savefig('/tmp/diagnostic_leverage_vs_residuals.pdf')

# Optional: Print and save all diagnostics into one document
# You can save all these plots to a multi-page PDF if desired.

#print("Diagnostic plots saved successfully.")

