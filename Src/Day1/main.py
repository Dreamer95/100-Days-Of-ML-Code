import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer  # Updated import
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split  # Updated import
from sklearn.preprocessing import StandardScaler

# Step 2: Importing dataset
dataset = pd.read_csv('/Users/dongdinh/Documents/Learning/100-Days-Of-ML-Code/datasets/Data.csv')
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, 3].values

# Step 3: Handling the missing data
imputer = SimpleImputer(missing_values=np.nan, strategy="mean")  # Updated usage
imputer = imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])

# Step 4: Encoding categorical data
labelencoder_X = LabelEncoder()
X[:, 0] = labelencoder_X.fit_transform(X[:, 0])

# Creating a dummy variable (updated approach)
from sklearn.compose import ColumnTransformer
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])], remainder='passthrough')
X = np.array(ct.fit_transform(X))

labelencoder_Y = LabelEncoder()
Y = labelencoder_Y.fit_transform(Y)

# Step 5: Splitting the datasets into training sets and Test sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

# Step 6: Feature Scaling
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)  # Only transform, don't fit again

print("Data preprocessing completed successfully!")
print(f"Training set shape: {X_train.shape}")
print(f"Test set shape: {X_test.shape}")