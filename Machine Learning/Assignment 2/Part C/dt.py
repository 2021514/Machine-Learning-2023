import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

data = pd.read_csv('Thyroid.csv')
print(data.T[-1].unique())
# # print(d)
# data = np.array(d)
# print(data)

label_encoder = LabelEncoder()

# Iterate through the columns and encode the categorical ones
categorical_columns = ["sex", "on thyroxine", "query on thyroxine", "on antithyroid medication", "sick", "pregnant", "thyroid surgery",
           "I131 treatment", "query hypothyroid", "query hyperthyroid", "lithium", "goitre", "tumor", "hypopituitary",
           "psych", "TSH measured", "T3 measured", "TT4 measured", "T4U measured", "FTI measured", "TBG measured", "referral source", "label"]

for column in categorical_columns:
    data[column] = label_encoder.fit_transform(data[column])

# Save the encoded data to a new CSV file
data.to_csv('encoded_data.csv', index=False)

# Display the first few rows of the encoded data
print(data.head())