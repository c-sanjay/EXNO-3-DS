## EXNO-3-DS

# AIM:
To read the given data and perform Feature Encoding and Transformation process and save the data to a file.

# ALGORITHM:
```
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Encoding for the feature in the data set.
STEP 4:Apply Feature Transformation for the feature in the data set.
STEP 5:Save the data to the file.
```
# FEATURE ENCODING:

1. Ordinal Encoding
An ordinal encoding involves mapping each unique label to an integer value. This type of encoding is really only appropriate if there is a known relationship between the categories. This relationship does exist for some of the variables in our dataset, and ideally, this should be harnessed when preparing the data.
2. Label Encoding
Label encoding is a simple and straight forward approach. This converts each value in a categorical column into a numerical value. Each value in a categorical column is called Label.
3. Binary Encoding
Binary encoding converts a category into binary digits. Each binary digit creates one feature column. If there are n unique categories, then binary encoding results in the only log(base 2)ⁿ features.
4. One Hot Encoding
We use this categorical data encoding technique when the features are nominal(do not have any order). In one hot encoding, for each level of a categorical feature, we create a new variable. Each category is mapped with a binary variable containing either 0 or 1. Here, 0 represents the absence, and 1 represents the presence of that category.

# Methods Used for Data Transformation:
  # 1. FUNCTION TRANSFORMATION
• Log Transformation
• Reciprocal Transformation
• Square Root Transformation
• Square Transformation
  # 2. POWER TRANSFORMATION
• Boxcox method
• Yeojohnson method

# CODING AND OUTPUT:
```py
import pandas as pd
df=pd.read_csv("Encoding Data.csv")
df
```
<img width="343" height="435" alt="image" src="https://github.com/user-attachments/assets/b1fc448e-a1fc-42db-b6f9-c7cc9f71630a" />

```py
from sklearn.preprocessing import LabelEncoder,OrdinalEncoder
pm=['Hot','Warm','Cold']
e1=OrdinalEncoder(categories=[pm])
e1.fit_transform(df[["ord_2"]])
```
<img width="158" height="231" alt="image" src="https://github.com/user-attachments/assets/259ad1ce-a0de-48d4-9c64-9827959dec5b" />

```py
df['bo2']=e1.fit_transform(df[["ord_2"]])
df
```
<img width="384" height="437" alt="image" src="https://github.com/user-attachments/assets/ff3bfec5-f316-4c67-8aae-2817afc48261" />

```py
le=LabelEncoder()
dfc=df.copy()
dfc['ord_2']=le.fit_transform(dfc['ord_2'])
dfc
```
<img width="386" height="371" alt="image" src="https://github.com/user-attachments/assets/cf7e1daf-797f-4428-a921-bcb583510958" />

```py
from sklearn.preprocessing import OneHotEncoder
ohe=OneHotEncoder(sparse_output=False)
df2=df.copy()
enc=pd.DataFrame(ohe.fit_transform(df2[["nom_0"]]))
df2=pd.concat([df2,enc],axis=1)
df2
```
<img width="522" height="442" alt="image" src="https://github.com/user-attachments/assets/ab283076-5d63-4f89-936f-61d24f7f1053" />

```py
pd.get_dummies(df2,columns=["nom_0"])
```
<img width="793" height="443" alt="image" src="https://github.com/user-attachments/assets/ffbecf48-c52c-4a62-9b88-b085bfafcc01" />

```py
from category_encoders import BinaryEncoder
df=pd.read_csv("data.csv")
df
```
<img width="581" height="446" alt="image" src="https://github.com/user-attachments/assets/0bf1a537-eab1-47e0-a3a0-c4b07f8e532b" />

```py
be=BinaryEncoder()
nd=be.fit_transform(df['Ord_2'])
df
```
<img width="596" height="433" alt="image" src="https://github.com/user-attachments/assets/f2d9b4a6-e901-4932-8e47-e1fa909410d1" />

```py
dfb=pd.concat([df,nd],axis=1)
dfb
```

```py
from category_encoders import TargetEncoder
te=TargetEncoder()
CC=df.copy()
new=te.fit_transform(X=CC["City"],y=CC["Target"])
CC=pd.concat([CC,new],axis=1)
CC
```
<img width="669" height="439" alt="image" src="https://github.com/user-attachments/assets/f147e125-28de-4848-910b-7e60e4ad5e8b" />

```py
import pandas as pd
from scipy import stats
import numpy as np
df=pd.read_csv("Data_to_Transform.csv")
df
```
<img width="961" height="515" alt="image" src="https://github.com/user-attachments/assets/fb05289e-a1a7-4074-b145-d4a74c762e7c" />

```py
df.skew()
```
<img width="364" height="244" alt="image" src="https://github.com/user-attachments/assets/eef4a70b-3ae8-472e-9e25-f6b2fdd13c79" />

```py
np.log(df["Highly Positive Skew"])
```
<img width="295" height="559" alt="image" src="https://github.com/user-attachments/assets/dac0eb2c-2d1d-4c75-a185-2d9f713523ff" />

```py
np.reciprocal(df["Moderate Positive Skew"])
```
<img width="342" height="561" alt="image" src="https://github.com/user-attachments/assets/17cd6707-86c4-4de6-8e99-13d452833cb1" />

```py
np.sqrt(df["Highly Positive Skew"])
```
<img width="313" height="558" alt="image" src="https://github.com/user-attachments/assets/06e7f316-9b81-45d9-9ed4-750b6dce2e7b" />

```py
np.square(df["Highly Positive Skew"])
```
<img width="292" height="559" alt="image" src="https://github.com/user-attachments/assets/ab3a201d-2541-4b29-9c33-fb0bde56fd90" />

```py
df["Highly Positive Skew_boxcox"], parameters=stats.boxcox(df["Highly Positive Skew"])
df
```
<img width="1255" height="511" alt="image" src="https://github.com/user-attachments/assets/fdc44cd0-c4cd-4859-afdc-dee83e88359b" />

```py
df.skew()
```
<img width="397" height="291" alt="image" src="https://github.com/user-attachments/assets/3425a113-ce74-4442-8738-b89eb84ab940" />

```py
df["Highly Negative Skew_yeojohnson"],parameters=stats.yeojohnson(df["Highly Negative Skew"])
df.skew()
```
<img width="439" height="332" alt="image" src="https://github.com/user-attachments/assets/df8c674e-a3b6-4df3-8731-8534ac4fc476" />

```py
from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal')
df["Moderate Negative Skew_1"]=qt.fit_transform(df[["Moderate Negative Skew"]])
df
```
<img width="1660" height="545" alt="image" src="https://github.com/user-attachments/assets/ec255bc8-f05c-4dff-8cf9-b16d881aef3b" />

```py
import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt
sm.qqplot(df["Moderate Negative Skew"],line='45')
plt.show()
```
<img width="771" height="541" alt="image" src="https://github.com/user-attachments/assets/64f73b90-4e41-4b43-abfb-bf4e698cd07f" />

```py
sm.qqplot(np.reciprocal(df["Moderate Negative Skew"]),line='45')
plt.show()
```
<img width="774" height="538" alt="image" src="https://github.com/user-attachments/assets/b9cb1eb6-f76d-417d-ae30-dd43f2e478b9" />

```py
from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal',n_quantiles=891)
df["Moderate Negative Skew"]=qt.fit_transform(df[["Moderate Negative Skew"]])
sm.qqplot(df["Moderate Negative Skew"],line='45')
plt.show()
```
<img width="750" height="547" alt="image" src="https://github.com/user-attachments/assets/9bf6c516-f852-4905-93a6-fe95ab636d98" />

```py
df["Highly Negative Skew_1"]=qt.fit_transform(df[["Highly Negative Skew"]])
sm.qqplot(df["Highly Negative Skew"],line='45')
plt.show()
```
<img width="745" height="536" alt="image" src="https://github.com/user-attachments/assets/ad052e3b-bcd1-4027-8e95-3660ec4aad41" />

# RESULT:
Thus the given data, Feature Encoding, Transformation process and save the data to a file
was performed successfully.

       
