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
<img width="627" height="434" alt="image" src="https://github.com/user-attachments/assets/e24422b3-ee5c-4b5a-b32f-09a8963cb59e" />

```py
from sklearn.preprocessing import LabelEncoder,OrdinalEncoder
pm=['Hot','Warm','Cold']
e1=OrdinalEncoder(categories=[pm])
e1.fit_transform(df[["ord_2"]])
```
<img width="357" height="224" alt="image" src="https://github.com/user-attachments/assets/6d9a20d8-b4c6-4152-bf23-6f840261cbec" />

```py
df['bo2']=e1.fit_transform(df[["ord_2"]])
df
```
<img width="608" height="430" alt="image" src="https://github.com/user-attachments/assets/4e74f0de-5e54-408f-bd2a-35ed5134e218" />

```py
le=LabelEncoder()
dfc=df.copy()
dfc['ord_2']=le.fit_transform(dfc['ord_2'])
dfc
```
<img width="489" height="421" alt="image" src="https://github.com/user-attachments/assets/568f6341-11aa-436b-8398-c8e02f7eb60b" />

```py
from sklearn.preprocessing import OneHotEncoder
ohe=OneHotEncoder(sparse_output=False)
df2=df.copy()
enc=pd.DataFrame(ohe.fit_transform(df2[["nom_0"]]))
df2=pd.concat([df2,enc],axis=1)
df2
```
<img width="586" height="439" alt="image" src="https://github.com/user-attachments/assets/dfb39ef6-6710-42c1-ad45-3fea72161229" />

```py
pd.get_dummies(df2,columns=["nom_0"])
```
<img width="850" height="443" alt="image" src="https://github.com/user-attachments/assets/d6a01e5e-2966-4cc6-9506-c4d0194b31a1" />

```py
from category_encoders import BinaryEncoder
df=pd.read_csv("data.csv")
df
```
<img width="622" height="427" alt="image" src="https://github.com/user-attachments/assets/851bf516-a46d-4c91-a371-be376972aa72" />

```py
be=BinaryEncoder()
nd=be.fit_transform(df['Ord_2'])
df
```
<img width="629" height="442" alt="image" src="https://github.com/user-attachments/assets/ab81aa0f-b7eb-42b2-bd4d-6a5e42c349a4" />

```py
dfb=pd.concat([df,nd],axis=1)
dfb
```
<img width="882" height="448" alt="image" src="https://github.com/user-attachments/assets/7a909901-c36f-4448-a50d-40bdce058a38" />

```py
from category_encoders import TargetEncoder
te=TargetEncoder()
CC=df.copy()
new=te.fit_transform(X=CC["City"],y=CC["Target"])
CC=pd.concat([CC,new],axis=1)
CC
```
<img width="732" height="434" alt="image" src="https://github.com/user-attachments/assets/0f2c4204-3fb6-4e57-a26c-8a084732e1b0" />

```py
import pandas as pd
from scipy import stats
import numpy as np
df=pd.read_csv("Data_to_Transform.csv")
df
```
<img width="1013" height="504" alt="image" src="https://github.com/user-attachments/assets/f90a5021-1675-493e-b1cb-9321656be392" />

```py
df.skew()
```
<img width="359" height="247" alt="image" src="https://github.com/user-attachments/assets/8f163a80-8c5e-4425-a5d2-0042ea135c6b" />

```py
np.log(df["Highly Positive Skew"])
```
<img width="357" height="561" alt="image" src="https://github.com/user-attachments/assets/d2876d57-c7f2-488e-bfcc-dfad3e53c5e0" />

```py
np.reciprocal(df["Moderate Positive Skew"])
```
<img width="338" height="513" alt="image" src="https://github.com/user-attachments/assets/88f356b6-c5c9-4dc1-8849-13ac5b2e3112" />

```py
np.sqrt(df["Highly Positive Skew"])
```
<img width="324" height="566" alt="image" src="https://github.com/user-attachments/assets/00867d87-a320-4562-87f8-2f888eab991f" />

```py
np.square(df["Highly Positive Skew"])
```
<img width="366" height="560" alt="image" src="https://github.com/user-attachments/assets/01c33dc5-c9bc-43a7-bff3-e28cbaa9eb13" />

```py
df["Highly Positive Skew_boxcox"], parameters=stats.boxcox(df["Highly Positive Skew"])
df
```
<img width="1296" height="519" alt="image" src="https://github.com/user-attachments/assets/21b6ef28-5993-4384-9fd9-78373f5ac421" />

```py
df.skew()
```
<img width="392" height="268" alt="image" src="https://github.com/user-attachments/assets/40bb22bc-1f31-4718-af4b-26a524e24312" />

```py
df["Highly Negative Skew_yeojohnson"],parameters=stats.yeojohnson(df["Highly Negative Skew"])
df.skew()
```
<img width="422" height="328" alt="image" src="https://github.com/user-attachments/assets/63f6f243-d1a0-4c4e-b739-50065f5eb989" />

```py
from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal')
df["Moderate Negative Skew_1"]=qt.fit_transform(df[["Moderate Negative Skew"]])
df
```
<img width="1353" height="539" alt="image" src="https://github.com/user-attachments/assets/048a6353-5ad4-409d-bbb1-a5af6437023b" />

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

       
