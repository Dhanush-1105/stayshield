"""
Outlier handling with the IQR rule  (from your outlier.py)

1. Find Q1, Q3 and IQR
2. Fences = Q1 - 1.5*IQR  and  Q3 + 1.5*IQR
3. Anything outside the fences is an outlier
4. Clip (cap) outliers to the fences, then Min-Max scale
"""
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

from data_loader import load_data
from utils import banner, finish_plot

df = load_data()
feature = "adr"   # average daily rate - has a long tail of very high prices

banner(f"Original statistics of '{feature}'")
print(df[feature].describe())

Q1 = df[feature].quantile(0.25)
Q3 = df[feature].quantile(0.75)
IQR = Q3 - Q1
lower_fence = Q1 - 1.5 * IQR
upper_fence = Q3 + 1.5 * IQR

print("\nQ1 =", Q1)
print("Q3 =", Q3)
print("IQR =", IQR)
print("Lower Fence =", lower_fence)
print("Upper Fence =", upper_fence)

outliers = df[(df[feature] < lower_fence) | (df[feature] > upper_fence)]
print("\nNumber of outliers:", len(outliers))
print(f"Share of data: {len(outliers) / len(df):.2%}")

df[f"{feature}_Clipped"] = df[feature].clip(lower=lower_fence, upper=upper_fence)

print("\nMinimum BEFORE clipping:", df[feature].min())
print("Maximum BEFORE clipping:", df[feature].max())
print("Minimum AFTER clipping:", df[f"{feature}_Clipped"].min())
print("Maximum AFTER clipping:", df[f"{feature}_Clipped"].max())

scaler = MinMaxScaler()
df[f"{feature}_MinMax"] = scaler.fit_transform(df[[f"{feature}_Clipped"]])

banner("Min-Max scaled data")
print(df[[feature, f"{feature}_Clipped", f"{feature}_MinMax"]].head())

banner("Statistics after Min-Max scaling")
print(df[f"{feature}_MinMax"].describe())

# Box plots: before vs after clipping
fig, axes = plt.subplots(1, 2, figsize=(9, 4))
axes[0].boxplot(df[feature], vert=True)
axes[0].set_title(f"{feature} - before clipping")
axes[1].boxplot(df[f"{feature}_Clipped"], vert=True)
axes[1].set_title(f"{feature} - after clipping")
finish_plot("outlier_boxplots")
