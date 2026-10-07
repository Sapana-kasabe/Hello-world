import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df=pd.read_csv('customer_segmentation.csv')
#print(df.isnull().sum())
df["Income"]=df['Income'].fillna(df['Income'].mean())


#print(df.describe())
#print(df['Education'].value_counts())
df["Dt_Customer"]=pd.to_datetime(df["Dt_Customer"],dayfirst=True)#changes datatype
print(df.info())#crossverify if the data type is changed or not
df["Age"]=2026-df['Year_Birth']#added age column as it is not there in the dataset
#print(df["Age"])
df["Total_childs"]=df["Kidhome"]+df["Teenhome"]#minimize table columns
#print(df["Total_childs"])
spend_cols=["MntWines","MntFruits","MntMeatProducts","MntFishProducts","MntSweetProducts","MntGoldProds"]
df["Total_spending"]=df[spend_cols].sum(axis=1)
#print(df["Total_spending"])
df["Customer_since"]=(pd.Timestamp("today")-df["Dt_Customer"]).dt.days
#print(df["Customer_since"])

#EDA
sns.histplot(df["Age"],bins=50,kde=True)
plt.title("Age Distribution")
plt.xlabel("Age")
plt.ylabel("count")
# plt.show()

sns.histplot(df["Income"],bins=30,kde=True)
plt.title("Income Distribution")
plt.xlabel("Income")
plt.ylabel("count")
# plt.show()

sns.histplot(df["Total_spending"],bins=30,kde=True)
plt.title("Total Spending Distribution")
# plt.show()
#relations with boxplot
sns.boxplot(x="Education",y="Income",data=df)
plt.title("Income by Education")
# plt.show()

corr=df[["Income","Recency","NumWebPurchases","NumCatalogPurchases","NumWebVisitsMonth","Total_spending"]].corr()
sns.heatmap(corr,annot=True,cmap="coolwarm")
plt.show()
plt.close()

#calculated averages
pivot_income=df.pivot_table(index="Education",values="Income",columns="Marital_Status",aggfunc="mean")
print(pivot_income)
# pivot_spending=df.pivot_table(index="Marital_Status",values="Total_spending",columns="Education",aggfunc="mean")
# print(pivot_spending)
sns.heatmap(pivot_income,annot=True,fmt=".0f",cmap="coolwarm")
plt.show()
plt.close()

#grouping to dig deeper
grp1=df.groupby("Education")["Total_spending"].mean().sort_values(ascending=False)
grp1.plot(kind="bar",color="skyblue")
plt.title("Average spending with education")
plt.ylabel("Average total spending")
plt.xticks(rotation=45)
plt.show()
plt.close()

# grp2=df.groupby("Education")["Mar "].mean().sort_values(ascending=False)
# grp2.plot(kind="bar",color="lightgreen")
# plt.title("Life stage of customers")
# plt.ylabel("Average Total Children")
# plt.xticks(rotation=45)
# plt.show()
# plt.close()

df["AcceptedAny"]=df[["AcceptedCmp1","AcceptedCmp2","AcceptedCmp3","AcceptedCmp4","AcceptedCmp5"]].sum(axis=1)
df["AcceptedAny"]=df["AcceptedAny"].apply(lambda x: 1 if x>0 else 0)
print(df["AcceptedAny"].value_counts())

grp2=df.groupby("Marital_Status")["AcceptedAny"].mean().sort_values(ascending=False)
grp2.plot(kind="bar",color="lightgreen")
plt.title("Acceptance rate by Marital Status")
plt.ylabel("Acceptance Rate")
plt.xticks(rotation=45)
plt.show()
plt.close()

bins=[18,30,40,50,60,70,90]
labels=["18-29","30-39","40-49","50-59","60-69","70+"]
df["Age_group"]=pd.cut(df["Age"],bins=bins,labels=labels) 
print(df["Age_group"].value_counts())

grp3=df.groupby("Age_group", observed=False)["Income"].mean()
grp3.plot(kind="bar",color="orange")
plt.title("Average income by Age group")
plt.ylabel("Income")
plt.xticks(rotation=45)
plt.show()
plt.close()
#group the features
features=df[["Income","Recency","NumWebPurchases","NumCatalogPurchases","NumWebVisitsMonth","Total_spending"]]
x=df[features.columns]


#scaling the data cuz have different ranges
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
x_scaled=scaler.fit_transform(x)
# print("X scaled\n",x_scaled)

#clustering
from sklearn.cluster import KMeans
# --- STEP 1: Cluster and Export Dataset ---
kmeans = KMeans(n_clusters=5, random_state=42)
df["Cluster"] = kmeans.fit_predict(x_scaled)

# Export labeled dataset to CSV
df.to_csv("segmented_customers.csv", index=False)

# Optional: PCA Visualization (kept from your code)
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
pca_data = pca.fit_transform(x_scaled)
df["PCA1"], df["PCA2"] = pca_data[:, 0], pca_data[:, 1]

sns.scatterplot(x="PCA1", y="PCA2", hue="Cluster", data=df, palette="Set1")
plt.title("Customer Segmentation using KMeans Clustering")
plt.show()
plt.close()


# --- STEPS 2 & 3: Train Classifier & Save Models ---
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

# Prepare training data (using raw feature vectors + assigned cluster labels)
X = df[features.columns]
y = df["Cluster"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit scaler on training set
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Train supervised classifier on feature-to-cluster mapping
classifier = RandomForestClassifier(n_estimators=100, random_state=42)
classifier.fit(X_train_scaled, y_train)

# Save classifier and scaler for Step 4 (Streamlit app)
joblib.dump(classifier, "cluster_classifier.pkl")
joblib.dump(scaler, "scaler.pkl")
print("Saved segmented dataset, classifier, and scaler successfully!")
