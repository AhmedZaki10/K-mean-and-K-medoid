#!/usr/bin/env python
# coding: utf-8

# In[1]:


# first we will see if the data heve missing values, incorrect data type, duplicates 
import pandas as pd

# Read the data set
df = pd.read_csv('C:\\Users\\lenovo\\OneDrive\\Desktop\\train.csv')

# Check for missing values
if df.isnull().values.any():
    print('There are missing values in the data set.')

# Check for incorrect data types
for column in df.columns:
    if df[column].dtype != df[column].dtypes.type:
        print('The data type for column {} is incorrect.'.format(column))

# Check for duplicate values
if df.duplicated().values.any():
    print('There are duplicate values in the data set.')


# In[2]:


# we will drop the duplicates
import pandas as pd

# Read the data set
df = pd.read_csv('C:\\Users\\lenovo\\OneDrive\\Desktop\\train.csv')

# Remove duplicate rows
df = df.drop_duplicates()

# Print the data set
print(df)


# In[3]:


# we will import the libiraries we will use
import pandas as pd 
from sklearn.cluster import KMeans
from sklearn_extra.cluster import KMedoids
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


# In[7]:


#k-mean algorithm

df = pd.read_csv("C:\\Users\\lenovo\\OneDrive\\Desktop\\train.csv")
pd.set_option('display.max_rows',0)


#plotting crimes using coordinates to get an idea of the distribution
pdf = df[['X','Y']]
fig = plt.figure()
plt.title('SF Crime Distribution')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.scatter(pdf.X, pdf.Y, s = 2, c = 'r')
plt.show()

#Finding how many outliers 
outlier90 = df[df["Y"] == 90]
print(outlier90.count())


#Catching the outlier crimes and removing it 
df_f = df.query('Y != 90')


#Plotting the actual distribution without the outliers
pdf = df_f[['X','Y']]
fig = plt.figure()
plt.title('SF Crime Distribution')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.scatter(pdf.X, pdf.Y, s = 2, c = 'r')
plt.show()




#Data is too big so we take part of the data to reduce computation time 
df_f = df_f.head(10000)





#Combining Coordinates to make them a single feature
df_coor = pd.DataFrame({'Longitude': df_f["X"], 'Latitude': df_f["Y"]})



#Find optimal number of clusters
silhouette_scores = []
k_values = range(2, 11)

for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(df_coor)
    silhouette_scores.append(silhouette_score(df_coor, kmeans.labels_))




#Plotting results from silhouette score testing    
plt.plot(k_values, silhouette_scores, marker='o')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Score for Kmeans')
plt.show()


# Fit a KMeans model with 3 clusters
model = KMeans(n_clusters=3).fit(df_coor)

# add the cluster labels to the original DataFrame
df_clusters = model.labels_
df_f['cluster_label'] = df_clusters

# convert the cluster labels to a pandas DataFrame
cluster_df = pd.DataFrame(df_clusters)
print(df_f.head(10))

# Calculate the silhouette score
silhouette_avg = silhouette_score(df_coor, model.labels_)


centers = model.cluster_centers_
print("Kmeans cluster centers:",centers)
long_centers = [i for i, _ in centers]
lat_centers = [i for _, i in centers]


# Plotting the cluster labels

plt.scatter(df_f["X"], df_f["Y"], c=df_f["cluster_label"])
plt.scatter(long_centers,lat_centers, s=50, c='r', marker='+')
plt.xlabel("X Coordinate")
plt.ylabel("Y Coordinate")
plt.title("San Francisco using k-means")


print("Silhouette Score using Kmeans:", silhouette_avg)


# In[5]:


# k-medoid algorithm

# Load the dataset
df = pd.read_csv("C:\\Users\\lenovo\\OneDrive\\Desktop\\train.csv")
pd.set_option('display.max_rows', 0)


# Catching the outlier crimes and removing them
df_f = df.query('Y != 90')


# Data is too big, so we take a part of the data to reduce computation time
df_f = df_f.head(10000)

# Combining Coordinates to make them a single feature
df_coor = pd.DataFrame({'Longitude': df_f["X"], 'Latitude': df_f["Y"]})

# Find optimal number of clusters
silhouette_scores = []
k_values = range(2, 11)

for k in k_values:
    kmedoids = KMedoids(n_clusters=k, random_state=42)
    kmedoids.fit(df_coor)
    silhouette_scores.append(silhouette_score(df_coor, kmedoids.labels_))

# Plotting results from silhouette score testing
plt.plot(k_values, silhouette_scores, marker='o')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Score for Kmedoids')
plt.show()

# Fit a KMedoids model with 3 clusters
model = KMedoids(n_clusters=3, random_state=42).fit(df_coor)

# Add the cluster labels to the original DataFrame
df_f['cluster_label'] = model.labels_

#Showing cluster centers
centers = model.cluster_centers_
print("K-medoids cluster centers:",centers)

# Plotting the cluster labels
plt.scatter(df_f["X"], df_f["Y"], c=df_f["cluster_label"])
plt.scatter(model.cluster_centers_[:, 0], model.cluster_centers_[:, 1], s=50, c='r', marker='+')
plt.xlabel("X Coordinate")
plt.ylabel("Y Coordinate")
plt.title("San Francisco Crime Clusters using Kmedoids")
plt.show()

# Calculate the silhouette score
silhouette_avg = silhouette_score(df_coor, model.labels_)
print("Silhouette Score using Kmedoids:", silhouette_avg)


# In[ ]:




