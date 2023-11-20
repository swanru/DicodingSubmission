#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from openpyxl import load_workbook


# In[2]:


Aotizhongxin = pd.read_csv("PRSA_Data_Aotizhongxin_20130301-20170228.csv")
Changping = pd.read_csv("PRSA_Data_Changping_20130301-20170228.csv")
Dingling = pd.read_csv("PRSA_Data_Dingling_20130301-20170228.csv")
Dongsi = pd.read_csv("PRSA_Data_Dongsi_20130301-20170228.csv")
Guanyuan = pd.read_csv("PRSA_Data_Guanyuan_20130301-20170228.csv")
Gucheng = pd.read_csv("PRSA_Data_Gucheng_20130301-20170228.csv")
Nongzhanguan = pd.read_csv("PRSA_Data_Nongzhanguan_20130301-20170228.csv")
Shunyi = pd.read_csv("PRSA_Data_Shunyi_20130301-20170228.csv")
Tiantan = pd.read_csv("PRSA_Data_Tiantan_20130301-20170228.csv")
Wanliu = pd.read_csv("PRSA_Data_Wanliu_20130301-20170228.csv")
Wanshouxigong = pd.read_csv("PRSA_Data_Wanshouxigong_20130301-20170228.csv")


# In[3]:


Aotizhongxin = Aotizhongxin[['month','day','year','hour','PM2.5','PM10','SO2','NO2','CO','O3','TEMP','PRES','DEWP','RAIN','wd','WSPM','station']]
Changping = Changping[['month','day','year','hour','PM2.5','PM10','SO2','NO2','CO','O3','TEMP','PRES','DEWP','RAIN','wd','WSPM','station']]
Dingling = Dingling[['month','day','year','hour','PM2.5','PM10','SO2','NO2','CO','O3','TEMP','PRES','DEWP','RAIN','wd','WSPM','station']]
Dongsi = Dongsi[['month','day','year','hour','PM2.5','PM10','SO2','NO2','CO','O3','TEMP','PRES','DEWP','RAIN','wd','WSPM','station']]
Guanyuan = Guanyuan[['month','day','year','hour','PM2.5','PM10','SO2','NO2','CO','O3','TEMP',"PRES","DEWP","RAIN","wd","WSPM","station"]]
Gucheng = Gucheng[['month','day','year','hour','PM2.5','PM10','SO2','NO2','CO','O3','TEMP',"PRES","DEWP","RAIN","wd","WSPM","station"]]
Nongzhanguan = Nongzhanguan[['month','day','year','hour','PM2.5','PM10','SO2','NO2','CO','O3','TEMP',"PRES","DEWP","RAIN","wd","WSPM","station"]]
Shunyi = Shunyi[['month','day','year','hour','PM2.5','PM10','SO2','NO2','CO','O3','TEMP',"PRES","DEWP","RAIN","wd","WSPM","station"]]
Tiantan = Tiantan[['month','day','year','hour','PM2.5','PM10','SO2','NO2','CO','O3','TEMP',"PRES","DEWP","RAIN","wd","WSPM","station"]]
Wanliu = Wanliu[['month','day','year','hour','PM2.5','PM10','SO2','NO2','CO','O3','TEMP',"PRES","DEWP","RAIN","wd","WSPM","station"]]
Wanshouxigong = Wanshouxigong[['month','day','year','hour','PM2.5','PM10','SO2','NO2','CO','O3','TEMP',"PRES","DEWP","RAIN","wd","WSPM","station"]]


# # Pertanyaan
# ## 1. Bagaimana Kualitas Udara pada setiap kota dalam kurun waktu tertentu ?
# ## 2. Nilai Polutan paling tinggi berdasarkan rata-rata seluruh waktu terdapat pada kota?
# ## 3. Kapan kualitas udara memiliki kualitas tidak baik dalam kurun waktu pada setiap kota?

# ### Dataset Air Quality
# ##### Data Wrangling
# ##### Exploratory Data Analysis
# ##### Data Visualization
# ##### Pengembangan Dashboard

# # Data Wrangling 
# ### 1. Gathering Data

# In[4]:


Aotizhongxin.head()


# In[5]:


Changping.head()  


# In[6]:


Dingling.head()


# In[7]:


Dongsi.head()


# In[8]:


Guanyuan.head()  


# In[9]:


Gucheng.head()


# In[10]:


Nongzhanguan.head()  


# In[11]:


Shunyi.head()


# In[12]:


Tiantan.head()  


# In[13]:


Wanliu.head() 


# In[14]:


Wanshouxigong.head()


# ### 2.Assessing Data

# In[15]:


data = [Aotizhongxin,Changping,Dingling,Dongsi,Guanyuan,Gucheng,Nongzhanguan,Shunyi,Tiantan,Wanliu,Wanshouxigong]
dataset = pd.concat(data)
dataset.info()


# #### Pengecekan Missing Value

# In[16]:


dataset.isna().sum()


# ### 3. Cleaning Data(Dropping Method)

# In[17]:


dataset.dropna(axis=0,inplace=True)


# #### After Cleaning Data with Dropping Method

# In[18]:


dataset.isna().sum()


# # Exploratory Data Analysis

# In[19]:


describe_dataset = dataset[['PM2.5','PM10','SO2','NO2','CO','O3','TEMP','PRES','DEWP','RAIN','wd','WSPM']].describe()
describe_dataset


# #### Korelasi antar data pada Dataset Polusi meggunakan heatmap

# In[20]:


Pollution_dataset = dataset[['PM2.5','PM10','SO2','NO2','CO','O3']]
plt.figure(figsize=(10,7))
sns.heatmap(Pollution_dataset.corr(),annot=True)


# ### Data Polusi tiap kota
# #### Data yang digunakan yaitu PM2.5,PM10,SO2,NO2,CO,O3

# In[21]:


Pollution_Aotizhongxin = Aotizhongxin[['month','day','year','hour','PM2.5','PM10','SO2','NO2','CO','O3']]
Pollution_Changping = Changping[['month','day','year','hour','PM2.5','PM10','SO2','NO2','CO','O3']]
Pollution_Dingling = Dingling[['month','day','year','hour','PM2.5','PM10','SO2','NO2','CO','O3']]
Pollution_Dongsi = Dongsi[['month','day','year','hour','PM2.5','PM10','SO2','NO2','CO','O3']] 
Pollution_Guanyuan = Guanyuan[['month','day','year','hour','PM2.5','PM10','SO2','NO2','CO','O3']] 
Pollution_Gucheng = Gucheng[['month','day','year','hour','PM2.5','PM10','SO2','NO2','CO','O3']] 
Pollution_Nongzhanguan = Nongzhanguan[['month','day','year','hour','PM2.5','PM10','SO2','NO2','CO','O3']] 
Pollution_Shunyi = Shunyi[['month','day','year','hour','PM2.5','PM10','SO2','NO2','CO','O3']] 
Pollution_Tiantan = Tiantan[['month','day','year','hour','PM2.5','PM10','SO2','NO2','CO','O3']] 
Pollution_Wanliu = Wanliu[['month','day','year','hour','PM2.5','PM10','SO2','NO2','CO','O3']] 
Pollution_Wanshouxigong = Wanshouxigong[['month','day','year','hour','PM2.5','PM10','SO2','NO2','CO','O3']] 


# #### Merubah format tanggal 

# In[22]:


Pollution_Aotizhongxin ['date'] = pd.to_datetime(Pollution_Aotizhongxin [['year','month','day']])
Pollution_Changping ['date'] = pd.to_datetime(Pollution_Changping [['year','month','day']])
Pollution_Dingling ['date'] = pd.to_datetime(Pollution_Dingling [['year','month','day']])
Pollution_Dongsi ['date'] = pd.to_datetime(Pollution_Dongsi [['year','month','day']])
Pollution_Guanyuan ['date'] = pd.to_datetime(Pollution_Guanyuan [['year','month','day']])
Pollution_Gucheng ['date'] = pd.to_datetime(Pollution_Gucheng [['year','month','day']])
Pollution_Nongzhanguan ['date'] = pd.to_datetime(Pollution_Nongzhanguan [['year','month','day']])
Pollution_Shunyi ['date'] = pd.to_datetime(Pollution_Shunyi [['year','month','day']])
Pollution_Tiantan ['date'] = pd.to_datetime(Pollution_Tiantan [['year','month','day']])
Pollution_Wanliu['date'] = pd.to_datetime(Pollution_Wanliu[['year','month','day']])
Pollution_Wanshouxigong ['date'] = pd.to_datetime(Pollution_Wanshouxigong [['year','month','day']])


# ### Keadaan Kualitas Udara pada setiap Kota

# In[23]:


dataset_station = dataset.groupby(by='station').agg({
    'PM2.5':'mean',
    'PM10':'mean',
    'SO2':'mean',
    'NO2':'mean',
    'O3':'mean',
})
plt.figure(figsize=(15, 8))
for column in dataset_station.columns:
    plt.plot(dataset_station.index, dataset_station[column], label=column)
plt.title('Tiap Kota')
plt.xticks(rotation=45)
plt.legend()
plt.show()
data = pd.DataFrame(dataset_station)
data


# In[24]:


dataset_station = dataset.groupby(by='station').agg({
    'CO':'mean',
})
plt.figure(figsize=(15, 8))
for column in dataset_station.columns:
    plt.plot(dataset_station.index, dataset_station[column], label=column)
plt.title('Tiap Kota')
plt.xticks(rotation=45)
plt.legend()
plt.show()
print(dataset_station)


# In[25]:


High_PM25 = pd.DataFrame(data["PM2.5"].sort_values(ascending=False).reset_index())
High_PM10 = pd.DataFrame(data["PM10"].sort_values(ascending=False).reset_index()) 
High_SO2  = pd.DataFrame(data["SO2"].sort_values(ascending=False).reset_index())
High_NO2  = pd.DataFrame(data["NO2"].sort_values(ascending=False).reset_index())
High_O3   = pd.DataFrame(data["O3"].sort_values(ascending=False).reset_index())
High_CO = pd.DataFrame(dataset_station["CO"].sort_values(ascending=False).reset_index())


# In[26]:


Hasil = pd.DataFrame({"Data Tertinggi": ["PM 2.5", "PM 10",
                                        "SO2","NO2",
                                        "O3","CO"], 
                      "Kota"          :[High_PM25.station[0], High_PM10.station[0],
                                        High_SO2.station[0], High_NO2.station[0],
                                        High_O3.station[0], High_CO.station[0]],
                      "Nilai"         :[High_PM25["PM2.5"][0], High_PM10.PM10[0],
                                        High_SO2.SO2[0], High_NO2.NO2[0],
                                        High_O3.O3[0], High_CO.CO[0]]})
Hasil.to_numpy()
Hasil


# ## Kualitas Udara pada kurun waktu Tahunan, Bulanan, Harian, Jam

# ## Dataset

# In[27]:


dataset_year = dataset.groupby(by='year').agg({
    'PM10':'mean',
    'PM2.5':'mean',
    'O3':'mean',
    'NO2':'mean',
    'SO2':'mean',
})
dataset_month = dataset.groupby(by='month').agg({
    'PM10':'mean',
    'PM2.5':'mean',
    'O3':'mean',
    'NO2':'mean',
    'SO2':'mean',
})
dataset_day = dataset.groupby(by='day').agg({
    'PM10':'mean',
    'PM2.5':'mean',
    'O3':'mean',
    'NO2':'mean',
    'SO2':'mean',
})
dataset_hour = dataset.groupby(by='hour').agg({
    'PM10':'mean',
    'PM2.5':'mean',
    'O3':'mean',
    'NO2':'mean',
    'SO2':'mean',
})


# In[28]:


fig, (ax1, ax2) = plt.subplots(1, 2,figsize=(15, 8))
for column in dataset_year.columns:
    ax1.plot(dataset_year.index, dataset_year[column], label=column)
ax1.set_xlabel("Tahunan")
ax1.legend()
for column in dataset_month.columns:
    ax2.plot(dataset_month.index, dataset_month[column], label=column)
ax2.set_xlabel("Bulanan")
ax2.legend()
fig, (ax3, ax4) = plt.subplots(1, 2,figsize=(15, 8))
for column in dataset_day.columns:
    ax3.plot(dataset_day.index, dataset_day[column], label=column)
ax3.set_xlabel("30 Hari")
ax1.legend()
for column in dataset_hour.columns:
    ax4.plot(dataset_hour.index, dataset_hour[column], label=column)
ax4.set_xlabel("Setiap Jam")
ax2.legend()


# ### Pollution Aotizhongxin

# In[29]:


dataset_year = Pollution_Aotizhongxin.groupby(by='year').agg({
    'PM10':'mean',
    'PM2.5':'mean',
    'O3':'mean',
    'NO2':'mean',
    'SO2':'mean',
})
dataset_month = Pollution_Aotizhongxin.groupby(by='month').agg({
    'PM10':'mean',
    'PM2.5':'mean',
    'O3':'mean',
    'NO2':'mean',
    'SO2':'mean',
})
dataset_day = Pollution_Aotizhongxin.groupby(by='day').agg({
    'PM10':'mean',
    'PM2.5':'mean',
    'O3':'mean',
    'NO2':'mean',
    'SO2':'mean',
})
dataset_hour = Pollution_Aotizhongxin.groupby(by='hour').agg({
    'PM10':'mean',
    'PM2.5':'mean',
    'O3':'mean',
    'NO2':'mean',
    'SO2':'mean',
})


# In[30]:


fig, (ax1, ax2) = plt.subplots(1, 2,figsize=(15, 8))
for column in dataset_year.columns:
    ax1.plot(dataset_year.index, dataset_year[column], label=column)
ax1.set_xlabel("Tahunan")
ax1.legend()
for column in dataset_month.columns:
    ax2.plot(dataset_month.index, dataset_month[column], label=column)
ax2.set_xlabel("Bulanan")
ax2.legend()
fig, (ax3, ax4) = plt.subplots(1, 2,figsize=(15, 8))
for column in dataset_day.columns:
    ax3.plot(dataset_day.index, dataset_day[column], label=column)
ax3.set_xlabel("30 Hari")
ax1.legend()
for column in dataset_hour.columns:
    ax4.plot(dataset_hour.index, dataset_hour[column], label=column)
ax4.set_xlabel("Setiap Jam")
ax2.legend()


# ## Pollution Changping

# In[31]:


dataset_year = Pollution_Changping.groupby(by='year').agg({
    'PM10':'mean',
    'PM2.5':'mean',
    'O3':'mean',
    'NO2':'mean',
    'SO2':'mean',
})
dataset_month = Pollution_Changping.groupby(by='month').agg({
    'PM10':'mean',
    'PM2.5':'mean',
    'O3':'mean',
    'NO2':'mean',
    'SO2':'mean',
})
dataset_day = Pollution_Changping.groupby(by='day').agg({
    'PM10':'mean',
    'PM2.5':'mean',
    'O3':'mean',
    'NO2':'mean',
    'SO2':'mean',
})
dataset_hour = Pollution_Changping.groupby(by='hour').agg({
    'PM10':'mean',
    'PM2.5':'mean',
    'O3':'mean',
    'NO2':'mean',
    'SO2':'mean',
})
fig, (ax1, ax2) = plt.subplots(1, 2,figsize=(15, 8))
for column in dataset_year.columns:
    ax1.plot(dataset_year.index, dataset_year[column], label=column)
ax1.set_xlabel("Tahunan")
ax1.legend()
for column in dataset_month.columns:
    ax2.plot(dataset_month.index, dataset_month[column], label=column)
ax2.set_xlabel("Bulanan")
ax2.legend()
fig, (ax3, ax4) = plt.subplots(1, 2,figsize=(15, 8))
for column in dataset_day.columns:
    ax3.plot(dataset_day.index, dataset_day[column], label=column)
ax3.set_xlabel("30 Hari")
ax3.legend()
for column in dataset_hour.columns:
    ax4.plot(dataset_hour.index, dataset_hour[column], label=column)
ax4.set_xlabel("Setiap Jam")
ax4.legend()


# In[ ]:





# ## Pollution Dingling

# In[32]:


dataset_year = Pollution_Dingling.groupby(by='year').agg({
    'PM10':'mean',
    'PM2.5':'mean',
    'O3':'mean',
    'NO2':'mean',
    'SO2':'mean',
})
dataset_month = Pollution_Dingling.groupby(by='month').agg({
    'PM10':'mean',
    'PM2.5':'mean',
    'O3':'mean',
    'NO2':'mean',
    'SO2':'mean',
})
dataset_day = Pollution_Dingling.groupby(by='day').agg({
    'PM10':'mean',
    'PM2.5':'mean',
    'O3':'mean',
    'NO2':'mean',
    'SO2':'mean',
})
dataset_hour = Pollution_Dingling.groupby(by='hour').agg({
    'PM10':'mean',
    'PM2.5':'mean',
    'O3':'mean',
    'NO2':'mean',
    'SO2':'mean',
})
fig, (ax1, ax2) = plt.subplots(1, 2,figsize=(15, 8))
for column in dataset_year.columns:
    ax1.plot(dataset_year.index, dataset_year[column], label=column)
ax1.set_xlabel("Tahunan")
ax1.legend()
for column in dataset_month.columns:
    ax2.plot(dataset_month.index, dataset_month[column], label=column)
ax2.set_xlabel("Bulanan")
ax2.legend()
fig, (ax3, ax4) = plt.subplots(1, 2,figsize=(15, 8))
for column in dataset_day.columns:
    ax3.plot(dataset_day.index, dataset_day[column], label=column)
ax3.set_xlabel("30 Hari")
ax3.legend()
for column in dataset_hour.columns:
    ax4.plot(dataset_hour.index, dataset_hour[column], label=column)
ax4.set_xlabel("Setiap Jam")
ax4.legend()


# ## Pollution Dongsi

# In[33]:


dataset_year = Pollution_Dongsi.groupby(by='year').agg({
    'PM10':'mean',
    'PM2.5':'mean',
    'O3':'mean',
    'NO2':'mean',
    'SO2':'mean',
})
dataset_month = Pollution_Dongsi.groupby(by='month').agg({
    'PM10':'mean',
    'PM2.5':'mean',
    'O3':'mean',
    'NO2':'mean',
    'SO2':'mean',
})
dataset_day = Pollution_Dongsi.groupby(by='day').agg({
    'PM10':'mean',
    'PM2.5':'mean',
    'O3':'mean',
    'NO2':'mean',
    'SO2':'mean',
})
dataset_hour = Pollution_Dongsi.groupby(by='hour').agg({
    'PM10':'mean',
    'PM2.5':'mean',
    'O3':'mean',
    'NO2':'mean',
    'SO2':'mean',
})
fig, (ax1, ax2) = plt.subplots(1, 2,figsize=(15, 8))
for column in dataset_year.columns:
    ax1.plot(dataset_year.index, dataset_year[column], label=column)
ax1.set_xlabel("Tahunan")
ax1.legend()
for column in dataset_month.columns:
    ax2.plot(dataset_month.index, dataset_month[column], label=column)
ax2.set_xlabel("Bulanan")
ax2.legend()
fig, (ax3, ax4) = plt.subplots(1, 2,figsize=(15, 8))
for column in dataset_day.columns:
    ax3.plot(dataset_day.index, dataset_day[column], label=column)
ax3.set_xlabel("30 Hari")
ax3.legend()
for column in dataset_hour.columns:
    ax4.plot(dataset_hour.index, dataset_hour[column], label=column)
ax4.set_xlabel("Setiap Jam")
ax4.legend()


# ## Pollution Guanyuan

# In[34]:


dataset_year = Pollution_Guanyuan.groupby(by='year').agg({
    'PM10':'mean',
    'PM2.5':'mean',
    'O3':'mean',
    'NO2':'mean',
    'SO2':'mean',
})
dataset_month = Pollution_Guanyuan.groupby(by='month').agg({
    'PM10':'mean',
    'PM2.5':'mean',
    'O3':'mean',
    'NO2':'mean',
    'SO2':'mean',
})
dataset_day = Pollution_Guanyuan.groupby(by='day').agg({
    'PM10':'mean',
    'PM2.5':'mean',
    'O3':'mean',
    'NO2':'mean',
    'SO2':'mean',
})
dataset_hour = Pollution_Guanyuan.groupby(by='hour').agg({
    'PM10':'mean',
    'PM2.5':'mean',
    'O3':'mean',
    'NO2':'mean',
    'SO2':'mean',
})
fig, (ax1, ax2) = plt.subplots(1, 2,figsize=(15, 8))
for column in dataset_year.columns:
    ax1.plot(dataset_year.index, dataset_year[column], label=column)
ax1.set_xlabel("Tahunan")
ax1.legend()
for column in dataset_month.columns:
    ax2.plot(dataset_month.index, dataset_month[column], label=column)
ax2.set_xlabel("Bulanan")
ax2.legend()
fig, (ax3, ax4) = plt.subplots(1, 2,figsize=(15, 8))
for column in dataset_day.columns:
    ax3.plot(dataset_day.index, dataset_day[column], label=column)
ax3.set_xlabel("30 Hari")
ax3.legend()
for column in dataset_hour.columns:
    ax4.plot(dataset_hour.index, dataset_hour[column], label=column)
ax4.set_xlabel("Setiap Jam")
ax4.legend()


# ## Pollution Gucheng

# In[35]:


dataset_year = Pollution_Gucheng.groupby(by='year').agg({
    'PM10':'mean',
    'PM2.5':'mean',
    'O3':'mean',
    'NO2':'mean',
    'SO2':'mean',
})
dataset_month = Pollution_Gucheng.groupby(by='month').agg({
    'PM10':'mean',
    'PM2.5':'mean',
    'O3':'mean',
    'NO2':'mean',
    'SO2':'mean',
})
dataset_day = Pollution_Gucheng.groupby(by='day').agg({
    'PM10':'mean',
    'PM2.5':'mean',
    'O3':'mean',
    'NO2':'mean',
    'SO2':'mean',
})
dataset_hour = Pollution_Gucheng.groupby(by='hour').agg({
    'PM10':'mean',
    'PM2.5':'mean',
    'O3':'mean',
    'NO2':'mean',
    'SO2':'mean',
})
fig, (ax1, ax2) = plt.subplots(1, 2,figsize=(15, 8))
for column in dataset_year.columns:
    ax1.plot(dataset_year.index, dataset_year[column], label=column)
ax1.set_xlabel("Tahunan")
ax1.legend()
for column in dataset_month.columns:
    ax2.plot(dataset_month.index, dataset_month[column], label=column)
ax2.set_xlabel("Bulanan")
ax2.legend()
fig, (ax3, ax4) = plt.subplots(1, 2,figsize=(15, 8))
for column in dataset_day.columns:
    ax3.plot(dataset_day.index, dataset_day[column], label=column)
ax3.set_xlabel("30 Hari")
ax1.legend()
for column in dataset_hour.columns:
    ax4.plot(dataset_hour.index, dataset_hour[column], label=column)
ax4.set_xlabel("Setiap Jam")
ax2.legend()


# ## Pollution Nongzhanguan

# In[36]:


dataset_year = Pollution_Nongzhanguan.groupby(by='year').agg({
    'PM10':'mean',
    'PM2.5':'mean',
    'O3':'mean',
    'NO2':'mean',
    'SO2':'mean',
})
dataset_month = Pollution_Nongzhanguan.groupby(by='month').agg({
    'PM10':'mean',
    'PM2.5':'mean',
    'O3':'mean',
    'NO2':'mean',
    'SO2':'mean',
})
dataset_day = Pollution_Nongzhanguan.groupby(by='day').agg({
    'PM10':'mean',
    'PM2.5':'mean',
    'O3':'mean',
    'NO2':'mean',
    'SO2':'mean',
})
dataset_hour = Pollution_Nongzhanguan.groupby(by='hour').agg({
    'PM10':'mean',
    'PM2.5':'mean',
    'O3':'mean',
    'NO2':'mean',
    'SO2':'mean',
})
fig, (ax1, ax2) = plt.subplots(1, 2,figsize=(15, 8))
for column in dataset_year.columns:
    ax1.plot(dataset_year.index, dataset_year[column], label=column)
ax1.set_xlabel("Tahunan")
ax1.legend()
for column in dataset_month.columns:
    ax2.plot(dataset_month.index, dataset_month[column], label=column)
ax2.set_xlabel("Bulanan")
ax2.legend()
fig, (ax3, ax4) = plt.subplots(1, 2,figsize=(15, 8))
for column in dataset_day.columns:
    ax3.plot(dataset_day.index, dataset_day[column], label=column)
ax3.set_xlabel("30 Hari")
ax3.legend()
for column in dataset_hour.columns:
    ax4.plot(dataset_hour.index, dataset_hour[column], label=column)
ax4.set_xlabel("Setiap Jam")
ax4.legend()


# ## Pollution Shunyi

# In[37]:


dataset_year = Pollution_Shunyi.groupby(by='year').agg({
    'PM10':'mean',
    'PM2.5':'mean',
    'O3':'mean',
    'NO2':'mean',
    'SO2':'mean',
})
dataset_month = Pollution_Shunyi.groupby(by='month').agg({
    'PM10':'mean',
    'PM2.5':'mean',
    'O3':'mean',
    'NO2':'mean',
    'SO2':'mean',
})
dataset_day = Pollution_Shunyi.groupby(by='day').agg({
    'PM10':'mean',
    'PM2.5':'mean',
    'O3':'mean',
    'NO2':'mean',
    'SO2':'mean',
})
dataset_hour = Pollution_Shunyi.groupby(by='hour').agg({
    'PM10':'mean',
    'PM2.5':'mean',
    'O3':'mean',
    'NO2':'mean',
    'SO2':'mean',
})
fig, (ax1, ax2) = plt.subplots(1, 2,figsize=(15, 8))
for column in dataset_year.columns:
    ax1.plot(dataset_year.index, dataset_year[column], label=column)
ax1.set_xlabel("Tahunan")
ax1.legend()
for column in dataset_month.columns:
    ax2.plot(dataset_month.index, dataset_month[column], label=column)
ax2.set_xlabel("Bulanan")
ax2.legend()
fig, (ax3, ax4) = plt.subplots(1, 2,figsize=(15, 8))
for column in dataset_day.columns:
    ax3.plot(dataset_day.index, dataset_day[column], label=column)
ax3.set_xlabel("30 Hari")
ax3.legend()
for column in dataset_hour.columns:
    ax4.plot(dataset_hour.index, dataset_hour[column], label=column)
ax4.set_xlabel("Setiap Jam")
ax4.legend()


# ## Pollution Tiantan

# In[38]:


dataset_year = Pollution_Tiantan.groupby(by='year').agg({
    'PM10':'mean',
    'PM2.5':'mean',
    'O3':'mean',
    'NO2':'mean',
    'SO2':'mean',
})
dataset_month = Pollution_Tiantan.groupby(by='month').agg({
    'PM10':'mean',
    'PM2.5':'mean',
    'O3':'mean',
    'NO2':'mean',
    'SO2':'mean',
})
dataset_day = Pollution_Tiantan.groupby(by='day').agg({
    'PM10':'mean',
    'PM2.5':'mean',
    'O3':'mean',
    'NO2':'mean',
    'SO2':'mean',
})
dataset_hour = Pollution_Tiantan.groupby(by='hour').agg({
    'PM10':'mean',
    'PM2.5':'mean',
    'O3':'mean',
    'NO2':'mean',
    'SO2':'mean',
})
fig, (ax1, ax2) = plt.subplots(1, 2,figsize=(15, 8))
for column in dataset_year.columns:
    ax1.plot(dataset_year.index, dataset_year[column], label=column)
ax1.set_xlabel("Tahunan")
ax1.legend()
for column in dataset_month.columns:
    ax2.plot(dataset_month.index, dataset_month[column], label=column)
ax2.set_xlabel("Bulanan")
ax2.legend()
fig, (ax3, ax4) = plt.subplots(1, 2,figsize=(15, 8))
for column in dataset_day.columns:
    ax3.plot(dataset_day.index, dataset_day[column], label=column)
ax3.set_xlabel("30 Hari")
ax3.legend()
for column in dataset_hour.columns:
    ax4.plot(dataset_hour.index, dataset_hour[column], label=column)
ax4.set_xlabel("Setiap Jam")
ax4.legend()


# ## Pollution Wanliu

# In[39]:


dataset_year = Pollution_Wanliu.groupby(by='year').agg({
    'PM10':'mean',
    'PM2.5':'mean',
    'O3':'mean',
    'NO2':'mean',
    'SO2':'mean',
})
dataset_month = Pollution_Wanliu.groupby(by='month').agg({
    'PM10':'mean',
    'PM2.5':'mean',
    'O3':'mean',
    'NO2':'mean',
    'SO2':'mean',
})
dataset_day = Pollution_Wanliu.groupby(by='day').agg({
    'PM10':'mean',
    'PM2.5':'mean',
    'O3':'mean',
    'NO2':'mean',
    'SO2':'mean',
})
dataset_hour = Pollution_Wanliu.groupby(by='hour').agg({
    'PM10':'mean',
    'PM2.5':'mean',
    'O3':'mean',
    'NO2':'mean',
    'SO2':'mean',
})
fig, (ax1, ax2) = plt.subplots(1, 2,figsize=(15, 8))
for column in dataset_year.columns:
    ax1.plot(dataset_year.index, dataset_year[column], label=column)
ax1.set_xlabel("Tahunan")
ax1.legend()
for column in dataset_month.columns:
    ax2.plot(dataset_month.index, dataset_month[column], label=column)
ax2.set_xlabel("Bulanan")
ax2.legend()
fig, (ax3, ax4) = plt.subplots(1, 2,figsize=(15, 8))
for column in dataset_day.columns:
    ax3.plot(dataset_day.index, dataset_day[column], label=column)
ax3.set_xlabel("30 Hari")
ax3.legend()
for column in dataset_hour.columns:
    ax4.plot(dataset_hour.index, dataset_hour[column], label=column)
ax4.set_xlabel("Setiap Jam")
ax4.legend()


# ## Pollution Wanshouxigong

# In[40]:


dataset_year = Pollution_Wanshouxigong.groupby(by='year').agg({
    'PM10':'mean',
    'PM2.5':'mean',
    'O3':'mean',
    'NO2':'mean',
    'SO2':'mean',
})
dataset_month = Pollution_Wanshouxigong.groupby(by='month').agg({
    'PM10':'mean',
    'PM2.5':'mean',
    'O3':'mean',
    'NO2':'mean',
    'SO2':'mean',
})
dataset_day = Pollution_Wanshouxigong.groupby(by='day').agg({
    'PM10':'mean',
    'PM2.5':'mean',
    'O3':'mean',
    'NO2':'mean',
    'SO2':'mean',
})
dataset_hour = Pollution_Wanshouxigong.groupby(by='hour').agg({
    'PM10':'mean',
    'PM2.5':'mean',
    'O3':'mean',
    'NO2':'mean',
    'SO2':'mean',
})
fig, (ax1, ax2) = plt.subplots(1, 2,figsize=(15, 8))
for column in dataset_year.columns:
    ax1.plot(dataset_year.index, dataset_year[column], label=column)
ax1.set_xlabel("Tahunan")
ax1.legend()
for column in dataset_month.columns:
    ax2.plot(dataset_month.index, dataset_month[column], label=column)
ax2.set_xlabel("Bulanan")
ax2.legend()
fig, (ax3, ax4) = plt.subplots(1, 2,figsize=(15, 8))
for column in dataset_day.columns:
    ax3.plot(dataset_day.index, dataset_day[column], label=column)
ax3.set_xlabel("30 Hari")
ax3.legend()
for column in dataset_hour.columns:
    ax4.plot(dataset_hour.index, dataset_hour[column], label=column)
ax4.set_xlabel("Setiap Jam")
ax4.legend()


# In[41]:


Polusi_Kota1 = Pollution_Aotizhongxin.groupby(by='date').agg({
    'PM10':'mean',
    'PM2.5':'mean',
    'O3':'mean',
    'NO2':'mean',
    'SO2':'mean',
})
Polusi_Kota2 = Pollution_Changping.groupby(by='date').agg({
    'PM10':'mean',
    'PM2.5':'mean',
    'O3':'mean',
    'NO2':'mean',
    'SO2':'mean',
})
Polusi_Kota3 = Pollution_Dingling.groupby(by='date').agg({
    'PM10':'mean',
    'PM2.5':'mean',
    'O3':'mean',
    'NO2':'mean',
    'SO2':'mean',
})
Polusi_Kota4 = Pollution_Dongsi.groupby(by='date').agg({
    'PM10':'mean',
    'PM2.5':'mean',
    'O3':'mean',
    'NO2':'mean',
    'SO2':'mean',
})
Polusi_Kota5 = Pollution_Guanyuan.groupby(by='date').agg({
    'PM10':'mean',
    'PM2.5':'mean',
    'O3':'mean',
    'NO2':'mean',
    'SO2':'mean',
})
Polusi_Kota6 = Pollution_Gucheng.groupby(by='date').agg({
    'PM10':'mean',
    'PM2.5':'mean',
    'O3':'mean',
    'NO2':'mean',
    'SO2':'mean',
})
Polusi_Kota7 = Pollution_Nongzhanguan.groupby(by='date').agg({
    'PM10':'mean',
    'PM2.5':'mean',
    'O3':'mean',
    'NO2':'mean',
    'SO2':'mean',
})
Polusi_Kota8 = Pollution_Shunyi.groupby(by='date').agg({
    'PM10':'mean',
    'PM2.5':'mean',
    'O3':'mean',
    'NO2':'mean',
    'SO2':'mean',
})
Polusi_Kota9 = Pollution_Tiantan.groupby(by='date').agg({
    'PM10':'mean',
    'PM2.5':'mean',
    'O3':'mean',
    'NO2':'mean',
    'SO2':'mean',
})
Polusi_Kota10 = Pollution_Wanliu.groupby(by='date').agg({
    'PM10':'mean',
    'PM2.5':'mean',
    'O3':'mean',
    'NO2':'mean',
    'SO2':'mean',
})
Polusi_Kota11 = Pollution_Wanshouxigong.groupby(by='date').agg({
    'PM10':'mean',
    'PM2.5':'mean',
    'O3':'mean',
    'NO2':'mean',
    'SO2':'mean',
})


# In[42]:


fig, (ax1, ax2) = plt.subplots(1, 2,figsize=(15, 8))
for column in Polusi_Kota1.columns:
    ax1.plot(Polusi_Kota1.index, Polusi_Kota1[column], label=column)
ax1.set_xlabel("Data Harian Aotizhongxin")
ax1.legend()
for column in Polusi_Kota2.columns:
    ax2.plot(Polusi_Kota2.index, Polusi_Kota2[column], label=column)
ax2.set_xlabel("Data Harian Changping")
ax2.legend()
fig, (ax3, ax4) = plt.subplots(1, 2,figsize=(15, 8))
for column in Polusi_Kota3.columns:
    ax3.plot(Polusi_Kota3.index, Polusi_Kota3[column], label=column)
ax3.set_xlabel("Data Harian Dingling")
ax3.legend()
for column in Polusi_Kota4.columns:
    ax4.plot(Polusi_Kota4.index, Polusi_Kota4[column], label=column)
ax4.set_xlabel("Data Harian Dongsi")
ax4.legend()


# In[43]:


fig, (ax1, ax2) = plt.subplots(1, 2,figsize=(15, 8))
for column in Polusi_Kota5.columns:
    ax1.plot(Polusi_Kota5.index, Polusi_Kota5[column], label=column)
ax1.set_xlabel("Data Harian Guanyuan")
ax1.legend()
for column in Polusi_Kota6.columns:
    ax2.plot(Polusi_Kota6.index, Polusi_Kota6[column], label=column)
ax2.set_xlabel("Data Harian Gucheng")
ax2.legend()
fig, (ax3, ax4) = plt.subplots(1, 2,figsize=(15, 8))
for column in Polusi_Kota7.columns:
    ax3.plot(Polusi_Kota7.index, Polusi_Kota7[column], label=column)
ax3.set_xlabel("Data Harian Nongzhanguan")
ax3.legend()
for column in Polusi_Kota8.columns:
    ax4.plot(Polusi_Kota8.index, Polusi_Kota8[column], label=column)
ax4.set_xlabel("Data Harian Shunyi")
ax4.legend()


# In[44]:


fig, (ax1, ax2,ax3) = plt.subplots(3, 1,figsize=(15, 12))
for column in Polusi_Kota9.columns:
    ax1.plot(Polusi_Kota9.index, Polusi_Kota9[column], label=column)
ax1.set_xlabel("Data Harian Tiantan")
ax1.legend()
for column in Polusi_Kota10.columns:
    ax2.plot(Polusi_Kota10.index, Polusi_Kota10[column], label=column)
ax2.set_xlabel("Data Harian Wanliu")
ax2.legend()
for column in Polusi_Kota11.columns:
    ax3.plot(Polusi_Kota11.index, Polusi_Kota11[column], label=column)
ax3.set_xlabel("Data Harian Wanshouxigong")
ax3.legend()


# In[45]:


# 2013 - 2017
Pollutant = ['PM2.5','PM10','SO2','NO2','O3']
data_kota1 = Pollution_Aotizhongxin.copy()
data_kota2 = Pollution_Changping.copy()
data_kota3 =Pollution_Dingling.copy()
data_kota4 = Pollution_Dongsi.copy()
data_kota5 =Pollution_Guanyuan.copy()
data_kota6 =Pollution_Gucheng.copy()
data_kota7 =Pollution_Nongzhanguan.copy()
data_kota8 =Pollution_Shunyi.copy()
data_kota9 =Pollution_Tiantan.copy()
data_kota10=Pollution_Wanliu.copy()
data_kota11=Pollution_Wanshouxigong.copy()


# In[46]:


data_prediksi1  = pd.pivot_table(data=data_kota1,index=["hour"],values=Pollutant,aggfunc='mean').round()
data_prediksi2  = pd.pivot_table(data=data_kota2,index=["hour"],values=Pollutant,aggfunc='mean').round()
data_prediksi3  = pd.pivot_table(data=data_kota3,index=["hour"],values=Pollutant,aggfunc='mean').round()
data_prediksi4  = pd.pivot_table(data=data_kota4,index=["hour"],values=Pollutant,aggfunc='mean').round()
data_prediksi5  = pd.pivot_table(data=data_kota5,index=["hour"],values=Pollutant,aggfunc='mean').round()
data_prediksi6  = pd.pivot_table(data=data_kota6,index=["hour"],values=Pollutant,aggfunc='mean').round()
data_prediksi7  = pd.pivot_table(data=data_kota7,index=["hour"],values=Pollutant,aggfunc='mean').round()
data_prediksi8  = pd.pivot_table(data=data_kota8,index=["hour"],values=Pollutant,aggfunc='mean').round()
data_prediksi9  = pd.pivot_table(data=data_kota9,index=["hour"],values=Pollutant,aggfunc='mean').round()
data_prediksi10 = pd.pivot_table(data=data_kota10,index=["hour"],values=Pollutant,aggfunc='mean').round()
data_prediksi11 = pd.pivot_table(data=data_kota11,index=["hour"],values=Pollutant,aggfunc='mean').round()


# In[47]:


data_prediksi1['kategori'] = pd.Series()
for index, row in enumerate(data_prediksi1.itertuples()):
    temp_treeshold = 0
    for cell in row:
        if type(cell) == int or type(cell) == float:
            if cell > temp_treeshold:
                if cell in range(1,50):
                    data_prediksi1.kategori.loc[index] = '1.Baik'
                elif cell in range(51,100):
                    data_prediksi1.kategori.loc[index] = '2.Sedang'
                elif cell in range(101,200):
                    data_prediksi1.kategori.loc[index] = '3.Tidak Sehat'
                elif cell in range(201,300):
                    data_prediksi1.kategori.loc[index] = '4.Sangat Tidak Sehat'
                else:
                    data_prediksi1.kategori.loc[index] = '5.Berbahaya'
                temp_treeshold = cell
data_prediksi2['kategori'] = pd.Series()
for index, row in enumerate(data_prediksi2.itertuples()):
    temp_treeshold = 0
    for cell in row:
        if type(cell) == int or type(cell) == float:
            if cell > temp_treeshold:
                if cell in range(1,50):
                    data_prediksi2.kategori.loc[index] = '1.Baik'
                elif cell in range(51,100):
                    data_prediksi2.kategori.loc[index] = '2.Sedang'
                elif cell in range(101,200):
                    data_prediksi2.kategori.loc[index] = '3.Tidak Sehat'
                elif cell in range(201,300):
                    data_prediksi2.kategori.loc[index] = '4.Sangat Tidak Sehat'
                else:
                    data_prediksi2.kategori.loc[index] = '5.Berbahaya'
                temp_treeshold = cell
data_prediksi3['kategori'] = pd.Series()
for index, row in enumerate(data_prediksi3.itertuples()):
    temp_treeshold = 0
    for cell in row:
        if type(cell) == int or type(cell) == float:
            if cell > temp_treeshold:
                if cell in range(1,50):
                    data_prediksi3.kategori.loc[index] = '1.Baik'
                elif cell in range(51,100):
                    data_prediksi3.kategori.loc[index] = '2.Sedang'
                elif cell in range(101,200):
                    data_prediksi3.kategori.loc[index] = '3.Tidak Sehat'
                elif cell in range(201,300):
                    data_prediksi3.kategori.loc[index] = '4.Sangat Tidak Sehat'
                else:
                    data_prediksi3.kategori.loc[index] = '5.Berbahaya'
                temp_treeshold = cell
data_prediksi4['kategori'] = pd.Series()
for index, row in enumerate(data_prediksi4.itertuples()):
    temp_treeshold = 0
    for cell in row:
        if type(cell) == int or type(cell) == float:
            if cell > temp_treeshold:
                if cell in range(1,50):
                    data_prediksi4.kategori.loc[index] = '1.Baik'
                elif cell in range(51,100):
                    data_prediksi4.kategori.loc[index] = '2.Sedang'
                elif cell in range(101,200):
                    data_prediksi4.kategori.loc[index] = '3.Tidak Sehat'
                elif cell in range(201,300):
                    data_prediksi4.kategori.loc[index] = '4.Sangat Tidak Sehat'
                else:
                    data_prediksi4.kategori.loc[index] = '5.Berbahaya'
                temp_treeshold = cell
data_prediksi5['kategori'] = pd.Series()
for index, row in enumerate(data_prediksi5.itertuples()):
    temp_treeshold = 0
    for cell in row:
        if type(cell) == int or type(cell) == float:
            if cell > temp_treeshold:
                if cell in range(1,50):
                    data_prediksi5.kategori.loc[index] = '1.Baik'
                elif cell in range(51,100):
                    data_prediksi5.kategori.loc[index] = '2.Sedang'
                elif cell in range(101,200):
                    data_prediksi5.kategori.loc[index] = '3.Tidak Sehat'
                elif cell in range(201,300):
                    data_prediksi5.kategori.loc[index] = '4.Sangat Tidak Sehat'
                else:
                    data_prediksi5.kategori.loc[index] = '5.Berbahaya'
                temp_treeshold = cell
data_prediksi6['kategori'] = pd.Series()
for index, row in enumerate(data_prediksi6.itertuples()):
    temp_treeshold = 0
    for cell in row:
        if type(cell) == int or type(cell) == float:
            if cell > temp_treeshold:
                if cell in range(1,50):
                    data_prediksi6.kategori.loc[index] = '1.Baik'
                elif cell in range(51,100):
                    data_prediksi6.kategori.loc[index] = '2.Sedang'
                elif cell in range(101,200):
                    data_prediksi6.kategori.loc[index] = '3.Tidak Sehat'
                elif cell in range(201,300):
                    data_prediksi6.kategori.loc[index] = '4.Sangat Tidak Sehat'
                else:
                    data_prediksi6.kategori.loc[index] = '5.Berbahaya'
                temp_treeshold = cell
data_prediksi7['kategori'] = pd.Series()
for index, row in enumerate(data_prediksi7.itertuples()):
    temp_treeshold = 0
    for cell in row:
        if type(cell) == int or type(cell) == float:
            if cell > temp_treeshold:
                if cell in range(1,50):
                    data_prediksi7.kategori.loc[index] = '1.Baik'
                elif cell in range(51,100):
                    data_prediksi7.kategori.loc[index] = '2.Sedang'
                elif cell in range(101,200):
                    data_prediksi7.kategori.loc[index] = '3.Tidak Sehat'
                elif cell in range(201,300):
                    data_prediksi7.kategori.loc[index] = '4.Sangat Tidak Sehat'
                else:
                    data_prediksi7.kategori.loc[index] = '5.Berbahaya'
                temp_treeshold = cell
data_prediksi8['kategori'] = pd.Series()
for index, row in enumerate(data_prediksi8.itertuples()):
    temp_treeshold = 0
    for cell in row:
        if type(cell) == int or type(cell) == float:
            if cell > temp_treeshold:
                if cell in range(1,50):
                    data_prediksi8.kategori.loc[index] = '1.Baik'
                elif cell in range(51,100):
                    data_prediksi8.kategori.loc[index] = '2.Sedang'
                elif cell in range(101,200):
                    data_prediksi8.kategori.loc[index] = '3.Tidak Sehat'
                elif cell in range(201,300):
                    data_prediksi8.kategori.loc[index] = '4.Sangat Tidak Sehat'
                else:
                    data_prediksi8.kategori.loc[index] = '5.Berbahaya'
                temp_treeshold = cell
data_prediksi9['kategori'] = pd.Series()
for index, row in enumerate(data_prediksi9.itertuples()):
    temp_treeshold = 0
    for cell in row:
        if type(cell) == int or type(cell) == float:
            if cell > temp_treeshold:
                if cell in range(1,50):
                    data_prediksi9.kategori.loc[index] = '1.Baik'
                elif cell in range(51,100):
                    data_prediksi9.kategori.loc[index] = '2.Sedang'
                elif cell in range(101,200):
                    data_prediksi9.kategori.loc[index] = '3.Tidak Sehat'
                elif cell in range(201,300):
                    data_prediksi9.kategori.loc[index] = '4.Sangat Tidak Sehat'
                else:
                    data_prediksi9.kategori.loc[index] = '5.Berbahaya'
                temp_treeshold = cell
data_prediksi10['kategori'] = pd.Series()
for index, row in enumerate(data_prediksi10.itertuples()):
    temp_treeshold = 0
    for cell in row:
        if type(cell) == int or type(cell) == float:
            if cell > temp_treeshold:
                if cell in range(1,50):
                    data_prediksi10.kategori.loc[index] = '1.Baik'
                elif cell in range(51,100):
                    data_prediksi10.kategori.loc[index] = '2.Sedang'
                elif cell in range(101,200):
                    data_prediksi10.kategori.loc[index] = '3.Tidak Sehat'
                elif cell in range(201,300):
                    data_prediksi10.kategori.loc[index] = '4.Sangat Tidak Sehat'
                else:
                    data_prediksi10.kategori.loc[index] = '5.Berbahaya'
                temp_treeshold = cell
data_prediksi11['kategori'] = pd.Series()
for index, row in enumerate(data_prediksi11.itertuples()):
    temp_treeshold = 0
    for cell in row:
        if type(cell) == int or type(cell) == float:
            if cell > temp_treeshold:
                if cell in range(1,50):
                    data_prediksi11.kategori.loc[index] = '1.Baik'
                elif cell in range(51,100):
                    data_prediksi11.kategori.loc[index] = '2.Sedang'
                elif cell in range(101,200):
                    data_prediksi11.kategori.loc[index] = '3.Tidak Sehat'
                elif cell in range(201,300):
                    data_prediksi11.kategori.loc[index] = '4.Sangat Tidak Sehat'
                else:
                    data_prediksi11.kategori.loc[index] = '5.Berbahaya'
                temp_treeshold = cell


# In[48]:


# Aotizhongxin
# Changping
# Dingling
# Dongsi 
# Guanyuan 
# Gucheng 
# Nongzhanguan 
# Shunyi 
# Tiantan 
# Wanliu 
# Wanshouxigong 
data_prediksi_kota1 = data_prediksi1.sort_values('kategori',ascending=False)
data_prediksi_kota2 = data_prediksi2.sort_values('kategori',ascending=False)
data_prediksi_kota3 = data_prediksi3.sort_values('kategori',ascending=False)
data_prediksi_kota4 = data_prediksi4.sort_values('kategori',ascending=False)
data_prediksi_kota5 = data_prediksi5.sort_values('kategori',ascending=False)
data_prediksi_kota6 = data_prediksi1.sort_values('kategori',ascending=False)
data_prediksi_kota7 = data_prediksi2.sort_values('kategori',ascending=False)
data_prediksi_kota8 = data_prediksi3.sort_values('kategori',ascending=False)
data_prediksi_kota9 = data_prediksi4.sort_values('kategori',ascending=False)
data_prediksi_kota10 = data_prediksi5.sort_values('kategori',ascending=False)
data_prediksi_kota11 = data_prediksi5.sort_values('kategori',ascending=False)
print("Prediksi Udara yang memiliki kualitas terburuk dalam kurun waktu pada kota\nAotizhongxin pada pukul",data_prediksi_kota1.index[0],"AM")
print("Changping pada pukul",data_prediksi_kota2.index[0],"AM")
print("Dingling pada pukul",data_prediksi_kota3.index[0],"PM")
print("Dongsi pada pukul",data_prediksi_kota4.index[0],"PM")
print("Guanyuan pada pukul",data_prediksi_kota5.index[0],"PM")
print("Gucheng pada pukul",data_prediksi_kota6.index[0],"PM")
print("Nongzhanguan pada pukul",data_prediksi_kota7.index[0],"PM") 
print("Shunyi pada pukul",data_prediksi_kota8.index[0],"PM") 
print("Tiantan pada pukul",data_prediksi_kota9.index[0],"PM") 
print("Wanliu pada pukul",data_prediksi_kota10.index[0],"PM") 
print("Wanshouxigong pada pukul",data_prediksi_kota11.index[0],"PM") 


# ## Kesimpulan

# 1. Bagaimana Kualitas Udara pada setiap kota dalam kurun waktu tertentu ?
# 2. Nilai Polutan paling tinggi berdasarkan rata-rata seluruh waktu terdapat pada kota?
# 3. Kapan kualitas udara memiliki kualitas tidak baik dalam kurun waktu tertentu pada setiap kota berdasarkan Indeks Standar Pencemar Udara?

# 1. Berdasarkan dari hasil penelitian dengan mengkategorikan periode waktu yaitu harian, 30 hari, bulanan, tahunan mendapatkan hasil bahwa pada 
# Berdasarkan hasil penelitian maka didapatkan kesimpulan 
# yaitu kota Aotizhongxin mulai memiliki kualitas udara
# terburuk pada tahun 2014 hingga 2016 menurun tetapi
# kembali memuncak pada tahun 2017,
# kota Changping memiliki kualitas udara terburuk pada tahun 
# 2014 dengan kualitas udara tertinggi dimulai dari bulan 3 
# menurun hingga bulan 10 kemudian kembali memuncak pada
# bulan 12 dan kualitas udara harian terburuk terjadi pada
# pertengahan bulan.
# kota Dingling memiliki kualitas udara terburuk mulai pada
# tahun 2014 dengan kualitas udara terburuk terjadi pada
# bulan 3 hingga bulan 9 kemudian kembali memuncak dibulan 12
# dan kualitas udara harian terburuk terjadi pada pertengahan
# bulan.
# Kota Dongsi memiliki kualitas udara terburuk pada tahun 
# 2017 dengan kualitas udara terburuk pada bulan 3 hingga bulan
# 8 kembali memuncak pada bulan 12 dan kualitas udara harian
# terburuk terjadi pada pertengahan bulan.
# Kota Guanyuan memiliki kualitas udara terburuk mulai pada tahun 2014
# kemudian memuncak pada tahun 2017 dengan kualitas terburuk
# pada bulan 3 hingga bulan 8 kembali memuncak pada bulan 12 dan 
# kualitas udara harian terburuk terjadi pada pertengahan bulan.
# Kota Gucheng memiliki kualitas udara terburuk pada tahun 2014
# dengan kualitas udara terburuk pada bulan 3 hingga bulan 8 kembali 
# memuncak pada tahun 12 dan kualitas udara harian terburuk terjadi
# pada pertengahan bulan
# Kota Nongzhanguan memiliki kualitas udara terburuk pada tahun 2014
# dengan puncak kualitas udara terjadi pada bulan 3 dan kualitas 
# udara harian terburuk terjadi pada pertengahan bulan.
# Kota Shunyi memiliki kualitas udara terburuk pada tahun 2014 dengan 
# puncak kualitas udara terburuk terjadi pada bulan 4 dan kualitas
# udara harian terburuk terjadi pada pertengahan bulan
# Kota Tiantan memiliki kualitas udara terburuk pada tahun 2017 dengan 
# puncak kualitas udara terburuk terjadi pada akhir tahun dan kualitas
# udara harian terburuk terjadi pada pertengahan bulan.
# Kota Wanliu memiliki kualitas udara terburuk pada tahun 2014 dengan
# puncak kualitas yang terjadi pada bulan 3 dan kualitas udara harian 
# terburuk terjadi pada pertengahan bulan
# Kota Wanshouxigong memiliki kualitas udara terburuk pada tahun 2014
# dengan puncak kualitas udara terburuk pada bulan 3 dan kualitas udara 
# harian terburuk terjadi pada pertengahan bulan.

# 2. Berdasarkan dari hasil penelitian maka didapatkan hasil yaitu data polutan PM2.5 pada kota Dongsi, polutan PM10 pada kota Guchen, polutan SO2 pada kota Nongzhanguan, polutan NO2 pada kota Wanliu, polutan O3 pada kota Dingling, polutan CO pada kota Wanshouxigong

# 3. Prediksi Udara yang memiliki kualitas terburuk dalam kurun waktu pada kota
# Aotizhongxin pada pukul 6 AM
# ,Changping pada pukul 9 AM
# ,Dingling pada pukul 17 PM
# ,Dongsi pada pukul 15 PM
# ,Guanyuan pada pukul 15 PM
# ,Gucheng pada pukul 6 PM
# ,Nongzhanguan pada pukul 9 PM
# ,Shunyi pada pukul 17 PM
# ,Tiantan pada pukul 15 PM
# ,Wanliu pada pukul 15 PM
# ,Wanshouxigong pada pukul 15 PM
