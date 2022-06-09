#!/usr/bin/env python
# coding: utf-8

# 
# # Project: Investigate a Dataset - [No-Show Appointments]
# 
# ## Table of Contents
# <ul>
# <li><a href="#intro">Introduction</a></li>
# <li><a href="#wrangling">Data Wrangling</a></li>
# <li><a href="#eda">Exploratory Data Analysis</a></li>
# <li><a href="#conclusions">Conclusions</a></li>
# </ul>

# <a id='intro'></a>
# ## Introduction
# 
# ### Dataset Description 
# 
# This dataset collects information from 100k medical appointments in Brazil and is focused on the question of whether or not patients show up for their appointment. A number of characteristics about the patient are included in each row.
# 
# ● PatientId: indicates the patient ID --> i've to check if there is more than one appointment for the same patient ID.
# 
# ● AppointmentID: indicates appoint ID --> it should be unique.
# 
# ● Gender: indicates the patient's gender Male or Female.
# 
# ● ScheduledDay: tells us on what day the patient set up their appointment.
# 
# ● AppointmentDay: indicates the date/time the patient called to book their appointment.
# 
# ● Age: indicates the patient's age.
# 
# ● Neighborhood: indicates the location of the hospital.
# 
# ● Scholarship: indicates whether or not the patient is enrolled in Brasilian welfare program Bolsa Família.
# 
# ● Hipertension: indicates whether or not the patient is experiencing Hypertension.
# 
# ● Diabetes: indicates whether or not the patient is experiencing Diabetes.
# 
# ● Alcoholism: indicates whether or not the patient is experiencing Alcoholism.
# 
# ● Handcap: indicates whether or not the patient is with special needs.
# 
# ● SMS_received: indicates whether or not the patient has received a reminder text message.
# 
# ● Show-up: ‘No’ if the patient showed up to their appointment, and ‘Yes’ if they did not show up.
# 
# 
# ### Research Questions:
#  What is the affect of (SMS recieving , Age ,Gender , Scholarship , Diseases , Handcap , Neighborhood) on showed or not showed patients ?

# In[4]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[5]:


# Upgrade pandas to use dataframe.explode() function. 
#!pip install --upgrade pandas==0.25.0


# <a id='wrangling'></a>
# ## Data Wrangling
# 
# ### General Properties
# 

# In[6]:


# Load your data and print out a few lines. Perform operations to inspect data
#   types and look for instances of missing or possibly errant data.
df = pd.read_csv('noshowappointments-kagglev2-may-2016.csv')
df.head()


# ##### Now i need to know more details about my dataset
# 1. its shape
# 2. if i have missing and null values
# 3. duplicating in patient ID
# 4. check if appointment ID is unique
# 5. its statistics 

# In[7]:


# 1
df.shape
#as shown i have 110527 rows & 14 columns


# In[8]:


# 2
df.info()
# i have no missing data


# In[9]:


df.isnull().sum()


# In[10]:


#3
sum(df.PatientId.duplicated())


# In[11]:


#4
sum(df.AppointmentID.duplicated())


# In[12]:


#5
df.describe()


# #### i noticed that i have 2 strange values in age column 
# the min value = -1
# and the max value = 115

# ###  Data cleaning

# In[13]:


# as shown above i have an age with (-1) which is wrong, age cant be negative
df.loc[df['Age'] == -1]


# In[14]:


# remove the row of (-)1 age , i used inplace to save changes in the original dataset
df.drop([99832], inplace=True)


# In[15]:


df.loc[df['Age'] == 115]


# #### now i have 5 rows age in it equals 115 but 4 rows of them are for the same patient
# so i have 2 patients 115 years old

# In[16]:


# i need to change this column name because it caused error because of '-' sign
df.rename(columns ={'No-show':'No_show'}, inplace = True)


# In[17]:


# remove patientid duplicating and its No_show data
df.drop_duplicates(['PatientId','No_show'], inplace = True)
df.shape


# Now no. of rows correct(110527 -(patientid duplicating '48228' + the deleted row of '-1' age)

# In[18]:


# i noticed that (ScheduledDay , AppointmentDay) were object(string) data type 
# i will covert them to datetime
df.AppointmentDay = pd.to_datetime(df.AppointmentDay)
df.ScheduledDay = pd.to_datetime(df.ScheduledDay)
df.info()


# <a id='eda'></a>
# ## Exploratory Data Analysis
# 
# ### Research Question 1 (If recieving SMS affect on No_show!)
# 
# 

# In[19]:


print(df.No_show.unique())


# In[20]:


# dividing No_show to showed and notshowed to make comparisons easy
showed = df.No_show == 'No'
notshowed = df.No_show == 'Yes'


# In[21]:


print(df[showed].mean()) 
print(df[notshowed].mean())


# i noticed that: the average 'SMS_received' in showed patients are less than notshowed patients
# 

# In[22]:


print(df.SMS_received.unique())
# 0 for not recieved & 1 for recieved


# In[27]:


# function to visualize the affect of SMS on showed and not showed
def visualshow (df,column_name,show,noshow):
    plt.figure(figsize = [8,8])
    df[column_name][showed].hist(alpha=.5 , bins=10, color='green' , label='showed')
    df[column_name][notshowed].hist(alpha=.5 , bins=10, color='red' , label='notshowed')
    plt.legend()
    plt.title('Showed VS NotShowed according to SMS_received ')
    plt.xlabel('SMS_received')
    plt.ylabel('Patients numbers')

visualshow(df,'SMS_received',showed,notshowed)   


# patients who didnt recieve SMS are the most

# ### Research Question 2  (If Age affect on 'No-show'!)

# In[26]:


sns.boxplot(x=df.No_show, y=df.Age)
plt.show()


# Age approximately from 0 to 15 and from 43 to 59 are more showed patient numbers and the lowest are approximately from 90 to 115

# ### Research Question3  (If Gender affect on 'No-show'!)

# In[22]:


# Exploring 'Gender'
print(df.Gender.unique())
print(df['Gender'].value_counts())


# In[23]:


df[showed].groupby('Gender').Age.median()
df[notshowed].groupby('Gender').Age.median()


# In[24]:


df[showed].groupby('Gender').Age.mean().plot(kind='bar',color='blue',label='showed')
df[notshowed].groupby('Gender').Age.mean().plot(kind='bar',color='black',label='notshowed')


# In[25]:


plt.figure(figsize = [8,8])
df['Gender'][showed].hist(alpha=.5 , bins=10, color='green' , label='showed')
df['Gender'][notshowed].hist(alpha=.5 , bins=10, color='red' , label='notshowed')
plt.legend()
plt.title('Showed VS NotShowed according to Gender ')
plt.xlabel('Gender')
plt.ylabel('Patients numbers')


# females in showed and not showed are more than males 

# ### Research Question3  (If Scholarship affect on 'No-show'!)

# In[26]:


ax=sns.countplot(x=df.Scholarship, hue = df.No_show,data=df)
ax.set_title('Showed patients according to Scholarship ')
plt.show()


# it seems like Scholarship doesnt affect while showed patients who have Scholarships are so small compared to patients who dont have Scholarships

# ### Research Question 4  (If diseases affect on 'No-show'!)

# In[27]:


df[showed].groupby(['Hipertension','Diabetes']).mean()['Age'].plot(kind='bar',color='blue',label='showed')
df[notshowed].groupby(['Hipertension','Diabetes']).mean()['Age'].plot(kind='bar',color='pink',label='notshowed')
plt.legend();


# 'Hipertension','Diabetes' dont affect on showed & not showed patients
# because the bar of no Hipertension, no Diabetes not showed also the most like others

# ### Research Question 5 (If Handcap affect on 'No-show'!)

# In[28]:


print(df.Handcap.unique())


# In[29]:


ax=sns.countplot(x=df.Handcap, hue = df.No_show,data=df)
ax.set_title('Showed patients according to Handcap ')
plt.show()


# so it is clear that patients who have no Handcap  most of them showed patients 
# 

# ### Research Question 6  (If Neighbourhood affect on 'No-show'!)

# In[30]:


print(df.Neighbourhood.unique())


# In[31]:


plt.figure(figsize = [18,8])
df['Neighbourhood'][showed].value_counts().plot(kind='bar' , color='green', label='showed')
df['Neighbourhood'][notshowed].value_counts().plot(kind='bar' , color='yellow', label='notshowed')
plt.legend()
plt.title('Showed patients according to Neighbourhood ')
plt.xlabel('Neighbourhood')
plt.ylabel('Patients numbers')


# 'JARDIM CAMBURI' has the most showed patients it might be the closest city to the hospital

# <a id='conclusions'></a>
# ## Conclusions
# 
# 1. SMS_recieving doesnt affect on No_show field
# showed patients more than not showed patients although they recieved SMS less than not showed patients.
# 
# 2. Age has affect on showed & not showed patients
# 
# 3. there is no relation between Gender and Age and their affect on No_show feild
# 
# 4. Scholarship doesnt affect on No_show field 
# most of patients who dont have Scholarship are showed patients
# 
# 5. diseases have no affect on No_show 
# 
# 6. Handcap affects on No_show
# most of patients who have no handcap are showed patients
# 
# 6.Neighbourhood affects on No_show
# 
# ### Limitations:
# The Nieghborhood data didnt helped me alot i need to know the destance between each city and the hospital to make sure for ezample if 'JARDIM CAMBURI' is the most showed patients beacuse it really is the nearest city

# In[32]:


from subprocess import call
call(['python', '-m', 'nbconvert', 'Investigate_a_Dataset.ipynb'])


# In[ ]:




