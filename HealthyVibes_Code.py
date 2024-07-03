#!/usr/bin/env python
# coding: utf-8

# <div style="background-color: #161D3A; border: 1px solid #000; margin: 0 2px; padding: 10px 1px 2px 2px;">
#     <center>
#         <b><font size="6" color=#FFFFFF face="sans-serif">HealthyVibes</font></b>
#     </center>
#     <center> </center>
#     <center>
#         <b><font size="4" color=#FA6400 face="sans-serif">Nancy Tran</font></b>
#     </center>
# </div>

# <div style="background-color: #161D3A ; border: 1px solid #000; margin: 0 2px; padding: 2px 1px 2px 2px;">
#     <b><font size="4" color=#FA6400 face="sans-serif">Table of Contents</font></b>
# </div>

# **[1. Assignment](#Assignment)** <br>
# **[2. Objective](#Objective)** <br>
# **[3. Performance Metrics](#Performance_Metrics)** <br>
# **[4. Data Preparation](#Data_Preparation)** <br>
# * [4.1 Gathering the data](#Gathering_Data)<br>
# * [4.2 Data Cleaning](#Data_Cleaning)<br>
# * [4.3 Data Construction](#Data_Construction)<br>
# 
# **[5. Data Exploration](#Data_Exploration)** <br>
# * [5.1 Checking for Normality](#Checking_Normality)<br>
# * [5.2 Data Visualizations](#Data_Visualizations)<br>
# * [5.3 Statistical Tests](#Statistical_Tests)<br>
# 
# **[6. Data Modeling](#Data_Modeling)** <br>
# **[7. Take Away](#Take_Away)** <br>
# **[8. Recommendations](#Recommendations)** <br>
# **[9. Presentation](#Presentation)** <br>
# **[10. Works Cited](#Works_Cited)** <br>

# <a id="Assignment"></a>
# <div style="background-color: #161D3A ; border: 1px solid #000; margin: 0 2px; padding: 2px 1px 2px 2px;">
#     <b><font size="4" color=#FA6400 face="sans-serif">1. Assignment</font></b>
# </div>

# Our company develops AI solutions to help optimize radiologist workflow and expedite patient care. For this exercise we will focus on a specific metric called Wait Time. The Wait Time is defined as the time interval between the study acquisition and the case open for reporting events. This metric is helpful to understand how long a case has been sitting unopened on the radiologist reading list.
# <br><br>
# A customer “HealthyVibes” has just deployed and integrated a set of our AI solutions in their radiologist workflow. The radiologists now get real time alerts for cases suspected to be positive. “HealthyVibes” would like to explore the impact of this AI driven workflow. “HealthyVibes” has kindly provided an export of 6 months of data since the solution went live.<br><br>
# “HealthyVibes” is so excited by the adoption of our solution, they are now wanting to set up a research study to measure the impact of our solution on study prioritization. They have asked us to perform an analysis for Wait Time and explore if AI helps expedite cases for reporting.
# <br><br>
# The data contains the following data elements ("HV_WaitTime_6m.csv"):
# - accession: A unique study level identifier
# - aidoc_site: The name of the site
# - algorithm: The algorithm the case was run on
# - patient_class: The patient class “source” of the patient, which can be from the emergency department, admitted/inpatient or ambulatory/outpatient.
# - aidoc_result: The AI result and if the cases was found to be suspected positive by the algorithm
# - wait_time_minutes: The time interval in minutes = (case_open_time - study_aquisition_time)
# - study_aquistion_time: The date time of when the scan of the patient was acquired.
# - case_open_time: The date time of when the case was opened for reporting by the radiologist.
# <br>
# 
# We would like you to conduct this analysis. You should explore the data across different dimensions of the data such as algorithm and patient class. You should create helpful statistics such as the mean, median, etc. to generate a good understanding of the Wait Time distributions.
# <br>
# 
# You should produce figures of your findings such as scatter plots or boxplots to provide easy to understand visualizations. The analysis should ideally be performed in python, but other languages will be considered if you find them more appropriate. We will require a copy of your script/code.
# <br>
# 
# You will then prepare a 10 to 15 minute PowerPoint presentation to give a high level overview of your methodology, steps and the findings, focusing on the most impactful use cases you find.
# Some leading questions - that you might find helpful:
# - Are there past published Aidoc studies that have performed similar analysis?
# - Is Wait Time the same across different patient classes?
# - Are there statistical significance tests that can be performed to add credibility to the results?
# - Are there techniques that can help clean up the signal such as reducing outliers?
# - Are there other ways to stratify the data to look at additional subgroups?

# <a id="Objective"></a>
# <div style="background-color: #161D3A ; border: 1px solid #000; margin: 0 2px; padding: 2px 1px 2px 2px;">
#     <b><font size="4" color=#FA6400 face="sans-serif">2. Objective</font></b>
# </div>

# The objective of this research study is to measure the impact of our AI solution on the prioritization of radiology study at HealthVibes. Specifically, we aim to analyze the wait time, which is defined as the time interval between the acquisition of a radiology study and the moment it is opened for reporting by a radiologist. By examining this metric, we can assess how effectively our AI-driven workflow integrates into existing radiologist processes and whether it contributes to expedited case reporting. Our goal is to determine if the implementation of real-time alerts for cases suspected to be positive has led to a significant reduction in Wait Time, thus enhancing the efficiency and responsiveness of radiologists. Through this analysis, we hope to demonstrate the tangible benefits of our AI solution in prioritizing critical cases and improving overall patient care.

# <a id="Performance_Metric"></a>
# <div style="background-color: #161D3A ; border: 1px solid #000; margin: 0 2px; padding: 2px 1px 2px 2px;">
#     <b><font size="4" color=#FA6400 face="sans-serif">3. Performance Metrics</font></b>
# </div>

# The primary metric of interest in this study is **wait time**, which is defined as the time interval between the acquisition of a radiology study and the moment it is opened for reporting by a radiologist. This metric is crucial for understanding the efficiency of the radiologist's workflow. By measuring wait time, we can evaluate how long a study remains in the queue before being reviewed and reported, which is a key indicator of workflow bottlenecks and prioritization effectiveness. A shorter wait time suggests a more responsive and efficient system, where critical cases are addressed promptly, thus potentially leading to improved patient outcomes. The analysis of wait time will help us assess the impact of our AI solution on optimizing the radiologist's workflow and ensuring timely reporting of studies.

# <a id="Data_Preparation"></a>
# <div style="background-color: #161D3A; border: 1px solid #000; margin: 0 2px; padding: 2px 1px 2px 2px;">
#     <b><font size="4" color=#FA6400 face="sans-serif">4. Data Preparation</font></b>
# </div>

# In[1]:


import pandas as pd
import math
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
from scipy import stats
from scipy.stats import kruskal
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error


# <a id="Gathering_Data"></a>
# ### 4.1 Gathering the Data

# **Reading in the csv files with pandas**

# In[2]:


wt_df = pd.read_csv("HV_WaitTime_6m.csv")
pathology_df = pd.read_csv("Pathologies.csv")

print("------Wait Time Data------\nRow Count:", wt_df.shape[0], "\nColumn Count: ", wt_df.shape[1], "\n")
print("Column Names:\n", list(wt_df.columns))

print("\n------Pathology Data------\nRow Count:", pathology_df.shape[0], "\nColumn Count: ", pathology_df.shape[1], "\n")
print("Column Names:\n", list(pathology_df.columns), "\n")


# <a id="Data_Cleaning"></a>
# ### 4.2 Data Cleaning

# **Correcting the spelling of the "study acquisition time" column in the wait time dataset.**

# In[3]:


wt_df = wt_df.rename(columns={"study_aquistion_time": "study_acquisition_time"})
print(list(wt_df.columns))


# **Removing trailing spaces (\xa0) in the column names of the pathology dataframe and subsetting for columns of interest.**

# In[4]:


pathology_df = pathology_df.rename(columns=lambda x: x.strip())[["Product","BodyArea", "Modality"]]

# Renaming columns to match the snake casing of the column names in the wait time data
pathology_df = pathology_df.rename(columns={"Product": "product", "BodyArea": "body_area", "Modality":"modality"})
list(pathology_df.columns)


# **Merging the wait time and pathology data frames**<br>
# &emsp;Step 1: Create a common algorithm_lower_case column in both data frames that converts the algorithm entries to lower case.<br>
# &emsp;Step 2: Equate ribs algorithm from wt_df to ribfx algorithm in pathology_df (the other categories do not require manual matching)<br>
# &emsp;Step 3: Merge the 2 data frames on algorithm_lower_case and drop algorithm_lower_case and Product to reduce redundancy

# In[5]:


# Step 1: Creating the lower_case columns
wt_df['algorithm_lower_case'] = wt_df['algorithm'].str.lower()  # converts algorithm to lower case
pathology_df['algorithm_lower_case'] = pathology_df['product'].str.split("for ").str[1].str.lower()  # splits, extracts, and lower cases the algorithm portion of the 'Product'

# Step 2: Replace ribdx with ribs
pathology_df['algorithm_lower_case'] = pathology_df['algorithm_lower_case'].replace("ribfx", "ribs") 

# Step 3: Merge both dataframes and drop the lower_case and Product columns
wait_time_df = wt_df.merge(pathology_df, how='left', on='algorithm_lower_case').drop(columns=['algorithm_lower_case','product'])

print("Row Count: ", wait_time_df.shape[0],"\nColumn Count: ", wait_time_df.shape[1], "\n")
wait_time_df.head()


# **Checking for missing values**

# In[6]:


missing_values_observations = pd.DataFrame(wait_time_df.isna().sum(), columns=['na_count'])
missing_values_observations['na_percent'] = missing_values_observations['na_count']/len(wait_time_df) * 100
missing_values_observations


# Approximately 0.229% of the wait_time_minutes and case_open_time entries contain missing values. Given the low proportion, it's reasonable to omit entries with missing values from the dataset, but before doing so I want to confirm that the 407 missing values occur at the same indices for wait_time_minutes and case_open_time.

# In[7]:


# Identifying rows with missing values
missing_wait_time_values = list(wait_time_df['wait_time_minutes'].isna().index)
missing_case_open_time_values = list(wait_time_df['case_open_time'].isna().index)

if missing_wait_time_values.sort() == missing_case_open_time_values.sort():
    print("The missing values for wait_time_minutes and case_open_time occur at the same indices.")
else:
    print("The missing values for wait_time_minutes and case_open_time do not occur at the same indices.")


# After confirming that the missing values occur at the same columns, we can drop the 407 rows from the data frame.

# In[8]:


# Dropping observations with missing values
wait_time_df = wait_time_df.dropna()
wait_time_df.shape


# **Handling outliers**

# In[9]:


# Creating a function to count outliers
def outlier_detection(df, col_name):
    Q1 = df[col_name].quantile(0.25)
    Q3 = df[col_name].quantile(0.75)
    IQR = Q3 - Q1
    
    wait_time_outliers = df[(df[col_name] < Q1 - 1.5 * IQR) | (df[col_name] > Q3 + 1.5 * IQR)]
    print(f"There are {wait_time_outliers.shape[0]} '{col_name}' outlier entries in the data({round(wait_time_outliers.shape[0] / len(df) * 100, 2)}%).")

# Checking for outliers
outlier_detection(wait_time_df, 'wait_time_minutes')


# Given that nearly 18% of the data are outliers, it would be impractical to remove them. Eliminating such a significant portion of the dataset could lead to a loss of valuable information and distort the analysis. Outliers can provide important insights, such as identifying extreme variations, unusual patterns, and rare events that may be crucial for understanding systemic bottlenecks affecting wait time. Therefore, I will explore alternative methods for handling outliers that allow me to retain as much data as possible, ensuring the quality and integrity of the analysis are not compromised.

# In[10]:


# Checking central tendency of wait time
print("Mean:", wait_time_df['wait_time_minutes'].mean(), "minutes")
print("Median:", wait_time_df['wait_time_minutes'].median(), "minutes")
print("Mode:", wait_time_df['wait_time_minutes'].mode()[0], "minutes")
print("\n", wait_time_df['wait_time_minutes'].describe(),"\n")

# Checking central tendency of wait time where aidoc_result is True
print("------------------\nTrue Aidoc Results\nMean:", wait_time_df[wait_time_df['aidoc_result']==True]['wait_time_minutes'].mean(), "minutes")
print("Median:", wait_time_df[wait_time_df['aidoc_result']==True]['wait_time_minutes'].median(), "minutes")
print("Mode:", wait_time_df[wait_time_df['aidoc_result']==True]['wait_time_minutes'].mode()[0], "minutes")
print("\n", wait_time_df[wait_time_df['aidoc_result']==True]['wait_time_minutes'].describe())

# Checking central tendency of wait time where aidoc_result is True
print("------------------\nFalse Aidoc Results\nMean:", wait_time_df[wait_time_df['aidoc_result']==False]['wait_time_minutes'].mean(), "minutes")
print("Median:", wait_time_df[wait_time_df['aidoc_result']==False]['wait_time_minutes'].median(), "minutes")
print("Mode:", wait_time_df[wait_time_df['aidoc_result']==False]['wait_time_minutes'].mode()[0], "minutes")
print("\n", wait_time_df[wait_time_df['aidoc_result']==False]['wait_time_minutes'].describe())


# **My observations regarding the data's central tendency:**
# - The Mean > Median > Mode relationship indicates that the data is right-skewed (positively skewed). 
# - A large standard deviation across the entire dataset and within each subset of the aidoc_result suggests significant data variability.
# - Extreme wait time values are observed:
#     - Approximately 39.5 days for the overall maximum and maximum for "False" aidoc_results.
#     - Approximately 21.02 days for "True" aidoc_results.
# 
# I initially assumed that wait times would be shorter for accessions flagged as True in the aidoc_result, which was confirmed by the data. However, the high variability and presence of extreme wait times indicate that the data is highly variable.

# **Creating visuals to explore wait time distribution**

# The relationship where the Mean > Median > Mode indicates that the data is right-skewed (positively skewed). The large standard deviation also suggests significant variation within the data. To handle outliers in such right-skewed data with high variability, I believe the best approach is to apply a log transformation. This method will help to scale and spread the data more evenly and reduce the influence of extreme values.

# **Log Transformation of wait_time_minutes**

# In[11]:


# Transforming wait_time_minutes by using the log method to handle outliers
import numpy as np
wait_time_df['wait_time_minutes_log'] = np.log(wait_time_df['wait_time_minutes'])

# Checking for outliers after the log transformation
outlier_detection(wait_time_df, 'wait_time_minutes_log')


# If I had the opportunity to consult with clinicians involved in these studies, I would seek their insight into the factors influencing these outliers. Specifically, I would inquire whether or not these extreme values were isolated incidents or representative of typical behavior in similar cases.
# 
# Regarding the ~3% of data that still contains outliers after log transformation, I believe retaining these outliers is crucial for a comprehensive analysis. They could reveal unique patterns or rare occurrences that might indicate underlying systemic bottlenecks or issues.
# 
# If we find that the 3% outlier rate is too high for our analysis goals, we could consider applying the Winsorization method. This approach involves replacing extreme values with the upper and lower percentile cutoffs of our choice, which helps to preserve the integrity of the dataset. For example, extreme wait times could be replaced with the 95th percentile value, while unusually low wait times could be replaced with the 5th percentile value.
# 
# The code snippet below has been temporarily commented out to retain the outliers for the current analysis. It can be uncommented later if we decide to implement the Winsorization method to handle outliers.

# In[12]:


# # Calculating the percentile values for Winsorization at 5th and 95th percentiles
# low_limit = np.percentile(wait_time_df['wait_time_minutes_log'], 5)
# high_limit = np.percentile(wait_time_df['wait_time_minutes_log'], 95)

# # Replacing extreme values with the 95th and 5th percentile values
# wait_time_df['wait_time_minutes_log'] = np.clip(wait_time_df['wait_time_minutes_log'], low_limit, high_limit)

# # Checking for outliers after Winsorization
# outlier_detection(wait_time_df, 'wait_time_minutes_log')


# <a id="Data_Construction"></a>
# ### 4.3 Data Construction

# **Breaking the date time attributes into more granular categories: weekday, month, and hour**
# 
# Wait times can be influenced by factors beyond Aidoc's AI programs. For example:
# - Day of the week: There could be longer wait times over the weekend due to higher patient volume and reduced staffing schedules.
# - Month: Summer months could lead to more outdoor injuries, affecting wait time.
# - Hour of the day: Scheduled facility activities like cleaning or system updates, and shift changes, can extend patient care processes.
# 
# By breaking down these factors into more detailed categories such as weekday, month, and hour, we can uncover unique insights and identify systemic bottlenecks that impact wait times. Understanding these bottlenecks can help streamline processes, reduce wait times, and ultimately improve patient outcomes.

# In[13]:


print(wait_time_df[['study_acquisition_time', 'case_open_time']].dtypes,"\n")
print("Min acquistion time:", wait_time_df['study_acquisition_time'].min(), "\nMax acquistion time:", wait_time_df['study_acquisition_time'].max())
print("\nMin case open time:", wait_time_df['case_open_time'].min(), "\nMax case open time:", wait_time_df['case_open_time'].max())

wait_time_df[['study_acquisition_time', 'case_open_time']].head()


# In[14]:


# Converting date columns from an object to a date time data type
wait_time_df['study_acquisition_time'] = pd.to_datetime(wait_time_df['study_acquisition_time'].astype(str), format="%m/%d/%y %H:%M", errors='coerce')
wait_time_df['case_open_time'] = pd.to_datetime(wait_time_df['case_open_time'].astype(str), format="%d/%m/%y %H:%M:%S%f", errors='coerce')

# Study Acquisition: weekday, month, hour 
wait_time_df['study_acquisition_weekday'] = wait_time_df['study_acquisition_time'].dt.day_name()
wait_time_df['study_acquisition_month'] = wait_time_df['study_acquisition_time'].dt.month_name()
wait_time_df['study_acquisition_hour'] = wait_time_df['study_acquisition_time'].dt.strftime("%H")

# Case Open: weekday, month, hour 
wait_time_df['case_open_weekday'] = wait_time_df['case_open_time'].dt.day_name()
wait_time_df['case_open_month'] = wait_time_df['case_open_time'].dt.month_name()
wait_time_df['case_open_hour'] = wait_time_df['case_open_time'].dt.strftime("%H")


# I would have explored by year as well, but the dataset only includes data from 2024. For this analysis, I only categorized the date times into weekday, month, and hour.

# **Creating dummy variables for the following categorical variables:**
# - algorithm
# - patient_class
# - aidoc_result
# - study_acquisition_weekday
# - study_acquisition_month
# - study_acquisition_hour
# - case_open_weekday
# - case_open_month
# - case_open_hour
# - body_area
# - modality

# In[15]:


# List of categorical columns in wait_time_df
categorical_columns = ['algorithm',
                       'patient_class',
                       'aidoc_result',
                       'study_acquisition_weekday',
                       'study_acquisition_month',
                       'study_acquisition_hour',
                       'case_open_weekday',
                       'case_open_month',
                       'case_open_hour',
                       'body_area',
                       'modality'
                      ]

# Creating binary integer dummy variables for each of the categorical variables
wait_time_df_dummies = pd.get_dummies(wait_time_df, columns=categorical_columns, dtype='int')
wait_time_df_dummies.head()


# <a id="Data_Exploration"></a>
# <div style="background-color: #161D3A; border: 1px solid #000; margin: 0 2px; padding: 2px 1px 2px 2px;">
#     <b><font size="4" color=#FA6400 face="sans-serif">Data Exploration</font></b>
# </div>

# <a id="Checking_Normality"></a>
# ### 5.1 Checking Normality

# **Observing wait time's (log transformed) central tendencies**

# In[16]:


# Creating a function that returns wait_time_minutes box plot and histogram

def box_hist_visuals(df, target_variable):
    box_fig = px.box(df, y=target_variable)

    # Box Plot--------------------------
    # Customizing the layout
    box_fig.update_layout(
        width=800,
        height=500,
        title='Box Plot of Wait Time',
        yaxis_title=f'{target_variable}',
        plot_bgcolor='white',
        font=dict(size=15)
    )
    
    box_fig.update_xaxes(
        title=dict(text=f'{target_variable}',
                   font=dict(size=18)),
        tickfont=dict(size=15),
        showgrid=True,
        gridwidth=1,
        gridcolor='rgba(200, 200, 200, 0.5)'
    )
    
    box_fig.update_yaxes(
        title=dict(text='Wait Time (minutes)', font=dict(size=18)),
        tickfont=dict(size=15),
        showgrid=True,
        gridwidth=1,
        gridcolor='rgba(200, 200, 200, 0.5)'
    )
    
    # Customizing the box colors
    box_fig.update_traces(marker=dict(color='#FA6400', line=dict(color='#FA6400', width=1)))

    # Histogram --------------------------
    # Using the sqrt rule to determine number of histogram bins
    hist_bins = int(np.sqrt(len(df))) 
    hist_fig = px.histogram(df, x=target_variable, nbins = hist_bins)
    hist_fig.update_layout(
        width=800,
        height=500,
        title='Histogram of Wait Time',
        plot_bgcolor='white',
        # paper_bgcolor='white',
        font=dict(size=15)
    )
    
    hist_fig.update_xaxes(
        title=dict(text=f'{target_variable}', font=dict(size=18)),
        tickfont=dict(size=15),
        showgrid=True,
        gridwidth=1,
        gridcolor='rgba(200, 200, 200, 0.5)'
    )
    
    hist_fig.update_yaxes(
        title=dict(text='Frequency', font=dict(size=18)),
        tickfont=dict(size=15),
        showgrid=True,
        gridwidth=1,
        gridcolor='rgba(200, 200, 200, 0.5)'
    )
    
    hist_fig.update_traces(marker=dict(color='#FA6400', line=dict(color='#FA6400', width=1)))
    
    # Displaying the visualizations
    box_fig.show()
    hist_fig.show()


# In[17]:


print("Mean:", wait_time_df['wait_time_minutes_log'].mean(), "minutes")
print("Median:", wait_time_df['wait_time_minutes_log'].median(), "minutes")
print("Mode:", wait_time_df['wait_time_minutes_log'].mode()[0], "minutes\n")
box_hist_visuals(wait_time_df, "wait_time_minutes_log")


# The histogram exhibits a bimodal distribution. I assume the mode will vary among the different aidoc_result True/False subgroups.

# **Observing wait time's (log transformed) central tendencies for False Aidoc Results**

# In[18]:


box_hist_visuals(wait_time_df[wait_time_df['aidoc_result']==False], "wait_time_minutes_log")# wait times for False results


# **Observing wait time's (log transformed) central tendencies for True Aidoc Results**

# In[19]:


box_hist_visuals(wait_time_df[wait_time_df['aidoc_result']==True], "wait_time_minutes_log")# wait times for True results


# The visualizations indicated that the mode behaviors for true and false results were very similar, suggesting that the aidoc_result is not the cause of the bimodal distribution. The Mean > Median > Mode relationship, box plot, and histogram indicates that the data is right-skewed (positively skewed), suggesting that the distribution is not normal. To confirm this, we will conduct a Kolmogorov-Smirnov test, which is ideal for our large dataset (N > 50).

# In[20]:


# Performing Kolmogorov-Smirnov to test for normality
stat, p = stats.kstest(wait_time_df['wait_time_minutes_log'], 'norm')
print('Kolmogorov-Smirnov Statistic:', stat)
print('p-value:', p)


# Given the very small p-value and large test statistic, we reject the null hypothesis. This indicates strong evidence against the data following a normal distribution. Therefore, we may want to use non-parametric tests on this data set moving forward.

# <a id="Data_Visualizations"></a>
# ### 5.2 Data Visualizations

# Preparing the data for visualization by sorting weekday and month categorical values in chronological order

# In[21]:


df = wait_time_df

# List of features to include in the dropdown list
features = ['algorithm',
            'patient_class',
            'aidoc_result',
            'study_acquisition_weekday',
            'study_acquisition_month',
            'study_acquisition_hour',
            'case_open_weekday',
            'case_open_month',
            'case_open_hour',
            'body_area',
            'modality'
            ]

# Assigning Week and Month order
week_order = {'Monday': 1, 'Tuesday': 2, 'Wednesday': 3, 'Thursday': 4, 'Friday': 5, 'Saturday': 6, 'Sunday': 7}
month_order = {'January': 1, 'February': 2, 'March': 3, 'April': 4, 'May': 5, 'June': 6,
               'July': 7, 'August': 8, 'September': 9, 'October': 10, 'November': 11, 'December': 12}

# Sorting weekday and months in chronological order
sorted_weekday = sorted(df['study_acquisition_weekday'].unique(), key=lambda x: week_order[x])
sorted_months = sorted(df['study_acquisition_month'].unique(), key=lambda x: month_order[x])


# **Bar plots of acquisition counts by different categorical features**

# In[22]:


# Create an initial bar chart
fig = go.Figure()

# Create the dropdown buttons
buttons = []

for feature in features:
    counts_df = df[feature].value_counts().reset_index()
    counts_df.columns = ['Category', f'{feature}_counts']
    
    fig.add_trace(go.Bar(x=counts_df['Category'], y=counts_df[f'{feature}_counts'], text=counts_df[f'{feature}_counts'], textposition='auto', name=feature, visible=True if feature == features[0] else False))
    
    if feature in ['study_acquisition_month', 'case_open_month']:
        sorted_cat_df = sorted_months
    elif feature in ['study_acquisition_weekday', 'case_open_weekday']:
        sorted_cat_df = sorted_weekday
    else:
        sorted_cat_df = sorted(df[feature].unique())
    
    buttons.append(dict(
    method='update',
    label=feature,
    args=[
        {'visible': [True if feature == f else False for f in features]},
        {'title': f"Acquisition Counts by {feature}",
         'xaxis': {'title': f"{feature}", 'categoryorder': 'array', 'categoryarray': sorted_cat_df},
         'yaxis': {'title': "Acquisition Counts"}}
        ])
    )

fig.update_layout(
    updatemenus=[dict(
        active=0,
        buttons=buttons,
        direction='down',
        showactive=True,
        font=dict(size=12),
        x=1,
        y=1.2)],
    width=1000,
    height=500,
    title=f"Acquisition Counts by {features[0]}",
    plot_bgcolor='white',
    font=dict(size=15)
)

fig.update_xaxes(
    title=dict(text=f"{features[0]}", font=dict(size=18)),
    tickfont=dict(size=15),
    showgrid=True,
    gridwidth=1,
    gridcolor='rgba(200, 200, 200, 0.5)'
)

fig.update_yaxes(
    title=dict(text='Counts', font=dict(size=18)),
    tickfont=dict(size=15),
    showgrid=True,
    gridwidth=1,
    gridcolor='rgba(200, 200, 200, 0.5)'
)

# Customize the bar colors
fig.update_traces(marker=dict(color='#FA6400', line=dict(color='#FA6400', width=1)))
fig.show()


# **Bar Plot Interpretations**
# 
# Algorithm
# - The pneumothorax, Intracranial hemorrhage , and Rib Fractures algorithm (ptx, ICH, Ribs) are the most utilized algorithms.
# - The intracranial vessel occlusion and Brain Aneurysm algorithms (VO, ba) are the least utilized algorithm.
# - It appears that the majority of cases that HealthyVibes handles are relate to instances of physical bodily trauma.
# 
# Patient Class
# - Most of the studies are from an Emergency setting.
# - The least number of studies are from an Outpatient setting.
# - Acute care settings like Emergency and Inpatient services make up the majority of the cases (~84.48%).
# - Ambulatory care like Outpatient services make up the minority of cases (15.52%)
# 
# Aidoc Result
# - There are significantly more studies flagged as "False" by the AI solution compared to "True" results.
# 
# Study Acquisition Weekday
# - Most of the acquisition studies occur during the weekday.
# - Weekends have the least amount of acquisitions (possibly due to lower patient volume).
# 
# Study Acquisition Month
# - Most of the acquisitions occur during May and June.
# - Mention something about low adoption in the inital stages, and then higher adoption later on.
# 
# Study Acquisition Hour
# - The least number of acquisitions occur between 3AM and 9AM.
# - The greatest number of acquisitions occur at 6PM.
# 
# 
# Based on these visuals, it appears that wait times may vary across different categories within the data set. To confirm these differences, we should run a non-parametric statistical test, like a Kruskal-Wallis test.

# **Checking average wait time per study acquisition hour**

# In[23]:


pd.DataFrame(df.groupby("study_acquisition_hour")[['wait_time_minutes','wait_time_minutes_log']].mean()).reset_index()


# It appears that the greatest wait time averages occur between 7AM and 12PM.

# **Checking average wait time per study acquisition hour per True/False result group**

# In[24]:


avg_results = pd.DataFrame(df.groupby(["aidoc_result", "study_acquisition_hour"])[['wait_time_minutes','wait_time_minutes_log']].mean()).reset_index()
avg_results.sort_values(by=["study_acquisition_hour"])


# We see shorter average wait times for cases flagged as True by the AI program in comparison to False results.

# **Box plots to observe distribution over different categorical groups**

# In[25]:


# Create an initial box plot
fig = go.Figure()

# Create the dropdown buttons
buttons = []

for feature in features:
    
    # Add box plot
    fig.add_trace(go.Box(x=df[feature], y=df['wait_time_minutes_log'], name=feature, visible=True if feature == features[0] else False))

    if feature in ['study_acquisition_month', 'case_open_month']:
        sorted_cat_df = sorted_months
    elif feature in ['study_acquisition_weekday', 'case_open_weekday']:
        sorted_cat_df = sorted_weekday
    else:
        sorted_cat_df = sorted(df[feature].unique())

    
    buttons.append(dict(
    method='update',
    label=feature,
    args=[
        {'visible': [True if feature == f else False for f in features]},
        {'title': f"Acquisitions Count by {feature}",
         'xaxis': {'title': f"{feature}", 'categoryorder': 'array', 'categoryarray': sorted_cat_df},
         'yaxis': {'title': "Log Scale of Wait Time (minutes)"}}
    ]
))

# Update layout with dropdown menu
fig.update_layout(
    updatemenus=[dict(
        active=0,
        buttons=buttons,
        direction='down',
        showactive=True,
        font=dict(size=12),
        x=1,
        y=1.18
    )],
    width=1000,
    height=600,
    title=f"Wait Time Distribution by {features[0]}",
    xaxis_title=f"{features[0]}",
    yaxis_title="Log Scale of Wait Time (minutes)",
    plot_bgcolor='white',
    font=dict(size=15)
)

# Customizing the x-axis
fig.update_xaxes(
    title=dict(text=f"{features[0]}",
               font=dict(size=18)),
    tickfont=dict(size=15),
    showgrid=True,
    gridwidth=1,
    gridcolor='rgba(200, 200, 200, 0.5)'
)

# Customize the y-axis
fig.update_yaxes(
    tickfont=dict(size=15),
    showgrid=True,
    gridwidth=1,
    gridcolor='rgba(200, 200, 200, 0.5)'
)
fig.update_traces(marker=dict(color='#FA6400', line=dict(color='#FA6400', width=1)))
fig.show()


# **Box Plot Interpretations**
# 
# Algorithm
# - The most variablilty in wait times are among the following algorithms:
#     - Pulmonary embolism algorithm (IPE)
#     - Malpositioning of Endothoracic Tubes (ETT)
#     - Pneumothorax (ptx)
# 
# - There is less variablilty in wait times amongst the following algorithms:
#     - Intracranial hemorrhage (ICH)
#     - Pulmonary embolism (PE)
#     - Intracranial vessel occlusion (VO)
#     - Aortic Dissection (ad)
#     - Brain Aneurysm (ba)
# 
# Patient Class
# - Emergency classes exhibit a smaller distribution of wait time.
# - Outpatient classes exhibit a larger distribution of wait time.
# 
# Aidoc Result
# - True results have fewer outliers.
# - Wait time variation amongst True results are slightly smaller than False results.
# 
# Study Acquisition Weekday
# - Weekends have lower wait time variability, and have the most outliers.
# 
# Study Acquisition Month
# - Greater variability in wait times during May and June.
# 
# Study Acquisition Hour
# - Increased wait time variability during 11AM to 1PM.
# 
# Modality
# - CT's have lower wait time variability in comparison to X-Rays.
# 
# Body Area
# - Chest related acquisitions have the greatest variability in wait time compared to Pulmonary Arteries and Brain.

# <a id="Statistical_Tests"></a>
# ### 5.3 Statistical Tests

# **Selecting Kruskal-Wallis as my choice for statistical testing**
# - It can be used to evaluate differences amongst categorical variables with two or more groups.
# - Ideal for skewed data distributions, like our wait time data.
# - H0, Null Hypothesis: The medians of each group are the same, meaning that all groups come from the same distribution.
# - Ha, Alternative Hypothesis: At least one of the groups has a different median, meaning at least one comes from a different distribution than the others.
# - If the calculated p-value is less than or equal to the specified significance level (typically 0.05), we would reject the null hypothesis. This suggests that there may be a different distribution among the groups.

# In[26]:


# Creating a function to perform a Kruskal Wallis test
def kruskal_cv_test(target_variable, categorical_feature):
    # Getting the unique categories from the categorical column
    categories = df[categorical_feature].unique()
    
    # Dictionary to store results
    results = {}
    
    # Using a for loop to perform Kruskal-Wallis test per category
    for category in categories:
        group_data = df[df[categorical_feature] == category][target_variable]
        stat, p_value = kruskal(*group_data)
        results[category] = {'statistic': stat, 'p-value': p_value}
    
    # Printing the results
    for category, result in results.items():
        print(f"Category: {category}, Statistic: {result['statistic']:.4f}, p-value: {result['p-value']:.4f}")

# Using a for loop to return the results of the Kruskal-Wallis Test per categorical feature
for ft in features:
    print(f"----------------------\nKruskal-Wallis Test for {ft}")
    kruskal_cv_test('wait_time_minutes_log', ft)


# **Kruskal-Wallis Test Interpretation**
# 
# Across all categorical variables, we observe a consistently large test statistic with p-values exceeding the significance level of 0.05. This leads us to reject the null hypothesis, indicating a statistically significant difference in wait times among each of the categorical groups. These findings suggest that the categorical variables analyzed have a notable impact on wait times, highlighting the variability in wait time outcomes across different categories.

# <a id="#Data_Modeling"></a>
# <div style="background-color: #161D3A; border: 1px solid #000; margin: 0 2px; padding: 2px 1px 2px 2px;">
#     <b><font size="4" color=#FA6400 face="sans-serif">Data Modeling</font></b>
# </div>

# **Lasso Regression**
# 
# I chose a Lasso regression modeling technique to determine the influences on wait time because:
# 
# - It is effective on skewed data.
# - It penalizes less important variables, which helps to highlight the more influential variables.
# - Its regularization technique and cross-validation helps to prevent the model from overfitting.
# - The simplicity of the model makes it easier to interpret the impact of each variable.

# In[27]:


X = wait_time_df_dummies.drop(columns=['accession', 'aidoc_site', 'wait_time_minutes', 'study_acquisition_time', 'case_open_time', 'wait_time_minutes_log'])
y = wait_time_df_dummies['wait_time_minutes_log']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the Lasso regression model
lasso = Lasso()

# Alpha parameters for cross validation (GridSearchCV)
param_grid = {'alpha': [0.001, 0.01, 0.1, 1.0, 10.0]}

# Performing GridSearchCV to find the best hyperparameters
grid_search = GridSearchCV(estimator=lasso, param_grid=param_grid, scoring='neg_mean_squared_error', cv=5)
grid_search.fit(X_train, y_train)

# Retrieve the best model and its hyperparameters
best_alpha = grid_search.best_params_['alpha']
best_lasso = grid_search.best_estimator_

# Determine which features are selected by the best model
selected_features = X.columns[best_lasso.coef_ != 0]
# print("Selected Features:")
# print(selected_features)

# Evaluate the best model on the test set
y_pred = best_lasso.predict(X_test)
mse = mean_squared_error(y_test, y_pred)

# Output results
print(f"Best alpha: {best_alpha}")
print(f"Mean Squared Error on test set: {mse}")

# Model with the best alpha value
lasso_model = Lasso(alpha=best_alpha)  # Set alpha value
lasso_model.fit(X, y)

# Get feature coefficients
feature_coefficients = pd.Series(lasso_model.coef_, index=X.columns)
sorted_coefficients = feature_coefficients.abs().sort_values(ascending=False)

print(f"\nMinimum coefficient: {sorted_coefficients.min()}\nMaximum coefficient: {sorted_coefficients.max()}")


# By using a 5-fold cross-validation method, I found that my best model operates with an alpha of 0.001. This small alpha choice indicates that the model behaves much like a simple linear regression, which makes it easier to understand how each feature affects wait time. The 5-fold cross-validation was a good choice, because it thoroughly tested the model's performance across various parts of the data. This ensured that the model was able to reliably predict wait times for new cases.

# **Plotting a Heat Map of the Correlation Coefficients provided by the best model**

# In[28]:


print("Number of features:", len(sorted_coefficients))
print("Number of features with a positive coefficient:", len(sorted_coefficients[sorted_coefficients > 0]))

plt.figure(figsize=(4, 18))
sns.heatmap(sorted_coefficients[sorted_coefficients > 0].to_frame(), cmap='Blues', annot=True)
plt.title('Correlation Coefficients Related to Wait Time')
plt.show()


# **Coorelation Heat Map Interpretation**
# 
# Variables with a strong positive correlation to Wait Time (log transformed):
# - Outpatient classes
# - Cases opened in June
# - Study acquisitions in June
# - Study acquisitions that took place between 7AM-10AM
# - Emergency classes
# 
# 
# Variables with a moderate positive correlation to Wait Time (log transformed):
# - CT modality
# - Cases opened on a Sunday
# - Study acquisitions at 3-6AM and 11AM
# - Cases opened at 1-2AM, and 10AM

# <a id="Take_Away"></a>
# <div style="background-color: #161D3A; border: 1px solid #000; margin: 0 2px; padding: 2px 1px 2px 2px;">
#     <b><font size="4" color=#FA6400 face="sans-serif">Take Aways</font></b>
# </div>

# Our AI solution does have an impact on the wait time for radiology studies, some of the relationships we see include:
# - Cases flagged with True results by the AI program on average have shorter wait times in comparison to False results.
# 
# Other interesting insights that may help HealthyVibes identify systemic bottlenecks within their business:
# - Longer wait times were correlated with study acquisitions that took place between 8 AM and 1 PM. This could be due to inefficient shift transitions or conflicting lunch break schedules. Identifying the root cause of these delays could potentially alleviate the extended wait times during this period.
# - A majority of the acquisitions occurred in the months of May and June. This suggests that warmer/summer months might see increased cases of injuries, possibly due to increased outdoor activities during the season. Anticipating this seasonal increase in patient volume could help HealthyVibes plan and allocate resources more effectively.

# <a id="Recommendations"></a>
# <div style="background-color: #161D3A; border: 1px solid #000; margin: 0 2px; padding: 2px 1px 2px 2px;">
#     <b><font size="4" color=#FA6400 face="sans-serif">Recommendations</font></b>
# </div>

# Analyses for another time
# - Effectiveness of Aidoc AI notification assistance by comparing wait time of groups with and without the use of our AI program.
# - Performance analysis of Aidoc results vs Radiologist diagnoses
# - Time Series analysis to observe trends and seasonality of medical emergencies

# <a id="Presentation"></a>
# <div style="background-color: #161D3A; border: 1px solid #000; margin: 0 2px; padding: 2px 1px 2px 2px;">
#     <b><font size="4" color=#FA6400 face="sans-serif">Presentation</font></b>
# </div>

# **Google Slides:** https://docs.google.com/presentation/d/1VSUzBYsay0b6tayVUxtQk8V0bKpOVX54peRpLLGVaBY/edit?usp=sharing
# 
# **GitHub:** https://github.com/ntran22/HealthyVibes

# <a id="Workes_Cited"></a>
# <div style="background-color: #161D3A; border: 1px solid #000; margin: 0 2px; padding: 2px 1px 2px 2px;">
#     <b><font size="4" color=#FA6400 face="sans-serif">Works Cited</font></b>
# </div>

# **Logo Colors**
# - https://brandfetch.com/aidoc.com?view=library&library=default&collection=colors
# 
# **How to extract string after a specific substring**
# - https://stackoverflow.com/questions/12572362/how-to-get-a-string-after-a-specific-substring
# 
# **Statistical test for normality**
# - https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6350423/#:~:text=The%20Shapiro%E2%80%93Wilk%20test%20is,used%20for%20n%20%E2%89%A550.
# 
# **Converting date time object to a date time type**
# - https://stackoverflow.com/questions/33957720/how-to-convert-column-with-dtype-as-object-to-string-in-pandas-dataframe
# - https://stackoverflow.com/questions/38067704/how-to-change-the-datetime-format-in-pandas
# 
# **Dummy Variables**
# - https://stackoverflow.com/questions/71705240/how-to-convert-boolean-column-to-0-and-1-by-using-pd-get-dummies
# 
# **Plotly**
# - Boxplots: https://plotly.com/python/box-plots/
# - Histograms: https://www.datacamp.com/tutorial/create-histogram-plotly
# 
# **Checking if missing values occur at the same indices**
# - https://www.geeksforgeeks.org/python-check-if-two-lists-are-identical/#
# 
# **Lasso Regressions**
# - https://www.linkedin.com/pulse/lasso-vs-ridge-regression-feature-selection-mouhssine-akkouh-diiye/
# - https://www.analyticsvidhya.com/blog/2016/01/ridge-lasso-regression-python-complete-tutorial/
# - https://medium.com/@daython3/mastering-the-art-of-feature-selection-python-techniques-for-visualizing-feature-importance-cacf406e6b71
# 
# **Kruskal-Wallis**
# - https://www.statology.org/kruskal-wallis-test-python/
# 
