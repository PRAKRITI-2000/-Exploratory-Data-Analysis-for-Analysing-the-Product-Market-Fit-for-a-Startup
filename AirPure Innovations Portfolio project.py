#%%
import pandas as pd
import numpy as np
import datetime as dt
import os
import matplotlib.pyplot as plt
from jupyter_server.nbconvert.handlers import date_format
from pandas import concat
#%%
dataset_folder = os.path.expanduser(r'C:\Users\Lenovo\OneDrive\Desktop\Codebasics Challenge')
csv_files=[f for f in os.listdir(dataset_folder) if f.endswith('.csv')]
dataframes=[]
for csv_file in csv_files:
    file_path=os.path.join(dataset_folder,csv_file)
    df=pd.read_csv(file_path)
    dataframes.append(df)
    print(f"{file_path}:{df.shape}")
#%%
aqi= dataframes[0]
population_projection= dataframes[1]
vahan=dataframes[2]
#%%
aqi.head(3)
## This dataset tells us about the air quality index of the states and the different cities in the states on specified dates
aqi.info()
#%%
## Data Cleaning and Data Pre-processing for the dataset aqi
## Checking for the Inconsistent Datatypes and Un-necessary columns
aqi['date']= pd.to_datetime(aqi['date'],dayfirst=True)
aqi.head(3)
#%%
### Exploratory Data Analysis for AQI dataset
aqi['aqi_value'].describe()
## Here we need to inspect the possible occurences of outliers in the aqi_value for which we have to first check the nature of graphical distribution.
## Here we can observe that when AQI=500 it denotes the severity of air pollution. These values are not outliers
aqi['year_month'] = aqi['date'].dt.strftime('%Y-%m')
agg_aqi_table = aqi.groupby('year_month').agg({'aqi_value': 'mean'}).reset_index().sort_values(by='year_month')
agg_aqi_table
# Plot
%matplotlib inline
plt.figure(figsize=(12, 8))
plt.plot(agg_aqi_table['year_month'], agg_aqi_table['aqi_value'], marker='o')
plt.xlabel('Year-Month')
plt.ylabel('Average AQI')
plt.title('Monthly Average AQI Values')
plt.xticks(rotation=90)
plt.tight_layout()
plt.grid()
plt.show()
#%% md
# ## From the above graph we can observe that
# 
# ## OBSERVATION 1
# The AQI value follows a seasonal pattern of increase.
# The AQI value rises rapidly from September (9th Month) to December (12th Month). These months in India
# coincide with the major festive season. People travel frequently, and pollution-causing factors like firecrackers,
# natural decomposition, and religious activities contribute to increased burning activities. This, in turn, leads to a rise in the severity of AQI.
# 
# ## OBSERVATION 2
# The AQI value shows a noticeable decrease from April (4th Month) to July (7th Month)
# This period coincides with the onset of summer and the beginning of the monsoon in India. During these months,
# increased temperatures and strong winds help disperse pollutants more effectively. Additionally, the arrival of monsoon rains significantly reduces airborne pollutants by washing away dust, soot, and other particulates.
# There are also fewer pollution-inducing events such as festivals or stubble burning. All these factors together
# contribute to a decline in AQI severity during this time.
#%% md
# ## Now we shall inspect the days when the AQI crossed the maximum values as well as the days when AQI was relatively less.
#%%
aqi[(aqi['aqi_value']==max(aqi['aqi_value']))|(aqi['aqi_value']==min(aqi['aqi_value']))]
#%% md
# ## OBSERVATION 3
# The AQI was utterly bad because of two prominent pollutants: PM2.5 and PM10.
#%% md
# ## Finding the most prevalent pollutants that are present throughout the year.
#%%
grouped_pollutants= aqi.groupby(['prominent_pollutants']).agg(frequency=('prominent_pollutants', 'count')).reset_index()
grouped_pollutants= grouped_pollutants.sort_values(by='frequency',ascending=False).reset_index(drop=True)
grouped_pollutants
## Here we can observe that the prominent pollutants are single as well as in pairs. We have to split the observations and find the final frequency
#%% md
# ## Assumption:
# We assume that on days when the pollutants are more than one, they equally contribute to the AQI conditions that day. For example if there are two pollutants in AQI on a given day, then each one of them contributes 50%. Likewise if there are three pollutants present on one day, then each contribute 33,33% to the total aqi that day.
#%%
# Count how many pollutants are listed on each row
grouped_pollutants['total_pollutant_count'] = grouped_pollutants['prominent_pollutants'].str.split(',').str.len()
grouped_pollutants['prominent_pollutants'] = grouped_pollutants['prominent_pollutants'].str.split(',')
grouped_pollutants = grouped_pollutants.explode('prominent_pollutants')
grouped_pollutants['actual_frequency']= round(grouped_pollutants['frequency']/grouped_pollutants['total_pollutant_count'],0).astype('int')
final_grouped_pollutants= grouped_pollutants[['prominent_pollutants','actual_frequency']]
final_grouped_pollutants
#%% md
# Logic behind the above code: We want to create different rows based on the number of pollutants present in one records. If there are three pollutants in a records, we would want three rows with the exact same columns. The column entries should be replicated as it is. Hence, we have used the function explode().
#%% md
# ## Analyzing the Key Pollutants
# In this step, we will identify the pollutants that are most frequently present in the AQI readings and should definitely be targeted by air purifiers.
#%%
final_grouped_pollutants.groupby(['prominent_pollutants']).agg({'actual_frequency':'sum'}).sort_values(by='actual_frequency',ascending=False).reset_index()
#%% md
# ## OBSERVATION 4
# The key pollutants that Airpure Innovations must address through their air purifiers are PM10, PM2.5, O₃, CO, SO₂, and NO₂.
# 
# To ensure maximum effectiveness, the purifiers should be designed as an integrated system capable of targeting all these major pollutants.
# 
# This comprehensive approach can also serve as a distinctive selling point for the company, highlighting its ability to tackle nearly all critical air quality concerns with a single solution.
#%%
## Observing the Relationship Between the Number of Monitoring Stations and AQI Values in India

## We aim to investigate whether a higher number of monitoring stations contributes to lower AQI values, thereby ## indicating better air quality management and awareness.
#%%
## Here we will find the strength of covariance from here
print(round(aqi['number_of_monitoring_stations'].corr(aqi['aqi_value']),2))
#%% md
# ## OBSERVATION 5
# From the strength of covariance, we can observe that 0.08 is close to 0 indicating that there is feeble or no relationship between the number of monitoring stations and the aqi_value. Because the cities are crowded, having a few monitoring units doesn't solve the pupose at large. They can't predict the AQI levels.
#%% md
# ## Finding the cities where AQI levels are poor, very poor and severe year to year.
#%%
aqi['air_quality_status'].unique()
#%%
print(aqi['year_month'].min())
print(aqi['year_month'].max())
#%%
aqi.head(5)
#%%
state_analysis= aqi.groupby(['state','air_quality_status']).agg(frequency=('prominent_pollutants', 'count')).reset_index()

target_data = state_analysis[(state_analysis['air_quality_status']=="Poor")|(state_analysis['air_quality_status']=="Very Poor")|(state_analysis['air_quality_status']=="Severe")].sort_values(by='frequency',ascending=False).reset_index(drop=True).groupby(['state']).agg(total_frequency=('frequency','sum')).sort_values(by='total_frequency',ascending=False).reset_index()

target_data

#%% md
# ## OBSERVATION 6
# Based on the table shown, we observe that Northern India is a good market to start with.
# 
# Bihar, Haryana, Uttar Pradesh, and Rajasthan are the states where air pollution levels are a major concern.
# 
# These states experience varying climatic conditions throughout the year. They are also prominent centers of cultural and religious heritage. So pollution is a persisting issue here. Hence, these states would be ideal to test the performance of air purifiers.
#%% md
# ## Now we want to see what are the cities which are highly polluted as per the data provided
#%%
state_analysis_1= aqi.groupby(['state','area','air_quality_status']).agg(frequency=('prominent_pollutants', 'count')).reset_index()

target_data_1 = state_analysis_1[(state_analysis_1['air_quality_status']=="Poor")|(state_analysis_1['air_quality_status']=="Very Poor")|(state_analysis_1['air_quality_status']=="Severe")].sort_values(by='frequency',ascending=False).reset_index(drop=True).groupby(['state','area']).agg(total_frequency=('frequency','sum')).sort_values(by='total_frequency',ascending=False).reset_index()

target_data_1
#%% md
# ## OBSERVATION 7
# 
# 1) Delhi NCR includes some of the most polluted urban cities in Delhi and Uttar Pradesh, indicating a potentially high demand for air purifiers in this region.
# 
# 2) The population in Delhi NCR largely comprises upper-middle and upper-class segments. This makes it easier to create awareness, as these groups are more likely to resonate with the problem statement of air pollution.
# 
# 3) Additionally, due to their socio-economic status, people in this region are more health-conscious, making them an ideal audience to pilot and test the effectiveness of air purifiers.
#%%
aqi['unit'].unique()
#%% md
# ## Analysis of the Impact of Vehicles and the Fuel on AQI
#%%
vahan
#%%
impact_of_vehicles = vahan.groupby(['vehicle_class'])['value'].mean().reset_index()
impact_of_vehicles.rename(columns={'value': 'average_aqi_emitted'}, inplace=True)
impact_of_vehicles = impact_of_vehicles.sort_values(by='average_aqi_emitted', ascending=False).head(10).reset_index(drop=True)
impact_of_vehicles
#%%
impact_of_vehicles_and_fuels = vahan.groupby(['vehicle_class','fuel'])['value'].mean().reset_index()
impact_of_vehicles_and_fuels.rename(columns={'value': 'average_aqi_emitted'}, inplace=True)
impact_of_vehicles_and_fuels = impact_of_vehicles_and_fuels.sort_values(by='average_aqi_emitted', ascending=False).reset_index()
refined_results_1 = impact_of_vehicles_and_fuels[impact_of_vehicles_and_fuels['vehicle_class'].isin(impact_of_vehicles['vehicle_class'])].reset_index(drop=True)

refined_results_1['fuel'].value_counts()
## Here we are analyzing what is the fuel primarily used by the top 10 pollution emitting vehicles as fetched in the impact_of_vehicles_dataset. We have to find out which type of fuels are the major problems behind the high AQI
#%%
## Classification of the vehicle types as: One wheeler, two-wheeler, three-wheeler, or four-wheeler

def classify_vehicle_type(vehicle):
    vehicle = vehicle.upper()
    if any(kw in vehicle for kw in ['MOPED', 'M-CYCLE', 'MOTOR CYCLE', 'SCOOTER', 'MOTORISED CYCLE', 'TWO WHEELER', 'MOTOR CYCLE/SCOOTER-WITH']):
        return 'Two-Wheeler'
    elif 'THREE WHEELER' in vehicle or 'E-RICKSHAW' in vehicle:
        return 'Three-Wheeler'
    else:
        return 'Four-Wheeler or More'

# Apply the classification
vahan['vehicle_type'] = vahan['vehicle_class'].apply(classify_vehicle_type)
vahan
#%%
vahan.groupby(['vehicle_type']).agg(mean_aqi=('value', 'mean'),total_frequency_usage=('vehicle_type', 'count')).reset_index()
#%% md
# ## OBSERVATION 8
# 
# 1) We can observe that the vehicles contributing to higher average AQI values are: ["M-CYCLE/SCOOTER","AGRICULTURAL TRACTOR", "MOTOR CAR", "E-RICKSHAW(P)", "MOPED", "GOODS CARRIER", "MOTORISED CYCLE (CC > 25CC)", "TRAILER (AGRICULTURAL)", "THREE WHEELER (PASSENGER)","TRACTOR (COMMERCIAL)"]
# 
# 2) Two-wheelers contribute the highest in pollution, followed by three-wheelers, and four-wheelers. Even though the usage of the two whellers is less, but is is contributing the highest to the pollution patterns.
# 
# 3) Fuels contributing to higher AQI levels (in descending order) are: Petrol, Electric (BOV), Pure EV, Ethanol, and CNG. These are the fules irrespective of beign used in any type of vehicle class, that are contributing to the pollutions patterns.
#%% md
# ## Analyzing the YOY Growth in the average of AQI
#%%
aqi['date'].dt.year.unique() ## Here we have the dataset of four years: 2025, 2024, 2023, 2022
#%%
# Using both DataFrames: aqi_summary and population_projection
aqi_summary = aqi.groupby(aqi['date'].dt.year).agg(mean_aqi=('aqi_value', 'mean')).reset_index()
aqi_summary.rename(columns={'date': 'year'}, inplace=True)
aqi_summary
#%%
refined_population_data = population_projection[population_projection['gender']=='Total'].reset_index(drop=True)
refined_population_data_summary = refined_population_data.groupby(['year','state'])['value'].sum().reset_index()

refined_population_data_summary.rename(columns={'value':'total_population_in_thousands'}, inplace=True)
refined_population_data_summary= refined_population_data_summary[refined_population_data_summary['state']!="All India"]
refined_population_data_summary
#%%
# Step 1: Extract year and create a new column
aqi['year'] = aqi['date'].dt.year
states_dealing_with_pollution = aqi.groupby(['state', 'year'])['aqi_value'].mean().reset_index()
states_dealing_with_pollution.rename(columns={'aqi_value': 'avg_aqi_value'}, inplace=True)
pivoted_data = states_dealing_with_pollution.pivot(index='state', columns='year', values='avg_aqi_value')
pivoted_data.fillna(0,inplace=True)
pivoted_data
#%%
pivoted_data[(pivoted_data[2023]>pivoted_data[2022]) & (pivoted_data[2024]>pivoted_data[2023]) & (pivoted_data[2025]>pivoted_data[2024])]
#%% md
# ## OBSERVATION 9
# 
# States that are persistently suffering from the pollution problem are: Arunachal Pradesh, Chhattisgarh, Delhi, Meghalaya, Nagaland and West Bengal.
#%% md
# ## The states where the market is prominent for the company are: Bihar, Haryana, Delhi, Uttar Pradesh, and Rajasthan
#%%
population_figures= population_projection[(population_projection['gender']=='Total') & (population_projection['state']!="All India") & (population_projection['state'].isin(['Delhi','Haryana','Uttar Pradesh','Bihar','Rajasthan']))][['year','state','value','unit']].reset_index(drop=True)

population_figures['value']=population_figures['value']*1000
population_figures=population_figures.drop_duplicates(subset=['state', 'year'])
population_figures= population_figures.sort_values(by=['state','year'],ascending=True).reset_index(drop=True)
population_figures=population_figures[['year','state','value']].rename(columns={'value':'total_population'})
population_figures
#%%
population_figures['previous_year_population']= population_figures.groupby('state')['total_population'].shift(1).fillna(0).astype('int')
yoy_growth_data= population_figures[population_figures['previous_year_population']>0].reset_index(drop=True)
yoy_growth_data['yoy%']= (yoy_growth_data['total_population']-yoy_growth_data['previous_year_population'])*100/yoy_growth_data['previous_year_population']

yoy_growth_data.groupby('state')['yoy%'].mean()
#%% md
# ## OBSERVATION 10
# 
# ## 1) HARYANA
# 
# The state with the highest YoY Growth in population.
# 
# The state exhibits rapid urban expansion, particularly in cities like Gurugram and Faridabad.
# 
# It also experiences high migration inflow due to increasing employment opportunities and robust infrastructure.
# 
# The growing population presents a fast-expanding consumer base, making Haryana a strategic testing ground for AirPure Innovations.
# 
# The state offers potential first-mover advantages, and there is a significant B2C market driven by urban demand and health-conscious consumers.
# 
# The state is also proactive about tackling the pollution causes, and hence further opens doors for collaboration with the government.
#%% md
# ## 2) BIHAR
# 
# Bihar faces challenges such as unemployment, high birth rates, and lower literacy levels, with a majority of the population leading a modest lifestyle.
# 
# As such, the state may not currently be the ideal launchpad for Air Pure Innovations in a traditional B2C model.
# 
# However, there is potential for growth through B2B partnerships or government-backed initiatives, especially in public health and infrastructure segments.
#%% md
# ## 3) RAJASTHAN
# 
# Rajasthan is a state that suffers from both air pollution and harsh summer climates. Pollution caused by sand and other particulate matter is quite prominent here, making it a suitable region for the introduction of air purifiers.
# 
#  Additionally, Rajasthan has a significant section of middle, upper-middle, and affluent populations. The state is also a popular destination for vacations and destination weddings and holds architectural and cultural significance, attracting a steady influx of tourists.
# 
# AirPure Innovations can scale their air purifiers at both B2C and B2B levels by collaborating with hospitality businesses such as hotels and resorts, offering them the opportunity to test and adopt their products.
#%% md
# ## 4) DELHI
# 
# Delhi exhibits characteristics similar to Haryana, with a high concentration of middle, upper-middle, and affluent segments.
# 
# The population here tends to prioritize health and quality of life, making it receptive to air purification products. Scalability in Delhi depends heavily on mass awareness and strategic positioning.
# 
# There is strong potential for both B2C growth and collaborations with government and institutions (B2B/Govt models), especially in public spaces and urban infrastructure.
#%% md
# 