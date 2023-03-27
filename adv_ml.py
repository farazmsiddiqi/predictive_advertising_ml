import pandas as pd
import matplotlib.pyplot as plt

# load dataset
data = pd.read_csv('KAG_conversion_data.csv')

# calculate correlation coefficients
corr_clicks = data[['Clicks', 'Total_Conversion']].corr(method='pearson').iloc[0,1]
corr_impressions = data[['Impressions', 'Total_Conversion']].corr(method='pearson').iloc[0,1]

# print the correlation coefficients
print('Correlation coefficient (Clicks, Total_Conversion):', corr_clicks)
print('Correlation coefficient (Impressions, Total_Conversion):', corr_impressions)

print(data['Clicks']) 

# plt.plot(data['Clicks'])
plt.plot(data['Impressions'].values)
plt.plot(data['Total_Conversion'].values)
plt.show()
