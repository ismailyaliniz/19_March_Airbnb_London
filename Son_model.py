import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from datetime import date
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, RobustScaler

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 500)


from sklearn.linear_model import LinearRegression

from sklearn.metrics import classification_report
from sklearn.metrics import mean_squared_error

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate
from sklearn.model_selection import GridSearchCV

#Warning ignore
import warnings
warnings.filterwarnings("ignore")

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)


pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 500)
pd.options.display.float_format = '{:.2f}'.format


from conda_env.installers import pip

#!pip install pandas openpyxl

data = pd.read_excel('Machine Learning/Final Projesi/son_model/3_Airbnb_Data_March_2024.xlsx')  # Update 'YourSheetName' with the name of your sheet

"""
def missing_values_table(dataframe, na_name=False):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]

    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])
    print(missing_df, end="\n")

    if na_name:
        return na_columns

missing_values_table(data)
"""
#####################
Missing Values
#####################
def missing_values_table(dataframe, na_name=False):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]

    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])
    print(missing_df, end="\n")

    if na_name:
        return na_columns

    return missing_df

# Get the missing values table
missing_df = missing_values_table(data)

# Plotting
plt.figure(figsize=(12, 8))
sns.barplot(x=missing_df.index, y='n_miss', data=missing_df, palette="viridis")
plt.xticks(rotation=45, ha='right')
plt.xlabel('Columns with Missing Values')
plt.ylabel('Number of Missing Values')
plt.title('Number of Missing Values per Column')
plt.show()

# Rotate the x-axis labels
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)  # Ensure y-axis labels are horizontal

############
############

# Drop columns with 100% missing values
data.drop(['calendar_updated', 'license', 'neighbourhood_group_cleansed'], axis=1, inplace=True)

#Dropping the host_neighbourhood, neighbourhood, and host_location columns
# as their high redundancy and the general nature of the values.
#This can help streamline our dataset and
#we can focus on more impactful variables for our analysis and model.

# Drop the columns from the DataFrame
data.drop(['host_neighbourhood','host_location'], axis=1, inplace=True)

# Verify the columns have been removed
data.info()

print("Shape of the dataset:", data.shape)
# Recoding 'host_is_superhost' where 't' becomes 1 and everything else becomes 0
data['host_is_superhost'] = data['host_is_superhost'].apply(lambda x: 1 if x == 't' else 0)

# Check the changes
print(data['host_is_superhost'].value_counts())

######################################
# Convert the 'price' column to numeric if it's not already
data['price'] = pd.to_numeric(data['price'].replace('[\$,]', '', regex=True), errors='coerce')

# Drop rows where 'price' is NaN after conversion
data = data.dropna(subset=['price'])

######################################
##########################
# Steps to Correct latitude and longitude Values
# Identify non-numeric values originally in the data
# Check for non-numeric entries
non_numeric_latitude = data['latitude'].isna().sum()
non_numeric_longitude = data['longitude'].isna().sum()

# Check for values outside the valid ranges for London
invalid_latitude = ((data['latitude'] < -90) | (data['latitude'] > 90)).sum()
invalid_longitude = ((data['longitude'] < -180) | (data['longitude'] > 180)).sum()

# Count null values in the dataset
null_latitude = data['latitude'].isnull().sum()
null_longitude = data['longitude'].isnull().sum()

# Correct latitude values
data.loc[(data['latitude'] > 52), 'latitude'] = data['latitude'] / 1000

# Correct longitude values
data.loc[(data['longitude'] < -1) | (data['longitude'] > 1), 'longitude'] = data['longitude'] / 1000

# Check corrections by finding any remaining out-of-range values
remaining_invalid_latitude = ((data['latitude'] < 51) | (data['latitude'] > 522)).sum()
remaining_invalid_longitude = ((data['longitude'] < -1) | (data['longitude'] > 1)).sum()

# Display the results to verify corrections
print(f"Non-numeric latitude entries: {non_numeric_latitude}")
print(f"Latitude values outside [51.3, 51.7]: {invalid_latitude}")
print(f"Latitude null values: {null_latitude}")

print(f"Non-numeric longitude entries: {non_numeric_longitude}")
print(f"Longitude values outside [-0.5, 0.3]: {invalid_longitude}")
print(f"Longitude null values: {null_longitude}")

print(f"Remaining invalid latitude values: {remaining_invalid_latitude}")
print(f"Remaining invalid longitude values: {remaining_invalid_longitude}")
data['latitude'].describe()
#####
#########################################
# Mock data for visualization
latitude_out_of_range = 495
longitude_out_of_range = 187
non_numeric_latitude = 0
non_numeric_longitude = 0
null_latitude = 0
null_longitude = 0

# Data for bar plot
issues = {
    'Non-numeric Latitude': non_numeric_latitude,
    'Latitude out of range': latitude_out_of_range,
    'Latitude null values': null_latitude,
    'Non-numeric Longitude': non_numeric_longitude,
    'Longitude out of range': longitude_out_of_range,
    'Longitude null values': null_longitude
}

issues_df = pd.DataFrame(list(issues.items()), columns=['Issue', 'Count'])

# Bar plot for issues
plt.figure(figsize=(10, 6))
sns.barplot(x='Issue', y='Count', data=issues_df, palette="viridis")
plt.xticks(rotation=45, ha='right')
plt.xlabel('Issue Type')
plt.ylabel('Count')
plt.title('Count of Issues in Latitude and Longitude Columns')
plt.show()

# Scatter plot for valid and out-of-range values
# Mock data for the sake of visualization
np.random.seed(42)
data = pd.DataFrame({
    'latitude': np.random.uniform(51.2, 51.8, 1000),
    'longitude': np.random.uniform(-0.6, 0.4, 1000)
})

valid_latitude = data[(data['latitude'] >= 51.3) & (data['latitude'] <= 51.7)]
invalid_latitude = data[(data['latitude'] < 51.3) | (data['latitude'] > 51.7)]
valid_longitude = data[(data['longitude'] >= -0.5) & (data['longitude'] <= 0.3)]
invalid_longitude = data[(data['longitude'] < -0.5) | (data['longitude'] > 0.3)]

# Scatter plot for Latitude
plt.figure(figsize=(10, 6))
plt.scatter(valid_latitude['longitude'], valid_latitude['latitude'], label='Valid Latitude', alpha=0.6)
plt.scatter(invalid_latitude['longitude'], invalid_latitude['latitude'], color='red', label='Invalid Latitude', alpha=0.6)
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('Scatter Plot of Latitude and Longitude')
plt.legend()
plt.show()

# Scatter plot for Longitude
plt.figure(figsize=(10, 6))
plt.scatter(valid_longitude['longitude'], valid_longitude['latitude'], label='Valid Longitude', alpha=0.6)
plt.scatter(invalid_longitude['longitude'], invalid_longitude['latitude'], color='red', label='Invalid Longitude', alpha=0.6)
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('Scatter Plot of Latitude and Longitude')
plt.legend()
plt.show()
##########################################
#Calculating center of the London with the coordinates of the top attractions
#########################################

#visualization of the map

import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.io.img_tiles as cimgt
from geopy.distance import geodesic

# Function to calculate the center and radius of smallest enclosing circle
def smallest_enclosing_circle(points):
    latitudes, longitudes = zip(*points)
    centroid = (sum(latitudes) / len(points), sum(longitudes) / len(points))
    radius = max(geodesic(centroid, point).meters for point in points)
    return centroid, radius

# Coordinates of the attractions in London
points = [
    (51.5014, -0.1419),  # Buckingham Palace
    (51.5194, -0.1270),  # The British Museum
    (51.4995, -0.1246),  # The Houses of Parliament
    (51.5081, -0.0759),  # The Tower of London
    (51.5080, -0.1281),  # Trafalgar Square
    (51.5033, -0.1196),  # The London Eye
    (51.5138, -0.0984),  # St. Paul’s Cathedral
    (51.5076, -0.0994),  # The Tate Modern
    (51.5118, -0.1233),  # Covent Garden
    (51.5101, -0.1346)   # Piccadilly Circus
]

center, radius_meters = smallest_enclosing_circle(points)

# Radius in degrees (approximation)
# Average degrees per meter at latitude for London (~51 degrees North)
degrees_per_meter = 1 / 111320
radius_degrees = radius_meters * degrees_per_meter

# Create a plot with a map background
fig, ax = plt.subplots(figsize=(10, 10), subplot_kw={'projection': ccrs.PlateCarree()})
ax.set_extent([-0.16, -0.08, 51.48, 51.54], crs=ccrs.PlateCarree())

# Add a map tile service using StadiaMapsTiles with an API key
api_key = '031d8dc5-f899-48ea-a407-ba11ad6f8421'  # Stadia Maps API key
stadia_maps = cimgt.StadiaMapsTiles(apikey=api_key, style='outdoors')
ax.add_image(stadia_maps, 12)

# Add a red pinpoint at the center
ax.plot(center[1], center[0], marker='o', color='red', markersize=10, alpha=1, transform=ccrs.Geodetic())

# Plot all other points with a different color
for point in points:
    ax.plot(point[1], point[0], marker='o', color='blue', markersize=5, alpha=1, transform=ccrs.Geodetic())

# Add a circle to enclose all points
circle = plt.Circle((center[1], center[0]), radius_degrees, color='green', fill=False, transform=ccrs.Geodetic(), linewidth=2, linestyle='dotted')
ax.add_patch(circle)

plt.title('Central London with Marked Center and Attractions')
plt.show(block=True)

###############################
#distance_to_center_km column
###############################

import pandas as pd
from geopy.distance import geodesic

# Verify and clean data
data['latitude'] = pd.to_numeric(data['latitude'], errors='coerce')
data['longitude'] = pd.to_numeric(data['longitude'], errors='coerce')
data.dropna(subset=['latitude', 'longitude'], inplace=True)  # Drop rows with invalid data

# Define the center of the smallest enclosing circle
center_point = (51.5083, -0.11728000000000002)

# Function to calculate the distance to the center point
def calculate_distance(row):
    data_location = (row['latitude'], row['longitude'])
    # Ensure latitude is within the valid range
    if -90 <= row['latitude'] <= 90 and -180 <= row['longitude'] <= 180:
        return geodesic(data_location, center_point).km
    else:
        return None  # Return None if coordinates are out of bounds

# Apply the function to each row in the DataFrame
data['distance_to_center_km'] = data.apply(calculate_distance, axis=1)

# Display the first 5 rows of the 'distance_to_center_km' column
print(data[['distance_to_center_km']].head())

####################################
#missing values

# Calculate the number of missing values in each column
missing_values = data.isnull().sum()

# Filter to show only columns where there are missing values
missing_values = missing_values[missing_values > 0]

# Display the columns with missing values and their counts
print("Columns with Missing Values:")
print(missing_values)

####################################
data.isnull().sum()
# Drop the columns from the DataFrame
data.drop(['description', 'host_response_time', 'neighbourhood', 'host_response_rate','host_acceptance_rate','bathrooms_text'], axis=1, inplace=True)
#'description', 'host_response_time', 'neighbourhood', 'host_response_rate','host_acceptance_rate','bathrooms_text' 
####################################

# Drop rows with missing values in the specified columns
data = data.dropna(subset=['bathrooms','bedrooms', 'beds'])
#high correlation
# Verify that the rows with missing values have been dropped
print(data.isnull().sum())  # This will show the count of missing values in each column
print(data.shape)  # This will show the shape of the DataFrame after dropping the rows
#############################
# Identify rows where 'beds' exceed 'accommodates'
anomaly_beds = data[data['beds'] > data['accommodates']]
# Identify rows where 'bathrooms' are unreasonably high given 'accommodates'
# This rule can be customized based on what you consider unreasonable
anomaly_bathrooms = data[data['bathrooms'] > data['accommodates'] * 2]  # Example: more than twice the accommodates
# Combine both anomalies
anomalies = pd.concat([anomaly_beds, anomaly_bathrooms]).drop_duplicates()
# Display the identified anomalies
#print(anomalies[['price', 'accommodates', 'bathrooms', 'bedrooms', 'beds']])

# If you want to handle these anomalies (e.g., by removing them), you can do so:
# Remove the anomalies from the dataset
data_cleaned = data.drop(anomalies.index)

# Display the shape of the cleaned dataset to confirm removal
print(data_cleaned.shape)
anomalies.shape
#######################################
#######################################

import pandas as pd
# Convert 'number_of_reviews_ltm' to string to avoid FutureWarning
data['number_of_reviews_ltm'] = data['number_of_reviews_ltm'].astype('object')

# Fill NaN values with 'no reviews'
data['number_of_reviews_ltm'].fillna('no reviews', inplace=True)

# Define the categorization function
def categorie(number):
    if number == 'no reviews':
        return number
    elif 0 <= number < 1:
        return '0'
    elif 1 <= number < 2:
        return '1'
    elif 2 <= number < 3:
        return '2'
    elif 3 <= number < 4:
        return '3'
    elif 4 <= number < 5:
        return '4'
    elif 5 <= number < 6:
        return '5'
    elif 6 <= number < 16:
        return '6-15'
    elif 16 <= number < 31:
        return '16-30'
    else:
        return '31_or_more'

# Apply the categorization function to the 'review_scores_rating' column
data['number_of_reviews_ltm_cat'] = data['number_of_reviews_ltm'].apply(categorie)
review_counts = data['number_of_reviews_ltm_cat'].value_counts()

# Display the result
print("Review Counts by Category:")
print(review_counts)
#####################
# Drop rows where 'number_of_reviews_ltm_cat' is '0' and '1'
data = data[~data['number_of_reviews_ltm_cat'].isin(['0', '1'])]

# Verify the changes
review_counts_after_dropping = data['number_of_reviews_ltm_cat'].value_counts()
print("Review Counts by Category after dropping '0' reviews:")
print(review_counts_after_dropping)
####################
# Get the counts and sort them
sorted_counts = data['number_of_reviews_ltm_cat'].value_counts().sort_values(ascending=False)

# Plot the sorted countplot
plt.figure(figsize=(12, 6))
sns.countplot(data=data, x='number_of_reviews_ltm_cat', palette='viridis', order=sorted_counts.index)
plt.title('Number of Reviews in the Last Twelve Months by Category')
plt.xlabel('Category')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.show()
####################

#print(f"\nTotal Number of Reviews: {num_reviews}")
#print(f"Highest Number of Reviews: {highest_reviews}")
#############################################
# List of columns to apply the categorization
review_score_columns = [
    'review_scores_rating', 'review_scores_accuracy', 'review_scores_cleanliness',
    'review_scores_checkin', 'review_scores_communication', 'review_scores_location',
    'review_scores_value'
]

# Convert columns to string to avoid FutureWarning and fill NaN values with 'no reviews'
for column in review_score_columns:
    data[column] = data[column].astype('object')
    data[column].fillna('no reviews', inplace=True)

# Define the categorization function
def categorie(number):
    if number == 'no reviews':
        return number
    elif 0 <= number < 4:
        return '0-3.99'
    elif 4 <= number < 4.30:
        return '4-4.29'
    elif 4.30 <= number < 4.50:
        return '4.30-4.49'
    elif 4.50 <= number < 4.75:
        return '4.50-4.74'
    elif 4.75 <= number < 4.90:
        return '4.75-4.89'
    elif 4.90 <= number < 5:
        return '4.90-4.99'
    elif number == 5:
        return '5'
    else:
        return 'no reviews'

# Apply the categorization function to each review score column
for column in review_score_columns:
    new_column = f'{column}_cat'
    data[new_column] = data[column].apply(categorie)

# Get the count of each category for each column and display the result
for column in review_score_columns:
    new_column = f'{column}_cat'
    review_rating = data[new_column].value_counts()
    print(f"Review Rating by Category for {column}:")
    print(review_rating)
    print("\n")

#################################
# List of review score columns
#################################

review_score_columns_cat = [
    'review_scores_rating_cat', 'review_scores_accuracy_cat', 'review_scores_cleanliness_cat',
    'review_scores_checkin_cat', 'review_scores_communication_cat', 'review_scores_location_cat',
    'review_scores_value_cat'
]

# Plotting the categories for each review score column
for column in review_score_columns_cat:
    # Get the counts and sort them
    sorted_counts = data[column].value_counts().sort_values(ascending=False)

    plt.figure(figsize=(12, 6))
    sns.countplot(data=data, x=column, palette='viridis', order=sorted_counts.index)
    plt.title(f'Review Rating by Category for {column}')
    plt.xlabel('Category')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.show()
    ##############################################
    # Define the categorization function for 'distance_to_center_km'
def categorize_distance(distance):
     if distance <= 2:
            return '0-1.99 km'
     elif 2 <= distance < 3:
            return '2-2.99 km'
     elif 3 <= distance < 4:
            return '3-3.99 km'
     elif 4 <= distance < 5:
            return '4-4.99 km'
     elif 5 <= distance < 6:
            return '5-5.99 km'
     elif 6 <= distance < 7:
            return '6-6.99 km'
     elif 7 <= distance < 8:
            return '7-7.99 km'
     elif 8 <= distance < 9:
            return '8-8.99 km'
     elif 9 <= distance < 10:
            return '9-9.99 km'
     elif 10 <= distance < 12:
            return '10-11.99 km'
     elif 12 <= distance < 15:
            return '12-14.99 km'
     elif 15 <= distance < 20:
            return '15-19.99 km'
     else:
            return '> 20 km'


    # Apply the categorization function to the 'distance_to_center_km' column
data['distance_to_center_km_category'] = data['distance_to_center_km'].apply(categorize_distance)

    # Get the count of each category in 'distance_to_center_km_category'
distance_category_counts = data['distance_to_center_km_category'].value_counts()

    # Display the result for distance categories
print("Distance to Center by Category:")
print(distance_category_counts)
########################
########################
sorted_counts = data['distance_to_center_km_category'].value_counts().sort_values(ascending=False)
plt.figure(figsize=(12, 6))
sns.countplot(data=data, x='distance_to_center_km_category', palette='viridis')
plt.title('Distance to Center by Category')
plt.xlabel('Distance Category')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.show()
    ##############################################
zero_beds_rows = data[data['beds'] == 0]
bathrooms_counts = zero_beds_rows.groupby('bathrooms').size()
bathrooms_mean_price = zero_beds_rows.groupby('bathrooms')['price'].mean()
bathrooms_summary = pd.DataFrame({'count': bathrooms_counts, 'mean_price': bathrooms_mean_price})

print("Summary for bathrooms where beds is 0:")
print(bathrooms_summary)

accommodates_counts = zero_beds_rows.groupby('accommodates').size()
accommodates_mean_price = zero_beds_rows.groupby('accommodates')['price'].mean()
accommodates_summary = pd.DataFrame({'count': accommodates_counts, 'mean_price': accommodates_mean_price})

print("Summary for accommodates where beds is 0:")
print(accommodates_summary)

#########################
#filling 0 values  in 'beds' based on 'accommodates'
def fill_beds(row):
    if row['beds'] == 0:
        if row['accommodates'] in [1, 2]:
            return 1
        elif row['accommodates'] in [3, 4]:
            return 2
        elif row['accommodates'] in [5, 6]:
            return 3
        elif row['accommodates'] in [7, 8]:
            return 4
        elif row['accommodates'] in [9, 10]:
            return 5
        elif row['accommodates'] in [11, 12]:
            return 6
        elif row['accommodates'] >= 13:
            return 7
    return row['beds']

# Apply the function to the DataFrame
data['beds'] = data.apply(fill_beds, axis=1)
########################
# Categorizing the 'price' column
# Define bins based on descriptive statistics and potential outliers
bins = [0, 50, 100, 150, 200, 300, 400, 500, 1000, 2000, float('inf')]
labels = ['Very Low', 'Low', 'Below Average', 'Average', 'Above Average', 'High', 'Very High', 'Luxury', 'Ultra Luxury', 'Outliers']

# Create a new column 'price_category' with more categories
data['price_category'] = pd.cut(data['price'], bins=bins, labels=labels, include_lowest=True)

# Display the DataFrame with the new 'price_category' column
#print(data[['price', 'price_category']])

# Display the distribution of price categories
price_category_counts = data['price_category'].value_counts()
print(price_category_counts)

########################
# Group by 'bathrooms'
grouped_bathrooms = data.groupby('bathrooms').agg(
    median_price=('price', 'median'),
    listing_count=('price', 'count')
).reset_index()

# Group by 'bedrooms'
grouped_bedrooms = data.groupby('bedrooms').agg(
    median_price=('price', 'median'),
    listing_count=('price', 'count')
).reset_index()

# Group by 'beds'
grouped_beds = data.groupby('beds').agg(
    median_price=('price', 'median'),
    listing_count=('price', 'count')
).reset_index()

# Group by 'minimum_nights'
grouped_minimum_nights = data.groupby('minimum_nights').agg(
    median_price=('price', 'median'),
    listing_count=('price', 'count')
).reset_index()

# Group by 'availability_30'
grouped_availability_30 = data.groupby('availability_30').agg(
    median_price=('price', 'median'),
    listing_count=('price', 'count')
).reset_index()

# Group by 'review_scores_rating'
grouped_review_scores_rating = data.groupby('review_scores_rating_cat').agg(
    median_price=('price', 'median'),
    listing_count=('price', 'count')
).reset_index()
#data.columns
# Group by 'property_type'
grouped_property_type = data.groupby('property_type').agg(
    median_price=('price', 'median'),
    listing_count=('price', 'count')
).reset_index()

# Group by 'room_type'
grouped_room_type = data.groupby('room_type').agg(
    median_price=('price', 'median'),
    listing_count=('price', 'count')
).reset_index()

# Group by 'number_of_reviews_ltm_cat'
grouped_number_of_reviews_ltm_cat = data.groupby('number_of_reviews_ltm_cat').agg(
    median_price=('price', 'median'),
    listing_count=('price', 'count')
).reset_index()

# Group by 'distance_to_center_km_category'
grouped_distance_to_center_km_category = data.groupby('distance_to_center_km_category').agg(
    median_price=('price', 'median'),
    listing_count=('price', 'count')
).reset_index()
# Group by 'price_category'
grouped_price_category = data.groupby('price_category').agg(
    median_price=('price', 'median'),
    listing_count=('price', 'count')
).reset_index()

# Display the grouped data
print("Grouped by Bathrooms:")
print(grouped_bathrooms.sort_values(by='bathrooms', ascending=False))

print("\nGrouped by Bedrooms:")
print(grouped_bedrooms)

print("\nGrouped by Beds:")
print(grouped_beds)

print("\nGrouped by Minimum Nights:")
print(grouped_minimum_nights)

print("\nGrouped by Availability 30:")
print(grouped_availability_30)

print("\nGrouped by Review Scores Rating Category:")
print(grouped_review_scores_rating)

print("\nGrouped by Property type Category:")
print(grouped_property_type)

print("\nGrouped by Room type Category:")
print(grouped_room_type)

print("\nGrouped by Number of reviews 12M Category:")
print(grouped_number_of_reviews_ltm_cat)

print("\nGrouped by Distance to center km Category:")
print(grouped_distance_to_center_km_category)

print("\nGrouped by Price Category:")
print(grouped_price_category)

####################################
## Define a function to create bar plots for median price and listing count
def plot_grouped_data(grouped_df, category_name, x_col):
    fig, ax1 = plt.subplots(figsize=(12, 6))

    color = 'tab:blue'
    ax1.set_xlabel(category_name)
    ax1.set_ylabel('Median Price', color=color)
    sns.barplot(x=x_col, y='median_price', data=grouped_df, ax=ax1, palette='viridis')
    ax1.tick_params(axis='y', labelcolor=color)
    plt.xticks(rotation=45)

    ax2 = ax1.twinx()
    color = 'tab:green'
    ax2.set_ylabel('Listing Count', color=color)
    sns.barplot(x=x_col, y='listing_count', data=grouped_df, ax=ax2, alpha=0.6, palette='Set2')
    ax2.tick_params(axis='y', labelcolor=color)

    plt.title(f'{category_name}: Median Price and Listing Count')
    fig.tight_layout()
    plt.show()

# Group the data
grouped_data = {
    'Bathrooms': ('bathrooms', grouped_bathrooms),
    'Bedrooms': ('bedrooms', grouped_bedrooms),
    'Beds': ('beds', grouped_beds),
    'Minimum Nights': ('minimum_nights', grouped_minimum_nights),
    'Availability 30': ('availability_30', grouped_availability_30),
    'Review Scores Rating Category': ('review_scores_rating_cat', grouped_review_scores_rating),
    'Property Type': ('property_type', grouped_property_type),
    'Room Type': ('room_type', grouped_room_type),
    'Number of Reviews LTM Category': ('number_of_reviews_ltm_cat', grouped_number_of_reviews_ltm_cat),
    'Distance to Center KM Category': ('distance_to_center_km_category', grouped_distance_to_center_km_category),
    'Price Category': ('price_category', grouped_price_category)
}

# Plot the grouped data
for category_name, (x_col, grouped_df) in grouped_data.items():
    plot_grouped_data(grouped_df, category_name, x_col)
#####################################
#Airbnb has a restriction that Entire units can be booked max 90 days per year.
# Drop rows where 'minimum_nights' is 90 or more
data = data[data['minimum_nights'] < 90]

# Verify the changes
print("Rows remaining after dropping listings with 'minimum_nights' 90 or more:")
print(data.shape)

####################################
grouped_price_category = data.groupby('price_category').agg(
    median_price=('price', 'median'),
    listing_count=('price', 'count'),
    avg_beds=('beds', 'mean'),
    avg_accommodates=('accommodates', 'mean'),
    avg_bathrooms=('bathrooms', 'mean')
).reset_index()
print("\nGrouped by Price Category:")
#print(grouped_price_category)
######################################
# Drop rows where 'price_category' is 'Outliers'
data = data[data['price_category'] != 'Outliers']

# Verify the changes
print("Rows remaining after dropping 'Outliers' from 'price_category':")
print(data.shape)


######################################

beds_to_drop = grouped_beds[(grouped_beds['median_price'] < 31) | (grouped_beds['listing_count'] < 3)]['beds']
bedrooms_to_drop = grouped_bedrooms[grouped_bedrooms['listing_count'] < 3]['bedrooms']
bathrooms_to_drop = grouped_bathrooms[grouped_bathrooms['listing_count'] < 5]['bathrooms']

# Drop rows from the original DataFrame based on these conditions
data_filtered = data[~data['beds'].isin(beds_to_drop)]
data_filtered = data_filtered[~data_filtered['bedrooms'].isin(bedrooms_to_drop)]
data_filtered = data_filtered[~data_filtered['bathrooms'].isin(bathrooms_to_drop)]

# Display the filtered DataFrame
#print("Filtered Data:")
#print(data_filtered)
################################
# Step 2: Drop rows from the original DataFrame based on these values
data.drop(data[data['beds'].isin(beds_to_drop)].index, inplace=True)
data.drop(data[data['bedrooms'].isin(bedrooms_to_drop)].index, inplace=True)
data.drop(data[data['bathrooms'].isin(bathrooms_to_drop)].index, inplace=True)
data.shape
"""(38971, 63)"""
###############################################

drop_col = ['first_review','last_review']
data=data.drop(columns=drop_col)
data['price'].describe()
data.columns

################

# Visualization of the distribution of prices to see the market competition
plt.hist(data['price'], color='#189AB4', bins=30, edgecolor='black', linewidth=0.5)
plt.title('Distribution of the Prices')
plt.xlabel('Prices')
plt.ylabel('Frequency')
plt.show(block=True)

print(data['price'].head())

###################################
import matplotlib.pyplot as plt
import math

# Apply the logarithmic transformation to the 'price' column
data['price_normal'] = data['price'].apply(math.log)

# Plot the histogram of the normalized prices
plt.hist(data['price_normal'], color='#189AB4', bins=30, edgecolor='black', linewidth=0.5)
plt.title('Distribution of the Normalized Prices')
plt.xlabel('Log(Prices)')
plt.ylabel('Frequency')
plt.show(block=True)
print(data['price_normal'].head())
#########################

import seaborn as sns

# Select numeric columns, assuming 'id' and 'host_id' columns do not exist or are already excluded
numeric_columns = data.select_dtypes(include=['int', 'float']).columns

# Create heatmap using numeric columns
plt.figure(figsize=[20, 8])  # Increase the figure size for better readability
heatmap = sns.heatmap(data[numeric_columns].corr(), annot=True, cmap='coolwarm', fmt='.2f')

# Rotate the x-axis labels
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)  # Ensure y-axis labels are horizontal
plt.title('Correlation Heatmap of Numeric Columns')
plt.show()

##############################################


grouped_data = data.groupby('distance_to_center_km_category')['price'].median().reset_index()

# Sort the grouped data by median price
grouped_data = grouped_data.sort_values(by='price', ascending=False)

# Plot the bar chart
plt.figure(figsize=[12, 6])
plt.bar(grouped_data['distance_to_center_km_category'], grouped_data['price'], alpha=0.7, edgecolor='black', linewidth=0.5)

# Set titles and labels
plt.title('Median Price by Distance to Center', fontsize=14)
plt.xlabel('Distance to Center (km)', fontsize=12)
plt.ylabel('Median Price ($)', fontsize=12)

# Rotate x-axis labels for better readability
plt.xticks(ticks=grouped_data['distance_to_center_km_category'], rotation=45, ha='right', fontsize=10)

# Add grid lines
plt.grid(axis='y', linestyle='--', alpha=0.5)

# Annotate bars with the median price values
for i, v in enumerate(grouped_data['price']):
    plt.text(i, v + 5, '${:.0f}'.format(v), ha='center', va='bottom', fontsize=10)

# Remove top and right spines for a cleaner look
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)

# Adjust layout to fit everything nicely
plt.tight_layout()
plt.show()

##############################
# Identify categorical columns
cat_columns = data.select_dtypes(include=['object', 'category']).columns

# Print the list of categorical columns
print("Categorical Columns:")
print(cat_columns)

##########################################
#Categorizing the 'host_since' column
from datetime import datetime
# End date
end_date = datetime(2024, 3, 22)

# Convert 'host_since' to datetime
data['host_since'] = pd.to_datetime(data['host_since'], errors='coerce')

# Calculate the difference in years
data['years_since_hosting'] = (end_date - data['host_since']).dt.days / 365.25

# Define the categorization function for 'years_since_hosting'
def categorize_years(years):
    if years < 1:
        return 'less than 1 year'
    elif 1 <= years < 2:
        return 'between 1-2 year'
    elif 2 <= years < 3:
        return 'between 2-3 years'
    elif 3 <= years < 4:
        return 'between 3-4 years'
    elif 4 <= years < 5:
        return 'between 4-5 years'
    elif 5 <= years < 6:
        return 'between 5-6 years'
    elif 6 <= years < 7:
        return 'between 6-7 years'
    elif 7 <= years < 8:
        return 'between 7-8 years'
    elif 8 <= years < 9:
        return 'between 8-9 years'
    elif 9 <= years < 10:
        return 'between 9-10 years'
    else:
        return '10 years or more'

# Apply the categorization function to 'years_since_hosting'
data['host_since_category'] = data['years_since_hosting'].apply(categorize_years)

# Get the count of each category in 'distance_to_center_km_category'
years_since_hosting_category_counts = data['host_since_category'].value_counts()

# Display the result for distance categories
print("Host Since by Category:")
print(years_since_hosting_category_counts)

############################################
#One-Hot Encoding the Categorical Variables

# List of categorical columns to encode
categorical_columns = [
    'neighbourhood_cleansed', 'room_type', 'property_type', 'host_is_superhost', 'instant_bookable',
    'number_of_reviews_ltm_cat', 'review_scores_rating_cat', 'review_scores_accuracy_cat',
    'review_scores_cleanliness_cat', 'review_scores_checkin_cat',
    'review_scores_communication_cat', 'review_scores_location_cat',
    'review_scores_value_cat', 'distance_to_center_km_category', 'host_since_category'
]

# One-hot encode the categorical columns
encoded_data = pd.get_dummies(data[categorical_columns], drop_first=True)

#######################################
#Calculating the Correlation with 'price'
# Include the 'price' column for correlation calculation
encoded_data['price'] = data['price']
#encoded_data['price_normal'] = data['price_normal']


# Calculate the correlation matrix
correlation_matrix = encoded_data.corr()

# Extract the correlation values with 'price'
price_correlation = correlation_matrix['price'].sort_values(ascending=False)

# Filter variables above a correlation threshold (e.g., 0.05)
threshold = 0.1
filtered_price_correlation = price_correlation[abs(price_correlation) > threshold]

# Remove highly correlated variables (e.g., correlation above 0.8 between them)
filtered_features = filtered_price_correlation.index.tolist()
correlation_matrix_filtered = correlation_matrix.loc[filtered_features, filtered_features]

# Identify and drop highly correlated features
to_drop = set()
for i in range(len(correlation_matrix_filtered.columns)):
    for j in range(i):
        if abs(correlation_matrix_filtered.iloc[i, j]) > 0.8:
            colname = correlation_matrix_filtered.columns[i]
            to_drop.add(colname)

filtered_features = [feature for feature in filtered_features if feature not in to_drop]
correlation_matrix_final = correlation_matrix.loc[filtered_features, filtered_features]

# Plot the heatmap of the filtered and final correlated features
plt.figure(figsize=(14, 10))
sns.heatmap(correlation_matrix_final, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Matrix Heatmap of Filtered Features')
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()

# Display the filtered correlations with 'price'
filtered_price_correlation_final = filtered_price_correlation[filtered_features]
print("Filtered Correlations of Categorical Variables with 'price':")
print(filtered_price_correlation_final)
#####################################
"""Filtered Correlations of Categorical Variables with 'price':
price_normal                                     1.00
property_type_Entire rental unit                 0.31
property_type_Entire home                        0.26
neighbourhood_cleansed_Westminster               0.23
neighbourhood_cleansed_Kensington and Chelsea    0.18
review_scores_location_cat_5                     0.17
property_type_Entire condo                       0.15
property_type_Entire townhouse                   0.13
property_type_Entire serviced apartment          0.11
distance_to_center_km_category_2-2.99 km         0.11
distance_to_center_km_category_3-3.99 km         0.10
distance_to_center_km_category_10-11.99 km      -0.10
room_type_Shared room                           -0.11
property_type_Shared room in hostel             -0.11
distance_to_center_km_category_12-14.99 km      -0.12
property_type_Private room in townhouse         -0.12
review_scores_location_cat_4.50-4.74            -0.13
number_of_reviews_ltm_cat_31_or_more            -0.14
distance_to_center_km_category_15-19.99 km      -0.16
property_type_Private room in condo             -0.19
property_type_Private room in rental unit       -0.32
property_type_Private room in home              -0.41
room_type_Private room                          -0.65
Name: price_normal, dtype: float64"""
##############################################
# List of categorical columns to be one-hot encoded
categorical_columns = [
    'neighbourhood_cleansed', 'room_type', 'property_type', 'host_is_superhost', 'instant_bookable',
    'number_of_reviews_ltm_cat', 'review_scores_rating_cat', 'review_scores_accuracy_cat',
    'review_scores_cleanliness_cat', 'review_scores_checkin_cat',
    'review_scores_communication_cat', 'review_scores_location_cat',
    'review_scores_value_cat', 'distance_to_center_km_category', 'host_since_category'
]

# One-hot encode the categorical columns
encoded_data = pd.get_dummies(data[categorical_columns], drop_first=True)

# Add the encoded columns back to the original dataframe, excluding the original categorical columns
data = pd.concat([data.drop(columns=categorical_columns), encoded_data], axis=1)
# List of significant variables based on correlation results
significant_vars = [
    'property_type_Entire rental unit',
    'property_type_Entire home',
    'neighbourhood_cleansed_Westminster',
    'neighbourhood_cleansed_Kensington and Chelsea',
    'review_scores_location_cat_5',
    'property_type_Entire condo',
    'property_type_Entire townhouse',
    'property_type_Entire serviced apartment',
    'distance_to_center_km_category_2-2.99 km',
    'review_scores_location_cat_4.90-4.99',
    'distance_to_center_km_category_3-3.99 km',
    'review_scores_checkin_cat_5',
    'review_scores_communication_cat_5',
    'review_scores_rating_cat_5',
    'review_scores_cleanliness_cat_5',
    'review_scores_accuracy_cat_5',
    'neighbourhood_cleansed_Camden',
    'number_of_reviews_ltm_cat_6-15',
    'review_scores_value_cat_4.50-4.74',
    'distance_to_center_km_category_4-4.99 km',
    'instant_bookable_t',
    'neighbourhood_cleansed_City of London',
    'neighbourhood_cleansed_Sutton',
    'property_type_Private room in bed and breakfast',
    'review_scores_communication_cat_4.90-4.99',
    'neighbourhood_cleansed_Enfield',
    'neighbourhood_cleansed_Barnet',
    'property_type_Private room in guesthouse',
    'neighbourhood_cleansed_Redbridge',
    'neighbourhood_cleansed_Haringey',
    'review_scores_location_cat_4-4.29',
    'neighbourhood_cleansed_Waltham Forest',
    'neighbourhood_cleansed_Bexley',
    'neighbourhood_cleansed_Ealing',
    'neighbourhood_cleansed_Brent',
    'review_scores_checkin_cat_4.90-4.99',
    'neighbourhood_cleansed_Hillingdon',
    'neighbourhood_cleansed_Lewisham',
    'distance_to_center_km_category_9-9.99 km',
    'distance_to_center_km_category_> 20 km',
    'review_scores_location_cat_4.30-4.49',
    'review_scores_value_cat_4.90-4.99',
    'neighbourhood_cleansed_Croydon',
    'distance_to_center_km_category_10-11.99 km',
    'property_type_Shared room in hostel',
    'room_type_Shared room',
    'distance_to_center_km_category_12-14.99 km',
    'property_type_Private room in townhouse',
    'review_scores_location_cat_4.50-4.74',
    'number_of_reviews_ltm_cat_31_or_more',
    'distance_to_center_km_category_15-19.99 km',
    'property_type_Private room in condo',
    'property_type_Private room in rental unit',
    'property_type_Private room in home',
    'room_type_Private room'
]

for var in significant_vars:
    count = data[var].value_counts()
    print(f"\nCounts for {var}:")
    print(count)

# Additionally, you can check the unique values and their respective counts for these categories
for var in significant_vars:
    unique_values = data[var].unique()
    print(f"\nUnique values and counts for {var}:")
    print(data[var].value_counts())


######################################
#####################################
import pandas as pd

# Assuming 'data' is your original dataframe

# Step 1: Select Numeric Columns
numeric_columns = data[['accommodates', 'bathrooms', 'bedrooms', 'beds', 'price']]

# List of significant variables based on correlation results
significant_vars = [
    'property_type_Entire rental unit',
    'property_type_Entire home',
    'neighbourhood_cleansed_Westminster',
    'neighbourhood_cleansed_Kensington and Chelsea',
    'review_scores_location_cat_5',
    'property_type_Entire condo',
    'property_type_Entire townhouse',
    'property_type_Entire serviced apartment',
    'distance_to_center_km_category_2-2.99 km',
    'review_scores_location_cat_4.90-4.99',
    'distance_to_center_km_category_3-3.99 km',
    'review_scores_checkin_cat_5',
    'review_scores_communication_cat_5',
    'review_scores_rating_cat_5',
    'review_scores_cleanliness_cat_5',
    'review_scores_accuracy_cat_5',
    'neighbourhood_cleansed_Camden',
    'number_of_reviews_ltm_cat_6-15',
    'review_scores_value_cat_4.50-4.74',
    'distance_to_center_km_category_4-4.99 km',
    'instant_bookable_t',
    'neighbourhood_cleansed_City of London',
    'neighbourhood_cleansed_Sutton',
    'property_type_Private room in bed and breakfast',
    'review_scores_communication_cat_4.90-4.99',
    'neighbourhood_cleansed_Enfield',
    'neighbourhood_cleansed_Barnet',
    'property_type_Private room in guesthouse',
    'neighbourhood_cleansed_Redbridge',
    'neighbourhood_cleansed_Haringey',
    'review_scores_location_cat_4-4.29',
    'neighbourhood_cleansed_Waltham Forest',
    'neighbourhood_cleansed_Bexley',
    'neighbourhood_cleansed_Ealing',
    'neighbourhood_cleansed_Brent',
    'review_scores_checkin_cat_4.90-4.99',
    'neighbourhood_cleansed_Hillingdon',
    'neighbourhood_cleansed_Lewisham',
    'distance_to_center_km_category_9-9.99 km',
    'distance_to_center_km_category_> 20 km',
    'review_scores_location_cat_4.30-4.49',
    'review_scores_value_cat_4.90-4.99',
    'neighbourhood_cleansed_Croydon',
    'distance_to_center_km_category_10-11.99 km',
    'property_type_Shared room in hostel',
    'room_type_Shared room',
    'distance_to_center_km_category_12-14.99 km',
    'property_type_Private room in townhouse',
    'review_scores_location_cat_4.50-4.74',
    'number_of_reviews_ltm_cat_31_or_more',
    'distance_to_center_km_category_15-19.99 km',
    'property_type_Private room in condo',
    'property_type_Private room in rental unit',
    'property_type_Private room in home',
    'room_type_Private room'
]

# Ensure all significant variables are present in the dataset
encoded_columns = data[significant_vars]

# Combine the numeric columns with the significant variables
final_dataset = pd.concat([numeric_columns, encoded_columns], axis=1)

# Verify the final dataset
print("Final dataset columns:")
print(final_dataset.columns)
print("Shape of the final dataset:", final_dataset.shape)

##############################################
#################################################
#Train and Evaluate the Models
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Assuming 'final_dataset' is your prepared dataset

# Step 1: Split the Data
X = final_dataset.drop('price', axis=1)
y = final_dataset['price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 2: Define the Models
models = {
    'Linear Regression': LinearRegression(),
    'Decision Tree': DecisionTreeRegressor(random_state=42),
    'Random Forest': RandomForestRegressor(random_state=42),
    'Gradient Boosting': GradientBoostingRegressor(random_state=42)
}


# Step 3: Train and Evaluate the Models
def evaluate_model(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    results = {
        'Train MAE': mean_absolute_error(y_train, y_pred_train),
        'Test MAE': mean_absolute_error(y_test, y_pred_test),
        'Train RMSE': mean_squared_error(y_train, y_pred_train, squared=False),
        'Test RMSE': mean_squared_error(y_test, y_pred_test, squared=False),
        'Train R2': r2_score(y_train, y_pred_train),
        'Test R2': r2_score(y_test, y_pred_test)
    }

    return results


results = {}
for model_name, model in models.items():
    results[model_name] = evaluate_model(model, X_train, y_train, X_test, y_test)

results_df = pd.DataFrame(results).transpose()
print(results_df)


#########################################
##############################################
#GridSearchCV for hyperparameter tuning
from sklearn.model_selection import GridSearchCV

# Hyperparameter tuning for Random Forest
rf_params = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

rf_grid = GridSearchCV(RandomForestRegressor(random_state=42), rf_params, cv=3, scoring='neg_mean_squared_error', n_jobs=-1)
rf_grid.fit(X_train, y_train)

print("Best parameters for Random Forest:", rf_grid.best_params_)
best_rf = rf_grid.best_estimator_

# Hyperparameter tuning for Gradient Boosting
gb_params = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 5, 7],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

gb_grid = GridSearchCV(GradientBoostingRegressor(random_state=42), gb_params, cv=3, scoring='neg_mean_squared_error', n_jobs=-1)
gb_grid.fit(X_train, y_train)

print("Best parameters for Gradient Boosting:", gb_grid.best_params_)
best_gb = gb_grid.best_estimator_

# Evaluate the best models
rf_results = evaluate_model(best_rf, X_train, y_train, X_test, y_test)
gb_results = evaluate_model(best_gb, X_train, y_train, X_test, y_test)

print("Random Forest with best parameters:", rf_results)
print("Gradient Boosting with best parameters:", gb_results)

###########################################
#Cross-Validation
from sklearn.model_selection import cross_val_score

# Cross-validation for Random Forest
rf_cv_scores = cross_val_score(best_rf, X, y, cv=5, scoring='r2')
print("Cross-Validation R2 Scores for Random Forest:", rf_cv_scores)
print("Mean Cross-Validation R2 Score for Random Forest:", rf_cv_scores.mean())

# Cross-validation for Gradient Boosting
gb_cv_scores = cross_val_score(best_gb, X, y, cv=5, scoring='r2')
print("Cross-Validation R2 Scores for Gradient Boosting:", gb_cv_scores)
print("Mean Cross-Validation R2 Score for Gradient Boosting:", gb_cv_scores.mean())

"""Cross-Validation R2 Scores for Random Forest: [ -6.17765274  -8.63139963 -10.95122028  -7.87807407  -4.09148103]
Mean Cross-Validation R2 Score for Random Forest: -7.545965550985414
Cross-Validation R2 Scores for Gradient Boosting: [-6.0804521  -7.2647237  -9.82596566 -6.4309672  -3.25299241]
Mean Cross-Validation R2 Score for Gradient Boosting: -6.571020213535013"""
####################################
#Feature Importance
# Feature importance for Random Forest
rf_feature_importances = pd.DataFrame(best_rf.feature_importances_, index=X_train.columns, columns=['importance']).sort_values('importance', ascending=False)
print("Feature Importances for Random Forest:")
print(rf_feature_importances)

# Feature importance for Gradient Boosting
gb_feature_importances = pd.DataFrame(best_gb.feature_importances_, index=X_train.columns, columns=['importance']).sort_values('importance', ascending=False)
print("Feature Importances for Gradient Boosting:")
print(gb_feature_importances)


################################
#Refined Model Development
# Consider only the most important features for the next iteration
important_features = [
    'room_type_Private room',
    'bedrooms',
    'bathrooms',
    'accommodates',
    'neighbourhood_cleansed_Westminster',
    'neighbourhood_cleansed_Kensington and Chelsea',
    'room_type_Shared room',
    'distance_to_center_km_category_2-2.99 km',
    'distance_to_center_km_category_3-3.99 km',
    'distance_to_center_km_category_15-19.99 km',
    'review_scores_location_cat_4.90-4.99',
    'review_scores_location_cat_5',
    'review_scores_location_cat_4.50-4.74',
    'number_of_reviews_ltm_cat_6-15',
    'review_scores_value_cat_4.50-4.74',
    'neighbourhood_cleansed_Camden',
    'property_type_Shared room in hostel',
    'distance_to_center_km_category_12-14.99 km',
    'distance_to_center_km_category_4-4.99 km',
    'instant_bookable_t'
]

# Create a new dataset with only the important features
X_important = data[important_features]
y = data['price']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X_important, y, test_size=0.2, random_state=42)

# Train and evaluate models with the simplified feature set
models = {
    'Linear Regression': LinearRegression(),
    'Decision Tree': DecisionTreeRegressor(random_state=42),
    'Random Forest': RandomForestRegressor(random_state=42),
    'Gradient Boosting': GradientBoostingRegressor(random_state=42)
}

results = {}
for model_name, model in models.items():
    results[model_name] = evaluate_model(model, X_train, y_train, X_test, y_test)

results_df = pd.DataFrame(results).transpose()
print(results_df)
####################################################
"""'price_normal'"""
"""                   Train MAE  Test MAE  Train RMSE  Test RMSE  Train R2  Test R2
Linear Regression       0.29      0.29        0.40       0.39      0.68     0.68
Decision Tree           0.18      0.32        0.28       0.46      0.85     0.57
Random Forest           0.21      0.29        0.29       0.40      0.83     0.67
Gradient Boosting       0.28      0.27        0.38       0.38      0.72     0.71"""
"""'price'"""
"""                   Train MAE  Test MAE  Train RMSE  Test RMSE  Train R2  Test R2
Linear Regression      48.52     47.56       81.69      76.89      0.57     0.56
Decision Tree           3.22     58.04       15.71     106.58      0.98     0.15
Random Forest          18.14     43.41       32.89      76.47      0.93     0.56
Gradient Boosting      42.57     41.81       74.81      73.63      0.64     0.59
"""
####################################################

# Further hyperparameter tuning for Gradient Boosting
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV
import numpy as np

# Define the parameter grid for Gradient Boosting
gb_params = {
    'n_estimators': [100, 200, 300, 400],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'max_depth': [3, 5, 7],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Create the GridSearchCV object
gb_grid = GridSearchCV(GradientBoostingRegressor(random_state=42), gb_params, cv=5, scoring='neg_mean_squared_error',
                       n_jobs=-1)

# Fit the model
gb_grid.fit(X_train, y_train)

# Get the best parameters and the best model
print("Best parameters for Gradient Boosting:", gb_grid.best_params_)
best_gb = gb_grid.best_estimator_


# Evaluate the best model
def evaluate_model(model, X_train, y_train, X_test, y_test):
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)

    train_mae = np.mean(np.abs(train_pred - y_train))
    test_mae = np.mean(np.abs(test_pred - y_test))
    train_rmse = np.sqrt(np.mean((train_pred - y_train) ** 2))
    test_rmse = np.sqrt(np.mean((test_pred - y_test) ** 2))
    train_r2 = model.score(X_train, y_train)
    test_r2 = model.score(X_test, y_test)

    results = {
        'Train MAE': train_mae,
        'Test MAE': test_mae,
        'Train RMSE': train_rmse,
        'Test RMSE': test_rmse,
        'Train R2': train_r2,
        'Test R2': test_r2
    }

    return results


# Print the evaluation results
gb_results = evaluate_model(best_gb, X_train, y_train, X_test, y_test)
print("Gradient Boosting with best parameters:", gb_results)

#################
"""Train MAE: 0.2627
Test MAE: 0.2699
Train RMSE: 0.3591
Test RMSE: 0.3724
Train R2: 0.7429
Test R2: 0.7131"""

#########################
import matplotlib.pyplot as plt
import seaborn as sns

# Predict on the test set
y_pred = best_gb.predict(X_test)

# Scatter plot of actual vs predicted values
plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_test, y=y_pred)
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.title('Actual vs Predicted Prices')
plt.show()

# Residual plot
residuals = y_test - y_pred
plt.figure(figsize=(10, 6))
sns.histplot(residuals, kde=True)
plt.xlabel('Residuals')
plt.title('Residuals Distribution')
plt.show()

###############################
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt

# Assuming 'data' is your dataframe
X = final_dataset.drop(columns=['price'])
y = final_dataset['price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=37)

####################
#Initializing lists for tracking performance metrics:
max_depth_values = range(1, 15)
training_times = []
scores = []
mse_values = []
mae_values = []
########################
#Training RandomForestRegressor for different max_depth values and recording metrics:
for max_depth in max_depth_values:
    rfr = RandomForestRegressor(max_depth=max_depth, random_state=42)

    start_time = time.time()
    rfr.fit(X_train, y_train)
    end_time = time.time()
    elapsed_time = end_time - start_time
    training_times.append(elapsed_time)

    y_pred = rfr.predict(X_test)
    score = rfr.score(X_test, y_test)
    scores.append(score)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    mse_values.append(mse)
    mae_values.append(mae)

    print(f"max_depth={max_depth}:")
    print(f"  Score (R2): {score:.4f}")
    print(f"  Training time: {elapsed_time:.2f} seconds")
    print(f"  MSE: {mse:.4f}")
    print(f"  MAE: {mae:.4f}")
    print("-" * 30)
    ######################

# Plotting
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))

# Plot Score and Training Time (Subplot 1)
ax1.plot(max_depth_values, scores, color='tab:blue', marker='o', label='Score (R2)')
ax1.set_xlabel('max_depth')
ax1.set_ylabel('Score (R2)', color='tab:blue')
ax1.tick_params(axis='y', labelcolor='tab:blue')
ax1.set_xticks(max_depth_values)

ax1_twin = ax1.twinx()
ax1_twin.plot(max_depth_values, training_times, color='tab:red', marker='s', label='Training Time')
ax1_twin.set_ylabel('Training Time (seconds)', color='tab:red')
ax1_twin.tick_params(axis='y', labelcolor='tab:red')

ax1.legend(loc='upper left')
ax1_twin.legend(loc='upper right')
ax1.set_title("Effect of max_depth on Score (R2) and Training Time")

# Plot MSE with Separate Y-axis (Subplot 2)
ax2.plot(max_depth_values, mse_values, color='tab:orange', linestyle='--', marker='s', label='MSE')
ax2.set_xlabel('max_depth')
ax2.set_ylabel('MSE', color='tab:orange')
ax2.tick_params(axis='y', labelcolor='tab:orange')
ax2.set_xticks(max_depth_values)

ax2_twin = ax2.twinx()
ax2_twin.plot(max_depth_values, mae_values, color='tab:green', linestyle='--', marker='^', label='MAE')
ax2_twin.set_ylabel('MAE', color='tab:green')
ax2_twin.tick_params(axis='y', labelcolor='tab:green')

ax2.legend(loc='upper left')
ax2_twin.legend(loc='upper right')
ax2.set_title("Effect of max_depth on MSE and MAE")

plt.tight_layout()
plt.show()

################################
################################
import time
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt

# Initializing lists for tracking performance metrics:
max_depth_values = range(1, 11)
training_times = []
scores = []
mse_values = []
mae_values = []

# Training DecisionTreeRegressor for different max_depth values and recording metrics:
for max_depth in max_depth_values:
    dtr = DecisionTreeRegressor(max_depth=max_depth, random_state=42)

    start_time = time.time()
    dtr.fit(X_train, y_train)
    end_time = time.time()
    elapsed_time = end_time - start_time
    training_times.append(elapsed_time)

    y_pred = dtr.predict(X_test)
    score = dtr.score(X_test, y_test)
    scores.append(score)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    mse_values.append(mse)
    mae_values.append(mae)

    print(f"max_depth={max_depth}:")
    print(f"  Score (R2): {score:.4f}")
    print(f"  Training time: {elapsed_time:.2f} seconds")
    print(f"  MSE: {mse:.4f}")
    print(f"  MAE: {mae:.4f}")
    print("-" * 30)

# Plotting
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))

# Plot Score and Training Time (Subplot 1)
ax1.plot(max_depth_values, scores, color='tab:blue', marker='o', label='Score (R2)')
ax1.set_xlabel('max_depth')
ax1.set_ylabel('Score (R2)', color='tab:blue')
ax1.tick_params(axis='y', labelcolor='tab:blue')
ax1.set_xticks(max_depth_values)

ax1_twin = ax1.twinx()
ax1_twin.plot(max_depth_values, training_times, color='tab:red', marker='s', label='Training Time')
ax1_twin.set_ylabel('Training Time (seconds)', color='tab:red')
ax1_twin.tick_params(axis='y', labelcolor='tab:red')

ax1.legend(loc='upper left')
ax1_twin.legend(loc='upper right')
ax1.set_title("Effect of max_depth on Score (R2) and Training Time")

# Plot MSE with Separate Y-axis (Subplot 2)
ax2.plot(max_depth_values, mse_values, color='tab:orange', linestyle='--', marker='s', label='MSE')
ax2.set_xlabel('max_depth')
ax2.set_ylabel('MSE', color='tab:orange')
ax2.tick_params(axis='y', labelcolor='tab:orange')
ax2.set_xticks(max_depth_values)

ax2_twin = ax2.twinx()
ax2_twin.plot(max_depth_values, mae_values, color='tab:green', linestyle='--', marker='^', label='MAE')
ax2_twin.set_ylabel('MAE', color='tab:green')
ax2_twin.tick_params(axis='y', labelcolor='tab:green')

ax2.legend(loc='upper left')
ax2_twin.legend(loc='upper right')
ax2.set_title("Effect of max_depth on MSE and MAE")

plt.tight_layout()
plt.show()