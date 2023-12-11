import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

"1. Distance Matrix Calculation"
def calculate_distance_matrix(file_path):
    df = pd.read_csv(file_path)
    all_ids = np.unique(np.concatenate([df['id_start'].unique(), df['id_end'].unique()]))
    distance_matrix = pd.DataFrame(0, index=all_ids, columns=all_ids)
    for index, row in df.iterrows():
        distance_matrix.at[row['id_start'], row['id_end']] += row['distance']
        distance_matrix.at[row['id_end'], row['id_start']] += row['distance']
    return distance_matrix

file_path = 'C:\\Users\\Dattatraya\\Documents\\My Tableau Repository\\MapUp-Data-Assessment-F\\datasets\\dataset-3.csv'
result = calculate_distance_matrix(file_path)
print("Distance Matrix Calculation: \n" , result)
plt.imshow(result.values, cmap='viridis', origin='upper')
plt.colorbar()
plt.savefig('task2-q1.png')

"2.Unroll Distance Matrix"
def unroll_distance_matrix(distance_matrix):
    dfs = []
    for id_start in distance_matrix.index:
        for id_end in distance_matrix.columns:
            if id_start != id_end:
                df = pd.DataFrame({
                    'id_start': [id_start],
                    'id_end': [id_end],
                    'distance': [distance_matrix.at[id_start, id_end]]})
                dfs.append(df)
    unrolled_df = pd.concat(dfs, ignore_index=True)
    return unrolled_df
unrolled_result = unroll_distance_matrix(result)
print("Unroll Distance Matrix: \n",unrolled_result)

"3.Finding IDs within Percentage Threshold"
def find_ids_within_ten_percentage_threshold(df, reference_value):
    reference_df = df[df['id_start'] == reference_value]
    if reference_df.empty:
        return []
    average_distance = reference_df['distance'].mean()
    if pd.isna(average_distance):
        return []
    lower_bound = average_distance - 0.1 * average_distance
    upper_bound = average_distance + 0.1 * average_distance
    within_threshold_df = df[(df['distance'] >= lower_bound) & (df['distance'] <= upper_bound)]
    result_ids = sorted(within_threshold_df['id_start'].unique())
    return result_ids
reference_value =3.5
result_ids = find_ids_within_ten_percentage_threshold(unrolled_result, reference_value)
print("Finding IDs within Percentage Threshold:",result_ids)

"4.Calculate Toll Rate"
def calculate_toll_rate(df):
    rate_coefficients = {'moto': 0.8, 'car': 1.2, 'rv': 1.5, 'bus': 2.2, 'truck': 3.6}
    for vehicle_type, rate_coefficient in rate_coefficients.items():
        column_name = f'{vehicle_type}_toll'
        df[column_name] = df['distance'] * rate_coefficient
    return df
result_with_toll = calculate_toll_rate(unrolled_result)
print("Calculate Toll Rate:\n",result_with_toll)

result_with_toll.to_csv('result_with_toll.csv', index=False)
result_with_toll = pd.read_csv('result_with_toll.csv')
plt.figure(figsize=(10, 6))
plt.imshow(result_with_toll.iloc[:, 3:].values, cmap='viridis', interpolation='none', aspect='auto')
plt.colorbar(label='Toll Rates')
plt.xlabel('Vehicle Types')
plt.ylabel('Rows')
plt.title('Toll Rates for Each Vehicle Type')
plt.xticks(range(len(result_with_toll.columns[3:])), result_with_toll.columns[3:], rotation=45)
plt.tight_layout()
plt.savefig('task2_q4.png')
plt.show()

"5.Calculate Time-Based Toll Rates"
##In que 3 it gives the empty list that's why I have created new sample dataframe to see the implementation of the code
from datetime import time, timedelta
unrolled_result = pd.DataFrame({
    'id_start': [1, 2, 3, 4, 5],
    'id_end': [6, 7, 8, 9, 10],
    'datetime': pd.to_datetime(
        ['2023-12-10 08:00:00', '2023-12-10 12:30:00', '2023-12-11 15:45:00', '2023-12-12 20:30:00',
         '2023-12-13 02:00:00']),
    'distance': [30, 45, 20, 60, 15],
    'toll': [10, 15, 8, 20, 5]
})
def find_ids_within_ten_percentage_threshold(df, reference_value):
    reference_df = df[df['id_start'] == reference_value]
    if reference_df.empty or pd.isna(reference_df['distance'].mean()):
        return []
    average_distance = reference_df['distance'].mean()
    lower_bound = average_distance - 0.1 * average_distance
    upper_bound = average_distance + 0.1 * average_distance

    within_threshold_df = df[(df['distance'] >= lower_bound) & (df['distance'] <= upper_bound)]
    result_ids = sorted(within_threshold_df['id_start'].unique())
    return result_ids
def calculate_time_based_toll_rates(df):
    time_ranges = [
        (time(0, 0, 0), time(10, 0, 0), 0.8),
        (time(10, 0, 0), time(18, 0, 0), 1.2),
        (time(18, 0, 0), time(23, 59, 59), 0.8)
    ]
    result_df = df.copy()
    for _, group in result_df.groupby(['id_start', 'id_end']):
        for day in range(7):
            for start_time, end_time, discount_factor in time_ranges:
                start_datetime = pd.to_datetime(f'{day} {start_time}', format='%w %H:%M:%S')
                end_datetime = pd.to_datetime(f'{day} {end_time}', format='%w %H:%M:%S')
                mask = (group['datetime'] >= start_datetime) & (group['datetime'] <= end_datetime)
                result_df.loc[mask, 'toll'] *= discount_factor
    result_df['start_day'] = result_df['datetime'].dt.day_name()
    result_df['start_time'] = result_df['datetime'].dt.time
    result_df['end_day'] = (result_df['datetime'] + timedelta(seconds=1)).dt.day_name()
    result_df['end_time'] = (result_df['datetime'] + timedelta(days=1) - timedelta(seconds=1)).dt.time
    return result_df
reference_value = 2
result_ids = find_ids_within_ten_percentage_threshold(unrolled_result, reference_value)
result_df = calculate_time_based_toll_rates(unrolled_result[unrolled_result['id_start'].isin(result_ids)])
print(result_df)

plt.figure(figsize=(10, 6))
plt.plot(result_df['datetime'], result_df['toll'], label='Toll Rates')
plt.title('Time-based Toll Rates')
plt.xlabel('Datetime')
plt.ylabel('Toll Rates')
plt.legend()
plt.savefig('task2_q5.png')
plt.show()

