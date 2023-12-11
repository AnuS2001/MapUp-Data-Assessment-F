import numpy as np
import pandas as pd

"1.Car Matrix Generation"
def generate_car_matrix(df)->pd.DataFrame:
    car_matrix = df.pivot(index='id_1', columns='id_2', values='car').fillna(0)
    np.fill_diagonal(car_matrix.values, 0)
    return car_matrix
df=pd.read_csv('C:\\Users\\Dattatraya\\Documents\\My Tableau Repository\\MapUp-Data-Assessment-F\\datasets\\dataset-1.csv',encoding='utf-8')
print("Car Matrix: ", generate_car_matrix(df))

"2.Car Type Count Calculation"
def get_type_count(df):
    # Add a new categorical column 'car_type' based on values of the column 'car'
    df['car_type'] = pd.cut(df['car'], bins=[-float('inf'), 15, 25, float('inf')], labels=['low', 'medium', 'high'])
    type_counts = df['car_type'].value_counts().to_dict()
    type_counts = dict(sorted(type_counts.items()))
    return type_counts
print("Car Type & It's Count: ",get_type_count(df))

"3.Bus Count Index Retrieval"
def get_bus_indexes(df):
    bus_mean = df['bus'].mean()
    bus_indexes = df[df['bus'] > 2 * bus_mean].index.tolist()
    bus_indexes.sort()
    return bus_indexes
print("Bus Indices: ",get_bus_indexes(df))

"4. Route Filtering"
def filter_routes(df):
    route_avg_truck = df.groupby('route')['truck'].mean()
    filtered_routes = route_avg_truck[route_avg_truck > 7].index.tolist()
    return filtered_routes
print("List of Route where Average Value of truck>7: ",filter_routes(df))

"5.Matrix Value Modification"
result_matrix=generate_car_matrix(df)
def multiply_matrix(result_matrix):
    modified_matrix = result_matrix.applymap(lambda x: x * 0.75 if x > 20 else x * 1.25)
    modified_matrix = modified_matrix.round(1)
    return modified_matrix
print(multiply_matrix(result_matrix))

modified_result=multiply_matrix(result_matrix)
stacked_result = modified_result.stack().reset_index(name='car')

import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(19,19))
ax.axis('off')
table = ax.table(cellText=stacked_result.values, colLabels=stacked_result.columns, cellLoc='center', loc='center', colColours=['#f0f0f0']*3)
table.auto_set_font_size(False)
table.set_fontsize(10)
plt.savefig('task1-q5.png', bbox_inches='tight', pad_inches=0.05)
plt.show()

"6.Time Check"
def verify_timestamps_completeness(file_path):
    df = pd.read_csv(file_path, parse_dates={'start_timestamp': ['startDay', 'startTime'], 'end_timestamp': ['endDay', 'endTime']})
    for col in ['start_timestamp', 'end_timestamp']:
        df[col] = pd.to_datetime(df[col], errors='coerce')
    invalid_entries = df[df['start_timestamp'].isnull() | df['end_timestamp'].isnull()]
    if not invalid_entries.empty:
        print("Invalid entries:")
        print(invalid_entries)
    df = df.dropna(subset=['start_timestamp', 'end_timestamp'])
    df['duration'] = df['end_timestamp'] - df['start_timestamp']
    invalid_durations = df[df['duration'].isnull()]
    if not invalid_durations.empty:
        print("Invalid durations:")
        print(invalid_durations)
    completeness_check = df.groupby(['id', 'id_2']).apply(lambda group: (
        (group['duration'].min() >= pd.Timedelta(days=1)) and
        (group['duration'].max() <= pd.Timedelta(days=1, seconds=86399)) and
        (group['start_timestamp'].min().weekday() == 0) and
        (group['end_timestamp'].max().weekday() == 6)
    ))
    return completeness_check
result = verify_timestamps_completeness('C:\\Users\\Dattatraya\\Documents\\My Tableau Repository\\MapUp-Data-Assessment-F\\datasets\\dataset-2.csv')
print(result)


