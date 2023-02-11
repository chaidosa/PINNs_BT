import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error

import matplotlib.pyplot as plt
import random



def fetch_data():

    df = pd.read_excel("19B-Data.xlsx", index_col = None, header = 0)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.set_index(df['Date'])
    df = df.sort_index()

    Day_1 = df[df['Date'] == '2013-09-18']
    Day_1 = Day_1.set_index(Day_1['Tripnumber'])
    Day_1 = Day_1.sort_index()
    
    # For first trip to start at time t=0
    Day_Temp = Day_1.copy()
    val = (Day_Temp['Time (seconds)'].iloc[0]).copy()    
    Day_Temp['Time (seconds)'] = Day_Temp['Time (seconds)'] - val
    
    # Dropping column which aren't required
    Day_Temp_only_sections = Day_Temp.copy()
    Day_Temp_only_sections = Day_Temp_only_sections.drop("Date", axis=1)
    Day_Temp_only_sections = Day_Temp_only_sections.drop("Day", axis=1)
    Day_Temp_only_sections = Day_Temp_only_sections.drop("Tripnumber", axis=1)
    Day_Temp_only_sections = Day_Temp_only_sections.drop("Time" , axis=1)

    Prev = np.array(Day_Temp_only_sections['Time (seconds)'])

    d = np.array(Day_Temp_only_sections.drop(['Time (seconds)'], axis=1))

    # By taking maximum speed as 60 Km/h clipping the data where it is greater than that
    mask = d < 100/16.6

    d[mask] = 100/16.6

    Day_Temp_only_sections = Day_Temp_only_sections.drop(['Time (seconds)'], axis=1)
    Day_Temp_only_sections = pd.DataFrame(d, columns = Day_Temp_only_sections.columns)

    # Taking first trip as initial condition
    initial_condition = Day_Temp_only_sections.iloc[0]
    initial_condition = 100 / initial_condition


    boundary_1_v = Day_Temp_only_sections['Section 1']
    boundary_1_v = 100 / boundary_1_v
    boundary_2_v = Day_Temp_only_sections['Section 280']
    boundary_2_v = 100 / boundary_2_v

    for column in Day_Temp_only_sections.columns:
        Day_Temp_only_sections[column] = Day_Temp_only_sections[column] + Prev
        Prev = Day_Temp_only_sections[column]
    
    max_val = 0
    for column in Day_Temp_only_sections.columns:
        temp = max(Day_Temp_only_sections[column])
        if temp > max_val:
            max_val = temp
    
    
    Temp = (Day_Temp_only_sections *5) / max_val

    boundary_1_t = Temp['Section 1']
    boundary_2_t = Temp['Section 280']

    t_tensor = np.zeros(280)

    tf_tensor = np.array(boundary_1_t)
    tf_tensor = np.append(tf_tensor, boundary_2_t)

    Temp_1 = np.zeros(37)
    Temp_2 = np.linspace(2.8000, 2.8000, 37)

    x_tensor = np.linspace(0, 2.8, 280)

    xf_tensor = np.array(Temp_1)
    xf_tensor = np.append(xf_tensor, Temp_2)
    x_tensor  = np.append(x_tensor, xf_tensor)
    
    t_tensor = np.append(t_tensor, tf_tensor)
    

    v_tensor = np.array(initial_condition)
    vf_tensor = np.array(boundary_1_v)
    # print(vf_tensor)
    vf_tensor = np.append(vf_tensor, boundary_2_v)
    v_tensor = np.append(v_tensor, vf_tensor)
    v_tensor = torch.tensor(v_tensor)
    
    v_tensor.flatten(0)
    XT_u_tensor = torch.vstack([torch.tensor(x_tensor, requires_grad=True), torch.tensor(t_tensor, requires_grad=True), v_tensor])
    XT_u_tensor = XT_u_tensor.T

    return XT_u_tensor

    pass

def fetch_test(tripNumber=1):
    df = pd.read_excel("19B-Data.xlsx", index_col = None, header = 0)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.set_index(df['Date'])
    df = df.sort_index()

    Day_1 = df[df['Date'] == '2013-09-18']
    Day_1 = Day_1.set_index(Day_1['Tripnumber'])
    Day_1 = Day_1.sort_index()
    
    # For first trip to start at time t=0
    Day_Temp = Day_1.copy()
    val = (Day_Temp['Time (seconds)'].iloc[0]).copy()    
    Day_Temp['Time (seconds)'] = Day_Temp['Time (seconds)'] - val
    
    # Dropping column which aren't required
    Day_Temp_only_sections = Day_Temp.copy()
    Day_Temp_only_sections = Day_Temp_only_sections.drop("Date", axis=1)
    Day_Temp_only_sections = Day_Temp_only_sections.drop("Day", axis=1)
    Day_Temp_only_sections = Day_Temp_only_sections.drop("Tripnumber", axis=1)
    Day_Temp_only_sections = Day_Temp_only_sections.drop("Time" , axis=1)

    Prev = np.array(Day_Temp_only_sections['Time (seconds)'])

    d = np.array(Day_Temp_only_sections.drop(['Time (seconds)'], axis=1))

    # By taking maximum speed as 60 Km/h clipping the data where it is greater than that
    mask = d < 100/16.6

    d[mask] = 100/16.6

    Day_Temp_only_sections = Day_Temp_only_sections.drop(['Time (seconds)'], axis=1)
    Day_Temp_only_sections = pd.DataFrame(d, columns = Day_Temp_only_sections.columns)

    # Test_velocity    
    test_v = Day_Temp_only_sections.iloc[tripNumber]
    test_v = 100 / test_v
   

    for column in Day_Temp_only_sections.columns:
        Day_Temp_only_sections[column] = Day_Temp_only_sections[column] + Prev
        Prev = Day_Temp_only_sections[column]
    
    max_val = 0
    for column in Day_Temp_only_sections.columns:
        temp = max(Day_Temp_only_sections[column])
        if temp > max_val:
            max_val = temp
    
    
    Temp = (Day_Temp_only_sections *5) / max_val

    t = Temp.iloc[tripNumber]
    
    return test_v, t   

pass


def fetch_data_dl_boundary(n_b_points):

    df = pd.read_excel("19B-Data.xlsx", index_col = None, header = 0)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.set_index(df['Date'])
    df = df.sort_index()

    Day_1 = df[df['Date'] == '2013-09-18']
    Day_1 = Day_1.set_index(Day_1['Tripnumber'])
    Day_1 = Day_1.sort_index()
    
    # For first trip to start at time t=0
    Day_Temp = Day_1.copy()
    val = (Day_Temp['Time (seconds)'].iloc[0]).copy()    
    Day_Temp['Time (seconds)'] = Day_Temp['Time (seconds)'] - val
    
    # Dropping column which aren't required
    Day_Temp_only_sections = Day_Temp.copy()
    Day_Temp_only_sections = Day_Temp_only_sections.drop("Date", axis=1)
    Day_Temp_only_sections = Day_Temp_only_sections.drop("Day", axis=1)
    Day_Temp_only_sections = Day_Temp_only_sections.drop("Tripnumber", axis=1)
    Day_Temp_only_sections = Day_Temp_only_sections.drop("Time" , axis=1)

    Prev = np.array(Day_Temp_only_sections['Time (seconds)'])

    d = np.array(Day_Temp_only_sections.drop(['Time (seconds)'], axis=1))

    # By taking maximum speed as 60 Km/h clipping the data where it is greater than that
    mask = d < 100/16.6

    d[mask] = 100/16.6

    Day_Temp_only_sections = Day_Temp_only_sections.drop(['Time (seconds)'], axis=1)
    Day_Temp_only_sections = pd.DataFrame(d, columns = Day_Temp_only_sections.columns)

    # Taking first trip as initial condition
    initial_condition = Day_Temp_only_sections.iloc[0]
    initial_condition = 100 / initial_condition



    # boundary_1_v = Day_Temp_only_sections['Section 1']
    boundary_1_v = np.zeros(n_b_points)
    # boundary_2_v = Day_Temp_only_sections['Section 280']
    boundary_2_v = np.zeros(n_b_points)

    for column in Day_Temp_only_sections.columns:
        Day_Temp_only_sections[column] = Day_Temp_only_sections[column] + Prev
        Prev = Day_Temp_only_sections[column]
    
    max_val = 0
    for column in Day_Temp_only_sections.columns:
        temp = max(Day_Temp_only_sections[column])
        if temp > max_val:
            max_val = temp
    
    
    Temp = (Day_Temp_only_sections *5) / max_val

    boundary_1_t = np.linspace(0, 2, n_b_points)
    boundary_2_t = np.linspace(0, 2, n_b_points)

    t_tensor = np.zeros(282)
    # t_tensor = Temp.iloc[0]

    # tf_tensor = np.array(boundary_1_t)
    tf_tensor = np.append(boundary_1_t, boundary_2_t)

    Temp_1 = np.zeros(n_b_points)
    Temp_2 = np.linspace(2.8000, 2.8000, n_b_points)

    x_tensor = np.linspace(0, 2.8, 282)

    xf_tensor = np.array(Temp_1)
    xf_tensor = np.append(xf_tensor, Temp_2)
    x_tensor  = np.append(x_tensor, xf_tensor)
    
    t_tensor = np.append(t_tensor, tf_tensor)
    

    v_tensor = np.array(initial_condition)
    v_tensor = np.insert(v_tensor, 0, 0, axis=0)
    v_tensor = np.insert(v_tensor, len(v_tensor), 0, axis=0)
    vf_tensor = np.array(boundary_1_v)
    # print(vf_tensor)
    vf_tensor = np.append(vf_tensor, boundary_2_v)
    v_tensor = np.append(v_tensor, vf_tensor)
    v_tensor = torch.tensor(v_tensor)
    
    v_tensor.flatten(0)
    XT_u_tensor = torch.vstack([torch.tensor(x_tensor, requires_grad=True), torch.tensor(t_tensor, requires_grad=True), v_tensor])
    XT_u_tensor = XT_u_tensor.T

    return XT_u_tensor

pass


def fetch_data_wo_b(n_b_points):

    df = pd.read_excel("19B-Data.xlsx", index_col = None, header = 0)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.set_index(df['Date'])
    df = df.sort_index()

    Day_1 = df[df['Date'] == '2013-09-18']
    Day_1 = Day_1.set_index(Day_1['Tripnumber'])
    Day_1 = Day_1.sort_index()
    
    # For first trip to start at time t=0
    Day_Temp = Day_1.copy()
    val = (Day_Temp['Time (seconds)'].iloc[0]).copy()    
    Day_Temp['Time (seconds)'] = Day_Temp['Time (seconds)'] - val
    
    # Dropping column which aren't required
    Day_Temp_only_sections = Day_Temp.copy()
    Day_Temp_only_sections = Day_Temp_only_sections.drop("Date", axis=1)
    Day_Temp_only_sections = Day_Temp_only_sections.drop("Day", axis=1)
    Day_Temp_only_sections = Day_Temp_only_sections.drop("Tripnumber", axis=1)
    Day_Temp_only_sections = Day_Temp_only_sections.drop("Time" , axis=1)

    Prev = np.array(Day_Temp_only_sections['Time (seconds)'])

    d = np.array(Day_Temp_only_sections.drop(['Time (seconds)'], axis=1))

    # By taking maximum speed as 60 Km/h clipping the data where it is greater than that
    mask = d < 100/16.6

    d[mask] = 100/16.6

    Day_Temp_only_sections = Day_Temp_only_sections.drop(['Time (seconds)'], axis=1)
    Day_Temp_only_sections = pd.DataFrame(d, columns = Day_Temp_only_sections.columns)

    # Taking first trip as initial condition
    initial_condition = Day_Temp_only_sections.iloc[0]
    initial_condition = 100 / initial_condition



    # boundary_1_v = Day_Temp_only_sections['Section 1']
    # boundary_1_v = np.zeros(n_b_points)
    # boundary_2_v = Day_Temp_only_sections['Section 280']
    # boundary_2_v = np.zeros(n_b_points)

    for column in Day_Temp_only_sections.columns:
        Day_Temp_only_sections[column] = Day_Temp_only_sections[column] + Prev
        Prev = Day_Temp_only_sections[column]
    
    max_val = 0
    for column in Day_Temp_only_sections.columns:
        temp = max(Day_Temp_only_sections[column])
        if temp > max_val:
            max_val = temp
    
    
    Temp = (Day_Temp_only_sections *5) / max_val

    # boundary_1_t = np.linspace(0, 2, n_b_points)
    # boundary_2_t = np.linspace(0, 2, n_b_points)

    t_tensor = np.zeros(280)
    # t_tensor = Temp.iloc[0]

    # tf_tensor = np.array(boundary_1_t)
    # tf_tensor = np.append(boundary_1_t, boundary_2_t)

    # Temp_1 = np.zeros(n_b_points)
    # Temp_2 = np.linspace(2.8000, 2.8000, n_b_points)

    x_tensor = np.linspace(0, 2.8, 280)

    # xf_tensor = np.array(Temp_1)
    # xf_tensor = np.append(xf_tensor, Temp_2)
    # x_tensor  = np.append(x_tensor, xf_tensor)
    
    # t_tensor = np.append(t_tensor, tf_tensor)
    

    v_tensor = np.array(initial_condition)
    # v_tensor = np.insert(v_tensor, 0, 0, axis=0)
    # v_tensor = np.insert(v_tensor, len(v_tensor), 0, axis=0)
    # vf_tensor = np.array(boundary_1_v)
    # print(vf_tensor)
    # vf_tensor = np.append(vf_tensor, boundary_2_v)
    # v_tensor = np.append(v_tensor, vf_tensor)
    v_tensor = torch.tensor(v_tensor)
    
    v_tensor.flatten(0)
    XT_u_tensor = torch.vstack([torch.tensor(x_tensor, requires_grad=True), torch.tensor(t_tensor, requires_grad=True), v_tensor])
    XT_u_tensor = XT_u_tensor.T

    return XT_u_tensor

pass

def fetch_boundary_spatial_temporal():
    df = pd.read_excel("19B-Data.xlsx", index_col = None, header = 0)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.set_index(df['Date'])
    df = df.sort_index()

    Day_1 = df[df['Date'] == '2013-09-18']
    Day_1 = Day_1.set_index(Day_1['Tripnumber'])
    Day_1 = Day_1.sort_index()
    
    # For first trip to start at time t=0
    Day_Temp = Day_1.copy()
    val = (Day_Temp['Time (seconds)'].iloc[0]).copy()
    Day_Temp['Time (seconds)'] = Day_Temp['Time (seconds)'] - val
    
    # Dropping column which aren't required
    Day_Temp_only_sections = Day_Temp.copy()
    Day_Temp_only_sections = Day_Temp_only_sections.drop("Date", axis=1)
    Day_Temp_only_sections = Day_Temp_only_sections.drop("Day", axis=1)
    Day_Temp_only_sections = Day_Temp_only_sections.drop("Tripnumber", axis=1)
    Day_Temp_only_sections = Day_Temp_only_sections.drop("Time" , axis=1)

    Prev = np.array(Day_Temp_only_sections['Time (seconds)'])

    d = np.array(Day_Temp_only_sections.drop(['Time (seconds)'], axis=1))

    # By taking maximum speed as 60 Km/h clipping the data where it is greater than that
    mask = d < 100/16.6

    d[mask] = 100/16.6

    Day_Temp_only_sections = Day_Temp_only_sections.drop(['Time (seconds)'], axis=1)
    Day_Temp_only_sections = pd.DataFrame(d, columns = Day_Temp_only_sections.columns)

    # Taking first trip as initial condition
    initial_condition = Day_Temp_only_sections.iloc[0]
    initial_condition = 100 / initial_condition



    # # boundary_1_v = Day_Temp_only_sections['Section 1']
    # boundary_1_v = np.zeros(n_b_points)
    # # boundary_2_v = Day_Temp_only_sections['Section 280']
    # boundary_2_v = np.zeros(n_b_points)

    for column in Day_Temp_only_sections.columns:
        Day_Temp_only_sections[column] = Day_Temp_only_sections[column] + Prev
        Prev = Day_Temp_only_sections[column]
    
    max_val = 0
    for column in Day_Temp_only_sections.columns:
        temp = max(Day_Temp_only_sections[column])
        if temp > max_val:
            max_val = temp
    
    
    Temp = (Day_Temp_only_sections *5) / max_val

    boundary_1_t = Temp['Section 1']
    boundary_2_t = Temp['Section 280']

    # t_tensor = np.zeros(282)
    # t_tensor = Temp.iloc[0]

    # tf_tensor = np.array(boundary_1_t)
    tf_tensor = np.append(boundary_1_t, boundary_2_t)
    n_b_points = Temp.shape[0]
    Temp_1 = np.zeros(n_b_points)
    Temp_2 = np.linspace(2.8000, 2.8000, n_b_points)

    # x_tensor = np.linspace(0, 2.8, 282)

    xf_tensor = np.array(Temp_1)
    xf_tensor = np.append(xf_tensor, Temp_2)
    # x_tensor  = np.append(x_tensor, xf_tensor)
    
    # t_tensor = np.append(t_tensor, tf_tensor)
    

    # v_tensor = np.array(initial_condition)
    # v_tensor = np.insert(v_tensor, 0, 0, axis=0)
    # v_tensor = np.insert(v_tensor, len(v_tensor), 0, axis=0)
    # vf_tensor = np.array(boundary_1_v)
    # # print(vf_tensor)
    # vf_tensor = np.append(vf_tensor, boundary_2_v)
    # v_tensor = np.append(v_tensor, vf_tensor)
    # v_tensor = torch.tensor(v_tensor)
    
    # v_tensor.flatten(0)
    XT_u_tensor = torch.vstack([torch.tensor(xf_tensor, requires_grad=True), torch.tensor(tf_tensor, requires_grad=True)])
    XT_u_tensor = XT_u_tensor.T

    return XT_u_tensor  

pass


def fetch_data_with_boundary(n_b_points):

    df = pd.read_excel("19B-Data.xlsx", index_col = None, header = 0)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.set_index(df['Date'])
    df = df.sort_index()

    Day_1 = df[df['Date'] == '2013-09-18']
    Day_1 = Day_1.set_index(Day_1['Tripnumber'])
    Day_1 = Day_1.sort_index()
    
    # For first trip to start at time t=0
    Day_Temp = Day_1.copy()
    val = (Day_Temp['Time (seconds)'].iloc[0]).copy()    
    Day_Temp['Time (seconds)'] = Day_Temp['Time (seconds)'] - val
    
    # Dropping column which aren't required
    Day_Temp_only_sections = Day_Temp.copy()
    Day_Temp_only_sections = Day_Temp_only_sections.drop("Date", axis=1)
    Day_Temp_only_sections = Day_Temp_only_sections.drop("Day", axis=1)
    Day_Temp_only_sections = Day_Temp_only_sections.drop("Tripnumber", axis=1)
    Day_Temp_only_sections = Day_Temp_only_sections.drop("Time" , axis=1)

    Prev = np.array(Day_Temp_only_sections['Time (seconds)'])

    d = np.array(Day_Temp_only_sections.drop(['Time (seconds)'], axis=1))

    # By taking maximum speed as 60 Km/h clipping the data where it is greater than that
    mask = d < 100/16.6

    d[mask] = 100/16.6

    Day_Temp_only_sections = Day_Temp_only_sections.drop(['Time (seconds)'], axis=1)
    Day_Temp_only_sections = pd.DataFrame(d, columns = Day_Temp_only_sections.columns)

    # Taking first trip as initial condition
    initial_condition = Day_Temp_only_sections.iloc[0]
    initial_condition = 100 / initial_condition



    boundary_1_v = Day_Temp_only_sections['Section 1']
    # boundary_1_v = np.zeros(n_b_points)
    boundary_2_v = Day_Temp_only_sections['Section 280']
    # boundary_2_v = np.zeros(n_b_points)

    for column in Day_Temp_only_sections.columns:
        Day_Temp_only_sections[column] = Day_Temp_only_sections[column] + Prev
        Prev = Day_Temp_only_sections[column]
    
    max_val = 0
    for column in Day_Temp_only_sections.columns:
        temp = max(Day_Temp_only_sections[column])
        if temp > max_val:
            max_val = temp
    
    
    Temp = (Day_Temp_only_sections *5) / max_val

    # boundary_1_t = np.linspace(0, 2, n_b_points)
    # boundary_2_t = np.linspace(0, 2, n_b_points)
    boundary_1_t = np.array(Temp['Section 1'])
    boundary_2_t = np.array(Temp['Section 280'])

    t_tensor = np.zeros(280)
    # t_tensor = Temp.iloc[0]

    # tf_tensor = np.array(boundary_1_t)
    tf_tensor = np.append(boundary_1_t, boundary_2_t)

    Temp_1 = np.zeros(Temp.shape[0])
    Temp_2 = np.linspace(2.8000, 2.8000, Temp.shape[0])

    x_tensor = np.linspace(0, 2.8, 280)

    xf_tensor = np.array(Temp_1)
    xf_tensor = np.append(xf_tensor, Temp_2)
    x_tensor  = np.append(x_tensor, xf_tensor)
    
    t_tensor = np.append(t_tensor, tf_tensor)
    

    v_tensor = np.array(initial_condition)
    # v_tensor = np.insert(v_tensor, 0, 0, axis=0)
    # v_tensor = np.insert(v_tensor, len(v_tensor), 0, axis=0)
    vf_tensor = np.array(boundary_1_v)
    # print(vf_tensor)
    vf_tensor = np.append(vf_tensor, boundary_2_v)
    v_tensor = np.append(v_tensor, vf_tensor)
    v_tensor = torch.tensor(v_tensor)
    
    v_tensor.flatten(0)
    XT_u_tensor = torch.vstack([torch.tensor(x_tensor, requires_grad=True), torch.tensor(t_tensor, requires_grad=True), v_tensor])
    XT_u_tensor = XT_u_tensor.T

    return XT_u_tensor

pass
