import pandas as pd
from data_preprocessing import Campus_Lunch_Data_Processing, Campus_Health_Office_Data_Processing, Campus_Hygiene_Data_Processing, Campus_Income_Data_Processing
from data_preprocessing import Campus_Outcome_Data_Processing, Campus_Lunch_Data_Processing, Campus_Space_Data_Processing, Afterschool_Program_Data_Processing
from processed_data_merge import merge_data

df_Campus_Hygiene = Campus_Hygiene_Data_Processing(pd.read_csv("data/2023_campus_hygiene.csv"))
df_Afterschool_Program = Afterschool_Program_Data_Processing(pd.read_csv("data/2025_afterschool_program.csv"))
df_Campus_HealthOffice = Campus_Health_Office_Data_Processing(pd.read_csv("data/2025_campus_healthoffice.csv"))
df_Campus_Income = Campus_Income_Data_Processing(pd.read_csv("data/2025_campus_income.csv"))
df_Campus_Outcome = Campus_Outcome_Data_Processing(pd.read_csv("data/2025_campus_Outcome.csv"))
df_Campus_Lunch = Campus_Lunch_Data_Processing(pd.read_csv("data/2025_campus_lunch.csv"))
df_Campus_Space = Campus_Space_Data_Processing(pd.read_csv("data/2025_campus_space.csv"))

df = merge_data(df_Campus_Hygiene,df_Afterschool_Program,df_Campus_HealthOffice,df_Campus_Income,df_Campus_Outcome,df_Campus_Lunch,df_Campus_Space)
df.to_csv("data/merge_data.csv")