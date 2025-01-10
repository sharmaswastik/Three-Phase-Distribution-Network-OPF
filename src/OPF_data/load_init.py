import os
import pandas as pd
import numpy as np
from glob import glob

def excel_sheet_load(PLoad_dict, QLoad_dict):
    cwd = os.getcwd()
    path = r"OPF_data\load"
    path = os.path.join(cwd, path)    
    csv_files = glob(os.path.join(path, "*.csv"))

    for f in csv_files:
        df = pd.read_csv(f)
        df = df.dropna(axis=1)
        df = df.dropna(axis=0)
        df.drop(df.index , inplace=True)
        p_df = os.path.basename(f)
        p = p_df.split('-')[-1].split('.')[0]
        p = p.capitalize()

        # Ensure the DataFrame has at least one row
        if df.empty:
            df = pd.DataFrame({'Phase': [p]})

        df.loc[0, 'Phase'] = p

        if p_df[0] == 'P':
            for key, value in PLoad_dict.items():
                if key == p:
                    for k, v in value.items():
                        if isinstance(v, list):
                            if len(v) != len(df.index):
                                df = df.reindex(range(len(v)))
                        df['Phase'] = p
                        df[k] = v
                    df.to_csv(f, index=False)
        else:
            for key, value in QLoad_dict.items():
                if key == p:
                    for k, v in value.items():
                        if isinstance(v, list):
                            if len(v) != len(df.index):
                                df = df.reindex(range(len(v)))
                        df['Phase'] = p
                        if isinstance(v, list):
                            df[k] = [0.99 * val for val in v]
                        else:
                            df[k] = 0.99 * v
                    df.to_csv(f, index=False)

Pload_dict = {"A":{'Bus0':0, 'Bus1':0, 'Bus2':0}, "B":{'Bus0':0, 'Bus1':0, 'Bus2':0}, "C":{'Bus0':0, 'Bus1':0, 'Bus2':0}}

Qload_dict = {"A":{'Bus0':0, 'Bus1':0, 'Bus2':0}, "B":{'Bus0':0, 'Bus1':0, 'Bus2':0}, "C":{'Bus0':0, 'Bus1':0, 'Bus2':0}}

excel_sheet_load(Pload_dict, Qload_dict)

