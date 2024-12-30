import pandas as pd
import os

def base_data():
    base_file_path = os.path.join("koreaculture/data/sample_data.csv")
    code_file_path = os.path.join("koreaculture/data/code_book.csv")
    base_df = pd.read_csv(base_file_path)
    code_df = pd.read_csv(code_file_path)
    id_df = base_df.iloc[:,0:1]
    user_culture_df = base_df.iloc[:,1:92]
    user_culture_df = pd.concat([id_df,user_culture_df], axis=1)
    user_ranking_df = base_df.iloc[:,92:97]
    user_ranking_df = pd.concat([id_df,user_ranking_df], axis=1)
    user_cost_df = base_df.iloc[:,97:102]
    user_cost_df = pd.concat([id_df,user_cost_df], axis=1)
    user_area_df = base_df.iloc[:,102:103]
    user_area_df = pd.concat([id_df,user_area_df], axis=1)
    code_service_df = code_df.iloc[0:3,:]
    service_code_val_dict = code_service_df.set_index(code_service_df.columns[0])[code_service_df.columns[1]].to_dict()
    code_culture_df = code_df.iloc[3:95,:]
    culture_code_val_dict_non = code_culture_df.set_index(code_culture_df.columns[0])[code_culture_df.columns[1]].to_dict()
    culture_code_val_dict = {key: value for key, value in culture_code_val_dict_non.items() if key != 'Q1_99'}
    code_area_df = code_df.iloc[95:112,:]
    area_code_val_dict = code_area_df.set_index(code_area_df.columns[0])[code_area_df.columns[1]].to_dict()
    
    data_dict = {
        "user_culture_df":user_culture_df,
        "user_ranking_df":user_ranking_df,
        "user_cost_df":user_cost_df,
        "user_area_df":user_area_df,
        "code_service_df":code_service_df,
        "service_code_val_dict":service_code_val_dict,
        "code_culture_df":code_culture_df,
        "culture_code_val_dict":culture_code_val_dict,
        "code_area_df":code_area_df,
        "area_code_val_dict":area_code_val_dict
    }
    
    return data_dict