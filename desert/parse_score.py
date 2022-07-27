import os
import numpy as np
import pandas as pd
import openpyxl

class ParseScore():
    def __init__(self):
        pass
    
    def segment_sample(self):
        deep_synergy_df = pd.read_excel('./DeepSynergy/DrugCombo-screen.xlsx', index_col=0)
        import pdb; pdb.set_trace()
        if os.path.exists('./OncoPoly') == False:
            os.mkdir('./OncoPoly')
        ### MAP CELL LINE LIST
        cell_line_list = list(deep_synergy_df['cell_line'])
        sorted_cell_line_list = sorted(list(set(cell_line_list)))
        cell_line_num_name_list = ['Cell' + str(x) for x in range(1, len(sorted_cell_line_list)+1)]
        cell_line_num_name_dict = dict(zip(sorted_cell_line_list, cell_line_num_name_list))
        ### MAP DRUG NAME
        drugA_list = list(deep_synergy_df['drugA_name'])
        drugB_list = list(deep_synergy_df['drugB_name'])
        sorted_drug_list = sorted(list(set(drugA_list + drugB_list)))
        drug_num_name_list = ['Drug' + str(x) for x in range(1, len(sorted_drug_list)+1)]
        drug_num_name_dict = dict(zip(sorted_drug_list, drug_num_name_list))
        ### REPLACE [Cell#] [Drug#]
        deep_synergy_df = deep_synergy_df.replace({'cell_line': cell_line_num_name_dict})
        deep_synergy_df = deep_synergy_df.replace({'drugA_name': drug_num_name_dict})
        deep_synergy_df = deep_synergy_df.replace({'drugB_name': drug_num_name_dict})
        ### AGGREGATE COLUMNS
        deep_synergy_df['sample_name'] = deep_synergy_df['drugA_name'] + '.' + deep_synergy_df['drugB_name'] + '.' + deep_synergy_df['cell_line']

        ### SELECT SAMPLES
        sample_name_list = sorted(list(set(list(deep_synergy_df['sample_name']))))
        k = 10
        sample_name_klist = []
        sample_name_arraylist = (np.array_split(sample_name_list, k))
        for i in range(k): 
            ith_onco_poly_path = './OncoPoly/' + str(i+1) + 'th_fold'
            if os.path.exists(ith_onco_poly_path) == False:
                os.mkdir(ith_onco_poly_path)
            sample_name_tmplist = sample_name_arraylist[i].tolist()
            for selected_sample_name_list in sample_name_tmplist:
                selected_drugA_num_name = selected_sample_name_list.split('.')[0]
                selected_drugB_num_name = selected_sample_name_list.split('.')[1]
                selected_deep_synergy_df = deep_synergy_df.loc[deep_synergy_df['sample_name'] == selected_sample_name_list]
                drugA_conc_list = list(set(selected_deep_synergy_df['drugA Conc (µM)']))
                drugA_conc_strlist = [str(x) for x in drugA_conc_list]
                drugB_conc_list = list(set(selected_deep_synergy_df['drugB Conc (µM)']))
                drugB_conc_strlist = [str(x) for x in drugB_conc_list]
                drugAB_conc_list = list(selected_deep_synergy_df['X/X0'])
                drugAB_conc_matrix = np.array(drugAB_conc_list).reshape(len(drugA_conc_list), len(drugA_conc_list))
                drugAB_matrix_df = pd.DataFrame(drugAB_conc_matrix, columns = drugB_conc_strlist)
                # FILL THE OTHER PLACES WITH ['None']
                drugAB_matrix_df.insert(0, 'None', drugA_conc_strlist)
                agent1_nonelist = ['(=Agent1)']
                agent1_nonelist += ['None'] * len(drugA_conc_list)
                agent1_dict = dict(zip(list(drugAB_matrix_df.columns), agent1_nonelist))
                # SETTLED FILLS AND ['None']
                st_agent1 = ['Agent1', selected_drugA_num_name] + ['None'] * (len(drugA_conc_list) - 1)
                st_agent1_dict = dict(zip(list(drugAB_matrix_df.columns), st_agent1))
                st_agent2 = ['Agent2', selected_drugB_num_name] + ['None'] * (len(drugA_conc_list) - 1)
                st_agent2_dict = dict(zip(list(drugAB_matrix_df.columns), st_agent2))
                st_unit1 = ['Unit1', '\muM'] + ['None'] * (len(drugA_conc_list) - 1)
                st_unit1_dict = dict(zip(list(drugAB_matrix_df.columns), st_unit1))
                st_unit2 = ['Unit2', '\muM'] + ['None'] * (len(drugA_conc_list) - 1)
                st_unit2_dict = dict(zip(list(drugAB_matrix_df.columns), st_unit2))
                st_title = ['Title', selected_sample_name_list] + ['None'] * (len(drugA_conc_list) - 1)
                st_title_dict = dict(zip(list(drugAB_matrix_df.columns), st_title))
                # COMBINE ALL ROWS NEED TO FILLED
                drugAB_matrix_df = drugAB_matrix_df.append(agent1_dict, ignore_index=True)
                drugAB_matrix_df = drugAB_matrix_df.append(st_agent1_dict, ignore_index=True)
                drugAB_matrix_df = drugAB_matrix_df.append(st_agent2_dict, ignore_index=True)
                drugAB_matrix_df = drugAB_matrix_df.append(st_unit1_dict, ignore_index=True)
                drugAB_matrix_df = drugAB_matrix_df.append(st_unit2_dict, ignore_index=True)
                drugAB_matrix_df = drugAB_matrix_df.append(st_title_dict, ignore_index=True)
                length_df = drugAB_matrix_df.shape[0]
                drugAB_matrix_df['(=Agent2)'] = ['None'] * length_df
                # REPLACE ['None']
                drugAB_matrix_none_df = drugAB_matrix_df.replace('None', '', regex=True)
                drugAB_matrix_none_df = drugAB_matrix_none_df.rename(columns={'None': ''})
                # SAVE DATAFRAME
                ith_onco_poly_path_sample = './OncoPoly/' + str(i+1) + 'th_fold/' + selected_sample_name_list
                if os.path.exists(ith_onco_poly_path_sample) == False:
                    os.mkdir(ith_onco_poly_path_sample)
                
                import pdb; pdb.set_trace()
                ith_onco_poly_path_sample_file = ith_onco_poly_path_sample + '/' + selected_sample_name_list + '.xls'
                drugAB_matrix_none_df.to_excel(ith_onco_poly_path_sample_file, index=False)


ParseScore().segment_sample()