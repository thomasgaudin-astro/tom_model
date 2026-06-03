import numpy as np
import pandas as pd

def read_init_files(schedule_file, table_file, r32=False, r32_file=None):

    schedule_filename = f'./Data/{schedule_file}.csv'
    table_filename = f'./Data/{table_file}.csv'

    match_schedule = pd.read_csv(schedule_filename, header=0, index_col=0)
    table = pd.read_csv(table_filename, header=0, index_col=0)

    if (r32 is True) & (r32_file is not None):

        r32_filename = f'./Data/{r32_file}.csv'
        r32_rules = pd.read_csv(r32_filename)
        
        return match_schedule, table, r32_rules

    else:
        return match_schedule, table

def make_placement_table(total_group_table):

    #create a placement table
    num_teams = len(total_group_table.index)
    placement_table = pd.DataFrame(np.zeros((num_teams, num_teams)),
                                    index=range(1, num_teams+1), 
                                    columns=list(total_group_table.index)
                                    )

    return placement_table