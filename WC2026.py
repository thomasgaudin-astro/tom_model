#!/usr/bin/env python

# -*- coding: utf-8 -*-
import tom_model as TM 
import pandas as pd

from copy import deepcopy

from collections import Counter

from tqdm import tqdm

class WC_2026:

    def __init__(self, schedule, table, iterations=1):

        #define match schedule and table for each group 
        self.schedule, self.table = TM.read_init_files(schedule, table)

        #read number of groups
        self.groups = Counter(self.table.loc[:, 'Group']).keys()

        #initialize total table for sum of all interations
        self.total_table = deepcopy(self.table)

        #initialize total placement table for each group
        self.total_placements = {}
        for group in self.groups:

            group_table = self.filter_group(self.table, group)
            self.total_placements[group] = TM.make_placement_table(group_table)

        #total third place initialization
        self.total_third_place = deepcopy(self.total_table.loc[:, ['Team', 'Points', 'Group']])
        self.total_third_place['Num Appearances'] = 0

        print(f'Simulating {iterations} World Cups.')
        for iteration in tqdm(range(iterations)):
            gs = TM.SimulateGroupStage(self.schedule,
                                        self.table,
                                        self.total_table,
                                        self.total_placements,                                                      
                                        50,
                                        third='Y'
                                        )

            # print(gs.third_place)

            # add third place team appearances and points to total table 
            for team in gs.third_place.index:
                tp_points = gs.third_place.loc[team, 'Points']

                self.total_third_place.loc[team, 'Num Appearances'] += 1
                self.total_group_table.loc[team, 'Points'] += tp_points

        print(self.total_third_place)

        for group in self.groups:

            #calculate the average placement table for each group then plot
            average_placement_table = self.total_placements[group] / iterations    
            TM.plot_avg_placements('WC2026', average_placement_table, group)

            #calculate the average table for each group then plot
            total_group_table = self.total_table[self.total_table['Group'] == group] 
            average_group_table = deepcopy(total_group_table.loc[:, ['Team', 'Points']])

            average_group_table['Points'] =  average_group_table['Points'] / iterations

        #calculate the average third place table and print



    def filter_group(self, table, group):

        group_table = table[table['Group'] == group]
        return group_table




def main():

    schedule_file = 'wc_2026_schedule'
    table_file = 'wc_2026_info'

    WC_2026(schedule_file, table_file, iterations=1000)

if __name__ == "__main__":

    main()
