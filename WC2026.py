#!/usr/bin/env python

# -*- coding: utf-8 -*-
import tom_model as TM 
import pandas as pd

from copy import deepcopy

from collections import Counter

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

        for iteration in iterations:
            self.total_table, self.total_placements = TM.SimulateGroupStage(self.schedule,
                                                                            self.table,
                                                                            self.total_table,
                                                                            self.total_placements,                                                      
                                                                            50
                                                                            )

        print(self.total_table)
        print('\n')
        for group in self.groups:
            print(self.total_placements[group])
            print('\n')

    def filter_group(self, table, group):

        group_table = table[table['Group'] == group]
        return group_table




def main():

    schedule_file = 'wc_2026_schedule'
    table_file = 'wc_2026_info'

    WC_2026(schedule_file, table_file, iterations=1000)

if __name__ == "__main__":

    main()
