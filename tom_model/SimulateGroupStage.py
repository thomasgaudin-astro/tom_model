import pandas as pd

from collections import Counter

import .SimulateGroup

class SimulateGroupStage:

    def __init__(self, matches, points_table, 
                 total_table, total_placements, 
                 location, Km):

        #initiate
        self.groups = Counter(points_table.loc[:, 'Group']).keys()
        self.group_tables = {}
        self.placement_tables = {}

        #run SimulateGroup for all groups
        SimulateAllGroups(self.groups)

        #add results to total tables
        for group in self.groups:
            total_table.add(self.group_tables[group], fill_value=0)
            total_placements[group].add(self.placement_tables[group], fill_values=0)
    
    def SimulateAllGroups(self, self.groups):

        #for each group, simulate all games and append to game dictionary
        for group in self.groups:

            group_results = SimulateGroup(matches, points_table, location, Km, group)

            self.group_tables[group] = group_results.group_table
            self.placement_tables[group] = group_results.placement_table

    

        


            

