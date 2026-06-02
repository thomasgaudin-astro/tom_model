import pandas as pd

from collections import Counter

from .SimulateGroup import SimulateSingleGroup

class SimulateGroupStage:

    def __init__(self, matches, points_table, 
                 total_table, total_placements, 
                 Km):

        #initiate
        self.groups = Counter(points_table.loc[:, 'Group']).keys()
        self.group_tables = {}
        self.placement_tables = {}

        #run SimulateGroup for all groups
        self.SimulateAllGroups(self.groups)

        #add results to total tables
        for group in self.groups:
            total_table.add(self.group_tables[group], fill_value=0)
            total_placements[group].add(self.placement_tables[group], fill_values=0)
    
    def SimulateAllGroups(self, groups):

        #for each group, simulate all games and append to game dictionary
        for group in groups:

            group_results = SimulateSingleGroup(matches, points_table, location, Km, group)

            self.group_tables[group] = group_results.group_table
            self.placement_tables[group] = group_results.placement_table

    

        


            

