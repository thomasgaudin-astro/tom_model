import SimulateSingleGame

class SimulateGroup:

    def __init__(self, location, Km, elo_rank, init_elo_rank, points_table, init_points_table, 
                       total_table, placement_table, iterations=1, count=1):

        self.percent_finish, self.final_tab = SimulateGroup(matches, 
                                                            elo_rank, init_elo_rank,
                                                            points_table, init_points_table,
                                                            total_table, placement_table,
                                                            location, Km,
                                                            iterations=1, count=1
                                                            )
    
    def SimulateGroup(matches, elo_rank, init_elo_rank, points_table, init_points_table, 
                       total_table, placement_table, iterations=1, count=1
                       ):
                       
        #There are two ways to write country names. This is converter code between the two
        list_teams = [points_table['Team'][ind] for ind in points_table.index]
        #teams_list = [points_table.index[val][1] for val in range(len(points_table.index))]
                
        #teams_dict = {teams_list[i]: list_teams[i] for i in range(len(list_teams))}

        for num in range(iterations):

            for match in matches:

                #initialize home team and ELO
                home_team = match[0]
                home_elo = elo_rank[home_team]

                #initialize away team and ELO
                away_team = match[1]
                away_elo = elo_rank[away_team]

                #use SimulateSingleGame class to generate all math associated with a single game
                game = SimulateSingleGame(home_elo, away_elo, location, Km)

                #update Elo ratings
                elo_rank[home_team] = game.new_home_elo
                elo_rank[away_team] = game.new_away_elo

                #home win
                if game.home_outcome == 1:

                    #update table
                    points_table.loc[points_table['Team'] == home_team, ['Points']] += 3
                    points_table.loc[points_table['Team'] == away_team, ['Points']] += 0    

                #draw
                elif game.home_outcome == 0.5:

                    #update table
                    points_table.loc[points_table['Team'] == home_team, ['Points']] += 1
                    points_table.loc[points_table['Team'] == away_team, ['Points']] += 1

                #away win
                else:

                    #update table
                    points_table.loc[points_table['Team'] == home_team, ['Points']] += 0
                    points_table.loc[points_table['Team'] == away_team, ['Points']] += 3


            #create final table, append to total table, and reset to initial table
            final_table = deepcopy(points_table)

            for team in list_teams:
                final_points = final_table.loc[final_table['Team'] == team, 'Points'].values[0]
                total_table[team].append(final_points)

            points_table = deepcopy(init_points_table)

            #reset ELO rankings
            for team in elo_rank.keys():
                elo_rank[team] = deepcopy(init_elo_rank[team])

            #rank final table and append to placement table
            final_table['Rank'] = final_table['Points'].rank(ascending = False)
            for team in list_teams:
                rank = int(final_table.loc[final_table['Team'] == team, ['Rank']].values[0])
                placement_table.loc[rank, team] += 1

            print(count)
            count += 1
        
        average_final_table = deepcopy(final_table)
        
        #create average points gained table
        for team in list_teams:
            total_points = sum(total_table[team])

            average_final_table[team] = round(total_points / iterations, 1)

        avg_final_table = pd.DataFrame(average_final_table.items(),
                                    index = range(1, len(average_final_table.index)+1),
                                    columns = ['Team', 'Points'])
        
        final_tab = avg_final_table.sort_values('Points', ascending = False)

        #create table showing the percent chance to finish in each spot
        percent_finish = (1 / iterations) * placement_table
        
        #add average ppg column to final table
        final_tab['Average PPG'] = 0
        
        #calculate average ppg
        for team in list_teams:
            final_tab.loc[final_tab['Team'] == team, ['Average PPG']] = final_tab.loc[final_tab['Team'] == team, 
                                                                                ['Points']].values[0] / 14
            
        #add chance to qualify automatically tab to final table
        final_tab['Chance to Qualify'] = 0

        #calculate percent qualify and append to average points table
        for column in percent_finish:
            percent_qualify = sum(percent_finish.loc[0:3, column].values)

            final_tab.loc[final_tab['Team'] == column, ['Chance to Qualify']] = percent_qualify
        
        return percent_finish, final_tab
