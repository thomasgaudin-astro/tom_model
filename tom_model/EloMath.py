class ValueNotInListError(Exception):
    """Please pick between 'home', 'away', and 'neutral'."""
    pass

class EloMath:
    """
        Code to run all calculations needed for each match of the Tom Model
    """

    def __init__(self, home_Ro, away_Ro, location, Km, outcome=None, GD=1, theta=1.7):

        try:
            if location == 'neutral':
                self.home_we = calculate_We(home_Ro, away_Ro, location)
                self.away_we = calculate_We(away_Ro, home_Ro, location)
            elif (location == 'home') | (location == 'away'):
                self.home_we = calculate_We(home_Ro, away_Ro, 'home')
                self.away_we = calculate_We(away_Ro, home_Ro, 'away')
            else:
                raise ValueNotInList("Error. Location not in list of possibilities.")

            self.home_wp = davidson_home_wp(self.home_we, self.away_we, theta=theta)
            self.away_wp = davidson_away_wp(self.home_we, self.away_we, theta=theta)
            self.draw_prob = davidson_tie_prob(self.home_we, self.away_we, theta=theta)

            self.home_elo_wp = calculate_home_win_probability(self.home_we, self.away_we)
            self.away_elo_wp = calculate_away_win_probability(self.home_we, self.away_we)
            if outcome:
                self.new_home_elo = calculate_elo(home_Ro, 
                                                  self.home_we, 
                                                  outcome, Km, GD=1
                                                  )

                self.new_away_elo = calculate_elo(away_Ro, 
                                                  self.away_we,
                                                  outcome, Km, GD=1
                                                  )

        except ValueNotInListError:
            print('') #not sure what to do here, but this line needs to change


    def calculate_We(Ro, opponent_Ro, location):
        """ Calculate the We from the formula given by ELO.
            Inputs:
                Ro (int) -> Elo rating of team 
                opponent_Ro (int) -> Elo rating of opponent 
                location (str) -> either 'home', 'away', or 'neutral'
            Returns:
                Win Expectancy, type: float
        """
        
        #if the team is at home, calcuate difference in Elo with home boost
        if location == 'home':
            dr = (Ro + 100) - opponent_Ro
            
        #if the team is on the road, calculate difference in Elo, boost for oppoent
        elif location == 'away':
            dr = Ro - (opponent_Ro + 100)

        #if at a neutral venue, calculate difference in Elo
        elif location == 'neutral'
            dr = Ro - opponent_Ro

            
        #formula from eloratings.net
        We = 1 / ( (10 ** (-dr / 400)) + 1)

        return We

    def davidson_home_wp(home_We, away_We, theta=1.7):
    """ Calculates the probability of a win for any team given the win expectancy
        calculated from the difference in Elo for each team. Formula given by 
        Davidson (1970).
        Inputs:
            home_We (float) -> win expectancy for home team 
            away_We (float) -> win expectancy for away team 
            theta (float) -> fudge factor to make equations work, I have found 
                    that 1.7 gives realistic results
        Returns:
            Win Probability, type: float
    """
    
    hwp = home_We / (home_We + (theta * away_We) )
    
    return hwp

    def davidson_away_wp(home_We, away_We, theta=1.7):
    """ Calculates the probability of a loss for any team given the win expectancy
        calculated from the difference in Elo for each team. Formula given by 
        Davidson (1970).
        Inputs:
            home_We - win expectancy for home team (float)
            away_We - win expectancy for away team (float)
            theta - float fudge factor to make equations work, I have found 
                    that 1.7 gives realistic results
        Returns:
            Loss Probability, type: float
    """
    
    awp = away_We / ( (theta * home_We) + away_We)
    
    return awp

    def davidson_tie_prob(home_We, away_We, theta=1.7):
        """ Calculates the probability of a draw for any match given the win expectancy
            calculated from the difference in Elo for each team. Formula given by 
            Davidson (1970).
            Inputs:
                home_We (float) -> win expectancy for home team 
                away_We (float) -> win expectancy for away team 
                theta (float) -> fudge factor to make equations work, I have found 
                        that 1.7 gives realistic results
            Returns:
                Draw Probability, type: float
        """
        
        tie = ( (theta**2 - 1) * home_We * away_We ) / ((home_We + (theta * away_We) ) * ( (theta * home_We) + away_We))
        
        return tie

    def calculate_home_win_probability(home_Ro, away_Ro):
        """ Win probability formula from Elo ratings website. 
            Can be used, but doesn't work well.
            Inputs:
                Ro (int) -> Elo rating of team 
                opponent_Ro (int) -> Elo rating of opponent 
            Returns:
                Win Probability, type: float
        """

        wp = min((1 / (1 + 10**((away_Ro - home_Ro)/400)))**1.75 + 0.1, 1)

        return wp

    def calculate_away_win_probability(home_Ro, away_Ro):
        """ Loss probability formula from Elo ratings website. 
            Can be used, but doesn't work well.
            Inputs:
                Ro (int) -> Elo rating of team 
                opponent_Ro (int) -> Elo rating of opponent 
            Returns:
                Loss Probability, type: float
        """

        wp = max((1 / (1 + 10**((home_Ro - away_Ro)/400)))**1.75 - 0.1, 0)

        return wp

    def calculate_elo(Ro, We, WLD, Km, GD=1):
        """ ELO formula used for calculation of new Elo after a match.
            Can calculate real Elo if your code simulates goals scored.
            Inputs:
                Ro (int) -> pre-match Elo 
                We (float) -> team win expectancy for match 
                WDL (float) -> Determined by the outcome generater function
                    1.0 for win, 0.5 for draw, 0.0 for loss 
                Km (int) -> tournament weight constant from eloratings.net 
                GD (int) -> Goal Difference of match, defaults to 1 
            Returns:
                Post-match Elo, type: float
        """
        
        #Adjust weight constant based on match GD
        if GD < 2:
            GDM = Km

        elif GD == 2:
            GDM = (1.5 * Km)

        elif GD == 3:
            GDM = (1.75 * Km)

        elif GD >= 4:
            GDM = (Km * (1.75 + (GD - 3) / 8 ))
            
        #calculate new Elo
        Rn = Ro + (GDM * (WLD - We))

        return Rn
