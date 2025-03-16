import pandas as pd

# need to combine pitcher speed table and pitcher rotations table and pitchers innings pitched table 
# desired columns: pitch speed, rotations/pitch, % of time throwing fastball, average innings pitched (innings/games)
#   need to look at that in comparison with the injury table info

# from injury table: need functionto calculate how many games the pitcher missed (injury/surgery date - return date)

def calc_days_missed(injury_date, return_date):
    injury_day_month_year = injury_date.split("/")
    return_day_month_year = return_date.split("/")
    