import pandas as pd
from datetime import datetime
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import numpy as np
import random

def calc_days_missed(injury_date, return_date):
    """
    calculate the number of days between two dates.

    Args:
        injury_date (str): the date the player is injured in 'MM/DD/YYYY' format
        return_date (str): the date the player returns in 'MM/DD/YYYY' format

    Returns:
        int: the absolute difference in days between the two dates

    Notes:
        - uses the datetime module for date parsing and calculations
        - assumes valid date inputs in 'MM/DD/YYYY' format
    """
    format_str = "%m/%d/%Y"
    date1 = datetime.strptime(injury_date, format_str)
    date2 = datetime.strptime(return_date, format_str)
    
    return abs((date1 - date2).days)

def calc_heuristic(total_pitches, pitch_pct_dict, velo_dict, spin_dict, age):
    # define hard-coded values:
    max_pitch_limit = 3200
    max_velo_dict = {"4-seam": 96.9, "slider": 90.6, "changeup": 89.9, "curve": 83.2, "sinker": 97.0, "cutter": 95.3, "splitter": 89.6}
    max_spin_dict = {"4-seam": 2557, "slider": 2982, "changeup": 2477, "curve": 3285, "sinker": 2968, "cutter": 2742, "splitter": 2007}
    ideal_age = 25
    pitch_num_coeff = 2
    velo_and_spin_coeff = 1
    age_coeff = 0.5
    # heuristic function
    workload_risk = total_pitches/max_pitch_limit
    stress_factor = sum(pitch_pct_dict[key](velo_dict[key] / max_velo_dict[key]) for key in max_velo_dict)
    age_penalty = 1 + (age - ideal_age)/10

    heuristic = pitch_num_coeff * workload_risk + velo_and_spin_coeff * stress_factor + age_coeff * age_penalty
    return heuristic

def calc_cost_function(avg_pitches, games_played, pitch_pct_dict, velo_dict, spin_dict, age):
    # define hard-coded values:
    avg_pitch_limit = 100
    max_velo_dict = {"4-seam": 96.9, "slider": 90.6, "changeup": 89.9, "curve": 83.2, "sinker": 97.0, "cutter": 95.3, "splitter": 89.6}
    max_spin_dict = {"4-seam": 2557, "slider": 2982, "changeup": 2477, "curve": 3285, "sinker": 2968, "cutter": 2742, "splitter": 2007}
    ideal_age = 25
    pitch_num_coeff = 2
    velo_and_spin_coeff = 1
    age_coeff = 0.5
    # cost function
    workload_risk = sum((avg_pitches/max_pitches) for _ in range(games_played))
    stress_factor = sum(pitch_pct_dict[key](velo_dict[key] / max_velo_dict[key]) for key in max_velo_dict)
    age_penalty = 1 + (age - ideal_age)/10

    cost_function = pitch_num_coeff * workload_risk + velo_and_spin_coeff * stress_factor + age_coeff * age_penalty
    return cost_function

# BEGINNING OF CODE FOR ITERATION 3 

def injury_probability(state, pipeline):
    """
    Compute probability of injury from a state using trained logistic model.

    Args:
        state (dict): Contains pitcher info: age, innings, pitch mix, etc.
        model (sklearn): Trained logistic regression model.

    Returns:
        float: Probability between 0 and 1.
    """

    input_features = ['Age', 'IP', 'vFA (pi)', 'FB%', 'SL%', 'CH%']
    x = pd.DataFrame([state])[input_features]
    prob = pipeline.predict_proba(x)[0][1]

    # Apply fatigue penalty or rest bonus
    rest_days = state.get('rest_days', 0)
    fatigue_modifier = 1.0 - min(rest_days * 0.1, 0.3)  # up to 30% injury reduction
    adjusted_prob = prob * fatigue_modifier

    return min(max(adjusted_prob, 0), 1)

def injury_penalty(state):
    age = state["Age"]
    if age < 25:
        return -5
    elif age < 30:
        return -10
    else:
        return -15

def evaluate_state(state, pipeline=None):
    """
    Assigns a value to a final state (e.g., based on innings, no injury).
    """
    base_reward = state["IP"]

    # penalize high IP without rest (fatigue multiplier)
    fatigue_penalty = 0.1 * state.get('consecutive_starts', 0)
    base_reward -= fatigue_penalty

    if pipeline:
        p_injury = injury_probability(state, pipeline)
        penalty = (p_injury ** 2) * abs(injury_penalty(state))
        return base_reward - penalty
    return base_reward


def get_possible_actions(state):
    """
    Returns valid actions - empty list if injured
    """
    if state.get('injured', False):
        return []  # injured pitcher is out for season
    return ['rest'] + (['pitch'] if state.get('consecutive_starts', 0) < 5 else [])


def transition(state, action):
    """
    Simulate next state based on action.

    Returns a new state dict.
    """
    new_state = state.copy()
    if new_state.get('injured', False):
        return new_state
    vFA = new_state.get('vFA (pi)', 95.0)  # default velocity

    if action == 'pitch':
        new_state['IP'] += random.uniform(4.0, 7.0) # typical start length 
        new_state['rest_days'] = 0
        new_state['vFA (pi)'] = vFA - 0.3  # fatigue effect - lower velo
        new_state['consecutive_starts'] = state.get('consecutive_starts', 0) + 1
    elif action == 'rest':
        new_state['rest_days'] += 1
        new_state['vFA (pi)'] = min(vFA + 0.2, 98.0)  # recovery benefits (capped)
        new_state['consecutive_starts'] = 0
    elif action == 'injured':
        new_state['injured'] = True

    return new_state
    
    
def expectimax_with_path(state, depth, is_chance_node, pipeline):
    """
    Expectimax search for pitcher workload management.
    
    Args:
        state (dict): Pitcher's current state (Age, IP, velocity, pitch usage, etc.)
        depth (int): Remaining decisions to simulate
        is_chance_node (bool): True if evaluating injury chance
        pipeline: Trained model pipeline for injury probability

    Returns:
        float: Expected utility of this state
    """
    if depth == 0 or state.get('injured', False):
        return evaluate_state(state, pipeline), []

    if is_chance_node:
        p_injury = injury_probability(state, pipeline)

        injured_state = transition(state.copy(), 'injured')
        injury_value, _ = expectimax_with_path(injured_state, depth-1, False, pipeline)
        
        healthy_value, healthy_path = expectimax_with_path(state.copy(), depth-1, False, pipeline)
        
        expected_value = p_injury * injury_value + (1 - p_injury) * healthy_value
        return expected_value, healthy_path

    best_value = float('-inf')
    best_path = []
    for action in get_possible_actions(state):
        # force rest after 5 consecutive starts
        if state.get('consecutive_starts', 0) >= 5 and action == 'pitch':
            continue

        next_state = transition(state, action)
        value, path = expectimax_with_path(next_state, depth, True, pipeline)

        if value > best_value or (random.random() < 0.1):  # 10% of the time, exploration allowed
            best_value = value
            best_path = [action] + path

    return best_value, best_path

def get_user_input():
    """Prompt user to enter pitcher attributes with validation."""
    print("Enter pitcher's initial state:")
    
    state = {
        'Age': int(input("Age (e.g., 27): ") or 27),
        'IP': float(input("Innings Pitched (e.g., 30.0): ") or 30),
        'vFA (pi)': float(input("Fastball Velocity (e.g., 95.0): ") or 95),
        'FB%': int(input("Fastball Usage % (e.g., 50): ") or 50),
        'SL%': int(input("Slider Usage % (e.g., 25): ") or 25),
        'CH%': int(input("Changeup Usage % (e.g., 10): ") or 10),
        'rest_days': int(input("Recent Rest Days (e.g., 0): ") or 0),
        'consecutive_starts': int(input("Consecutive Starts (e.g., 0): ") or 0),
        'injured': False
    }
    
    # validate pitch mix sums to <= 100%
    total_pitch = state['FB%'] + state['SL%'] + state['CH%']
    if total_pitch > 100:
        raise ValueError("Pitch percentages exceed 100%")
    
    return state