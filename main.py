import myutils
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer


def main():
    injury_page = pd.read_csv("csv_files/injury_report_24.csv")
    pitch_breakdown = pd.read_csv("csv_files/fangraphs_leaderboard_all_pitchers.csv")

    injured_pitchers = injury_page[injury_page["Pos"] == "SP"]
    injured_pitchers_with_age = pd.merge(injured_pitchers, pitch_breakdown, on="Name", how="inner")

    merged = pd.merge(
        pitch_breakdown,
        injured_pitchers,
        on="Name",        
        how="left",       # keep all pitchers
        indicator=True    # adds a "_merge" column
    )

    # add classification of 1 if injured, 0 if not
    pitch_breakdown["injury_label"] = merged["_merge"].apply(lambda x: 1 if x == "both" else 0)

    features = [
        'Age',
        'IP',
        'vFA (pi)',
        'FB%',
        'SL%',
        'CH%',
    ]

    X = pitch_breakdown[features]
    y = pitch_breakdown["injury_label"] 

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('model', LogisticRegression(max_iter=1000))  # use more iterations if it doesn't converge
    ])

    # Fit the pipeline
    pipeline.fit(X_train, y_train)

    initial_state = myutils.get_user_input()

    value, best_plan = myutils.expectimax_with_path(
        initial_state,
        depth=6,
        is_chance_node=False,
        pipeline=pipeline
    )

    print(f"Expected total innings: {value:.2f}")
    print("Best action sequence:", best_plan)


if __name__ == "__main__":
    main()