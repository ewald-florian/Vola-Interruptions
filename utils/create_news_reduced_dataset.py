"""Create a reduced version of the RP news dataset.
1. news_reduced only contains the columns:
    "TIMESTAMP_UTC",
    "EVENT_TEXT",
    "EVENT_SIMILARITY_DAYS",
    "EVENT_RELEVANCE",
    "EVENT_SENTIMENT_SCORE"
2. news_reduced only contains samples where "EVENT_TEXT" is not NaN
"""
import pandas as pd

# Load combined news file
news_df = pd.read_csv("/news_old/news_combined.csv.gz")

# Convert the 'datetime_utc' column to datetime format
news_df["TIMESTAMP_UTC"] = pd.to_datetime(news_df["TIMESTAMP_UTC"],
                                          format='%d%b%y:%H:%M:%S.%f')

# Define relevant columns.
news_reduced = news_df.loc[:, ["TIMESTAMP_UTC", "ENTITY_NAME", "EVENT_TEXT",
                              "EVENT_SIMILARITY_DAYS", "EVENT_RELEVANCE",
                              "EVENT_SENTIMENT_SCORE"]]

# Drop rows where EVENT_TEXT is NaN
news_reduced.dropna(subset=["EVENT_TEXT"], inplace=True)

# Store news reduced to csv.
filepath = "/news/news_reduced.csv.gz"

news_reduced.to_csv(filepath, index=False, compression='gzip')