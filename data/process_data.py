import sys

sys.path.append('/Users/diogomatoschaves/PycharmProjects/disaster-response-analysis')

import pandas as pd
from sqlalchemy import create_engine
from disaster_response_pipeline.utils import TABLE_NAME


def load_data(messages_filepath, categories_filepath):
    """
    Loads the messages and categories csv's onto data frames

    :param messages_filepath: filepath for the csv containing messages
    :param categories_filepath: filepath for the csv containing categories
    :return: the messages and categories data frames
    """

    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)

    return messages, categories


def clean_data(messages, categories):
    """
    Cleans and transforms the input messages and categories

    :param messages: data frame containing the messages
    :param categories: data frame containing the categories
    :return: the cleaned and merged data frame
    """

    # Split categories into separate category columns.
    categories = pd.concat(
        [categories["id"], categories["categories"].str.split(";", expand=True)], axis=1
    )

    row = categories.iloc[0, :]

    cat_ids = [val.split("-")[0] for val in row.values[1:]]
    category_colnames = ["id", *cat_ids]
    categories.columns = category_colnames

    # Convert category values to just numbers 0 or 1.
    for column in categories:
        if column == "id":
            continue
        # set each value to be the last character of the string
        categories[column] = categories[column].apply(
            lambda x: x.split("-")[1] if not pd.isnull(x) and "-" in x else x
        )

        # convert column from string to numeric
        categories[column] = pd.to_numeric(categories[column])

        categories[column] = categories[column].apply(lambda x: 1 if x and x > 1 else x)

    df = messages.merge(categories, on="id")

    df.drop_duplicates(inplace=True)

    return df


def save_data(df, database_filename):
    """
    Saves the cleaned and merged dataset onto a sqlite database

    :param df: data frame to be saved
    :param database_filename: filepath pointing to the database to be saved
    :return: None
    """

    engine = create_engine(f"sqlite:///{database_filename}")
    df.to_sql(TABLE_NAME, engine, index=False, if_exists='replace')


def main():
    """
    Main thread of the program where the data cleaning pipeline is executed,
    and calls all the sub methods in sequence.

    :return: None
    """
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print(
            "Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}".format(
                messages_filepath, categories_filepath
            )
        )
        messages, categories = load_data(messages_filepath, categories_filepath)

        print("Cleaning data...")
        df = clean_data(messages, categories)

        print("Saving data...\n    DATABASE: {}".format(database_filepath))
        save_data(df, database_filepath)

        print("Cleaned data saved to database!")

    else:
        print(
            "Please provide the filepaths of the messages and categories "
            "datasets as the first and second argument respectively, as "
            "well as the filepath of the database to save the cleaned data "
            "to as the third argument. \n\nExample: python process_data.py "
            "disaster_messages.csv disaster_categories.csv "
            "DisasterResponse.db"
        )


if __name__ == "__main__":
    main()
