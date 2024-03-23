from src.autograder.methodology import add_grades_and_comments
from src.autograder.utils import process_submissions
from src.autograder.methodologyCOT import (
    add_grades_and_comments_COT,
    generate_data_store,
)
from src.autograder.methodologyReAct import add_grades_and_comments_ReAct

from src.autograder.logging import logger
import json
import os


def export_as_json(data, filename):
    with open(filename, "w") as f:
        json.dump(data, f, indent=4)


if __name__ == "__main__":
    folder_path = input("Enter the path to the folder containing student submissions: ")
    gradebook_path = input("Enter the path to the folder containing the Gradebook: ")
    submissions_dict = process_submissions(folder_path)
    logger.info("Student submissions processed.")
    logger.info(f"Submissions dictionary: {submissions_dict}")
    if submissions_dict:
        # Add percentage grades and comments using M1
        updated_gradebook_path, comments_list = add_grades_and_comments(
            submissions_dict, gradebook_path
        )
    logger.info(comments_list)
    export_as_json(comments_list, "commentsM1.json")
    logger.info("Comments exported using Methodolgy 1 as commentsM1.json.")

    # Generate and store data in vector DB before processing submissions
    print("Generating and storing data in vector DB...")
    generate_data_store()
    if submissions_dict:
        # Add percentage grades and comments using M2
        updated_gradebook_path, comments_list = add_grades_and_comments_COT(
            submissions_dict, gradebook_path
        )
    logger.info(comments_list)
    export_as_json(comments_list, "commentsM2.json")
    logger.info("Comments exported using Methodolgy 2 as commentsM2.json.")

    # -----------ReAct--------------------#
    if submissions_dict:
        # Add percentage grades and comments using M1
        updated_gradebook_path, comments_list = add_grades_and_comments_ReAct(
            submissions_dict, gradebook_path
        )
    logger.info(comments_list)
    export_as_json(comments_list, "commentsM3.json")
    logger.info("Comments exported using Methodolgy 3 as commentsM3.json.")
