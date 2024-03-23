from src.autograder.methodology import add_grades_and_comments
from src.autograder.utils import process_submissions
from src.autograder.methodologyCOT import (
    add_grades_and_comments_COT,
    generate_data_store,
)
from src.autograder.methodologyReAct import add_grades_and_comments_ReAct

from src.autograder.logging import logger
import json


def export_as_json(data, filename):
    with open(filename, "w") as f:
        json.dump(data, f, indent=4)


def apply_methodology(methodology, submissions, gradebook_path):
    if methodology == "M1":
        return add_grades_and_comments(
            submissions,
            gradebook_path,
            assignment_name,
            possible_points,
            question_file_path,
            rubric_file_path,
        )
    elif methodology == "M2":
        return add_grades_and_comments_COT(
            submissions,
            gradebook_path,
            assignment_name,
            possible_points,
            question_file_path,
            rubric_file_path,
        )
    elif methodology == "M3":
        return add_grades_and_comments_ReAct(
            submissions,
            gradebook_path,
            assignment_name,
            possible_points,
            question_file_path,
            rubric_file_path,
        )
    else:
        logger.error("Unknown methodology.")
        return None, None


if __name__ == "__main__":
    folder_path = input("Enter the path to the folder containing student submissions: ")
    gradebook_path = input("Enter the path to the folder containing the Gradebook: ")
    assignment_name = input("Enter the Assignment Name: ")
    possible_points = int(input("Enter the total points possible: "))
    question_file_path = input(
        "Enter the path to the text file containing the question: "
    ).strip('"')
    rubric_file_path = input(
        "Enter the path to the text file containing the rubric: "
    ).strip('"')
    submissions_dict = process_submissions(folder_path)

    if not submissions_dict:
        logger.error("No submissions found.")
    else:
        logger.info("Student submissions processed.")
        # Pre-process for M2 if needed
        print("Generating and storing data in vector DB for Methodology 2...")
        generate_data_store()

        for methodology in ["M1", "M2", "M3"]:
            updated_gradebook_path, comments_list = apply_methodology(
                methodology, submissions_dict, gradebook_path
            )
            if comments_list:
                comments_file = f"comments{methodology}.json"
                export_as_json(comments_list, comments_file)
                logger.info(
                    f"Comments exported using Methodology {methodology} as {comments_file}."
                )
            else:
                logger.error(
                    f"Failed to process submissions using Methodology {methodology}."
                )
