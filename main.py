import time
import os
from src.autograder.methodology import add_grades_and_comments
from src.autograder.utils import process_submissions
from src.autograder.methodologyCOTRAG import (
    add_grades_and_comments_COTRAG,
    generate_data_store,
)
from src.autograder.methodologyReAct import add_grades_and_comments_ReAct
from src.autograder.methodologyCOT import add_grades_and_comments_COT
from src.autograder.methodologyCOTMultiCalls import (
    add_grades_and_comments_COTMultiCalls,
)
from src.autograder.logging import logger
import json


def compute_total_size_in_kb(folder_path):
    folder_path = folder_path.strip('"')
    total_size = 0
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            file_path = os.path.join(root, file)
            print(f"Processing file: {file_path}")  # Debug print
            total_size += os.path.getsize(file_path)
    total_size_in_kb = total_size / 1024
    return total_size_in_kb


def export_as_json(data, filename):
    with open(filename, "w") as f:
        json.dump(data, f, indent=4)


def apply_methodology(methodology, submissions, gradebook_path):

    if methodology == "M1":
        logger.info(
            "###############################################################################___________Applying Simple GPT prompt_______________############################################################"
        )
        return add_grades_and_comments(
            submissions,
            gradebook_path,
            assignment_name,
            possible_points,
            question_file_path,
            rubric_file_path,
        )
    elif methodology == "M2":
        logger.info(
            "###############################################################################___________Applying COT with RAG_______________############################################################"
        )
        return add_grades_and_comments_COTRAG(
            submissions,
            gradebook_path,
            assignment_name,
            possible_points,
            question_file_path,
            rubric_file_path,
        )
    elif methodology == "M3":
        logger.info(
            "###############################################################################___________Applying ReAct_______________############################################################"
        )
        return add_grades_and_comments_ReAct(
            submissions,
            gradebook_path,
            assignment_name,
            possible_points,
            question_file_path,
            rubric_file_path,
        )
    elif methodology == "M4":
        logger.info(
            "###############################################################################___________Applying COT_______________############################################################"
        )
        return add_grades_and_comments_COT(
            submissions,
            gradebook_path,
            assignment_name,
            possible_points,
            question_file_path,
            rubric_file_path,
        )
    elif methodology == "M5":
        logger.info(
            "###############################################################################___________Applying COT with Multi-Calls_______________############################################################"
        )
        return add_grades_and_comments_COTMultiCalls(
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
    total_size_kb = compute_total_size_in_kb(folder_path=folder_path)
    logger.info(f"Total size of files: {total_size_kb} KB")
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

        for methodology in ["M1", "M2", "M3", "M4", "M5"]:
            start_time = time.time()
            updated_gradebook_path, comments_list = apply_methodology(
                methodology, submissions_dict, gradebook_path
            )
            end_time = time.time()
            execution_time_seconds = end_time - start_time
            # Corrected calculation for average processing time per KB
            if total_size_kb > 0:
                avg_speed_per_kb = total_size_kb / execution_time_seconds
                logger.info(
                    f"Methodology {methodology} average processing time: {avg_speed_per_kb} kb per second."
                )
        else:
            logger.info(
                f"Methodology {methodology} completed in {execution_time_seconds} seconds. No file size to calculate average time per KB."
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
