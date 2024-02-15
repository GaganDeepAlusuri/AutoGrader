from src.autograder.utils import (
    process_submissions,
    add_grades_and_comments,
    generate_student_reports,
)
from src.autograder.logging import logger
import json
import os

if __name__ == "__main__":
    folder_path = input("Enter the path to the folder containing student submissions: ")
    gradebook_path = input("Enter the path to the folder containing the Gradebook: ")
    submissions_dict = process_submissions(folder_path)
    logger.info("Student submissions processed.")
    logger.info(f"Submissions dictionary: {submissions_dict}")

    if submissions_dict:
        # Add percentage grades and comments
        updated_gradebook_path, comments_list = add_grades_and_comments(
            submissions_dict, gradebook_path
        )
    print(comments_list)
