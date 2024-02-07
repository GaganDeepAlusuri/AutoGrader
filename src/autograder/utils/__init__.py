import os
import nbformat
import pandas as pd
from src.autograder.logging import logger
import re
import random


def read_file_content(file_path):
    """Reads and returns the content of a file."""
    with open(file_path, "r", encoding="utf-8") as file:
        return file.read()


# F1
def process_submissions(folder_path):
    """
    Processes student submissions in a given folder.

    Args:
    folder_path (str): Path to the folder containing student submissions.

    Returns:
    dict: Dictionary containing submission data, including student ID, name,
          path to the original file, and processed Python code.
    """
    # Strip double quotes from the folder_path if present
    folder_path = folder_path.strip('"')

    # Initialize dictionary to store submission data
    submissions_dict = {"SID": [], "S_NAME": [], "RAW_FILE": [], "PROCESSED_FILE": []}

    pattern = r"U(\d+)_(\w+)\."

    for filename in os.listdir(folder_path):
        if filename.endswith(".py") or filename.endswith(".ipynb"):
            match = re.match(pattern, filename)
            if match:
                student_id = "U" + match.group(1)
                student_name = match.group(2).lstrip(
                    "_"
                )  # Assuming names start with '_'

                original_file_path = os.path.join(folder_path, filename)
                raw_content = original_file_path  # Store the path to the original file

                processed_content = ""
                if filename.endswith(".py"):
                    processed_content = read_file_content(original_file_path)
                elif filename.endswith(".ipynb"):
                    try:
                        notebook_content = nbformat.reads(
                            read_file_content(original_file_path), as_version=4
                        )
                        python_code = ""
                        for cell in notebook_content.cells:
                            if cell.cell_type == "code":
                                python_code += cell.source + "\n\n"
                            elif cell.cell_type == "markdown":
                                markdown_lines = cell.source.split("\n")
                                for line in markdown_lines:
                                    if line.strip():
                                        python_code += f"# {line}\n"
                        processed_content = python_code
                    except Exception as e:
                        logger.error(f"Error processing {original_file_path}: {e}")

                submissions_dict["SID"].append(student_id)
                submissions_dict["S_NAME"].append(student_name)
                submissions_dict["RAW_FILE"].append(raw_content)
                submissions_dict["PROCESSED_FILE"].append(processed_content)

                logger.info(f"Processed submission for Student ID: {student_id}")
            else:
                logger.warning(f"Filename does not match pattern: {filename}")

    return submissions_dict


# F2
def add_grades_and_comments(submissions_dict):
    """
    Adds percentage grades and comments to the submission data.

    Args:
    submissions_dict (dict): Dictionary containing submission data.

    Returns:
    dict: Dictionary with added 'PERCENTAGE_GRADE' and 'COMMENTS' columns.
    """
    # Initialize the modified dictionary with additional columns
    modified_submissions_dict = submissions_dict.copy()
    modified_submissions_dict["PERCENTAGE_GRADE"] = []
    modified_submissions_dict["COMMENTS"] = []

    for i in range(len(submissions_dict["SID"])):
        percentage_grade = random.uniform(
            60, 100
        )  # Generate a random percentage grade between 60 and 100
        comments = f"Random comment for Student ID: {submissions_dict['SID'][i]}"  # Generate a unique comment

        modified_submissions_dict["PERCENTAGE_GRADE"].append(percentage_grade)
        modified_submissions_dict["COMMENTS"].append(comments)

        # Log the information for each student
        logger.info(
            f"Student ID: {submissions_dict['SID'][i]}, Percentage Grade: {percentage_grade}, Comments: {comments}"
        )

    return modified_submissions_dict


# F3
def prepare_for_canvas_upload(submissions_data, section, assignment_name):
    """
    Prepares a CSV file for uploading grades to Canvas Gradebook.

    Args:
    submissions_data (dict): Data structure from F2 containing student info and grades.
    section (str): The section for the assignment.
    assignment_name (str): The name of the assignment.

    Returns:
    str: Path to the created CSV file.
    """
    logger.info("Starting to prepare data for Canvas upload.")

    try:
        # Convert dictionary to DataFrame
        df = pd.DataFrame(submissions_data)

        # Map the necessary columns to Canvas requirements
        canvas_df = pd.DataFrame(
            {
                "Student": df["S_NAME"],
                "ID": "",
                "SIS User ID": df["SID"],
                "SIS Login ID": "",  # To be filled if available
                "Section": section,
                "Assignment": df["PERCENTAGE_GRADE"],
            }
        )

        # Specify the assignment name dynamically in the filename
        csv_file_path = f"grades_{assignment_name.replace(' ', '_')}.csv"
        canvas_df.to_csv(csv_file_path, index=False)

        logger.info(
            f"CSV file successfully created at {csv_file_path} for Canvas upload."
        )

    except Exception as e:
        logger.error(f"Failed to prepare data for Canvas upload: {e}")
        raise e  # Rethrow exception after logging

    return csv_file_path


# F4
def generate_student_reports(submissions_data, directory_path, assignment_name):
    """
    Generates individual Markdown reports for each student.

    Args:
    submissions_data (dict): The output data structure from F2 containing student grades and comments.
    directory_path (str): Path to the directory where reports will be saved.
    assignment_id (str): Identifier for the assignment.
    """
    # Ensure the directory exists
    os.makedirs(directory_path, exist_ok=True)

    for i, sid in enumerate(submissions_data["SID"]):
        # Construct the report filename
        filename = f"report_{assignment_name}_{sid}.md"
        filepath = os.path.join(directory_path, filename)

        # Gather data for the current student
        student_name = submissions_data["S_NAME"][i]
        percentage_grade = submissions_data["PERCENTAGE_GRADE"][i]
        comments = submissions_data["COMMENTS"][i]

        # Format the report content
        report_content = f"""# Assignment: {assignment_name}
## Student ID: {sid}
## Student Name: {student_name}
### Percentage Mark: {percentage_grade:.2f}%
### Comments:
{comments}
"""

        # Write the report to a Markdown file
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(report_content)

        logger.info(f"Report generated for {student_name} ({sid}) at {filepath}")
