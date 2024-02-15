import os
import nbformat
import pandas as pd
from src.autograder.logging import logger
import re
import random
import numpy as np


# Helper functions
def read_file_content(file_path):
    """Reads and returns the content of a file."""
    with open(file_path, "r", encoding="utf-8") as file:
        return file.read()


def find_csv_filename(directory):
    directory = directory.strip('"')
    for file in os.listdir(directory):
        if file.endswith(".csv"):
            return os.path.join(directory, file)
    logger.warning("No CSV file found in the directory.")
    return None


# F1
def process_submissions(folder_path):
    """
    Processes student submissions in a given folder based on the new file naming convention.

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

    # Pattern to match the file naming convention: studentname_ID_SomeOtherID_NameofTheFile
    pattern = r"([^_]+)_(\d+)_\w+_(.+)"

    for filename in os.listdir(folder_path):
        if filename.endswith(".py") or filename.endswith(".ipynb"):
            match = re.match(pattern, filename)
            if match:
                student_name = match.group(1)  # Extract student name
                student_id = match.group(2)  # Extract student ID as SID

                original_file_path = os.path.join(folder_path, filename)
                raw_content = original_file_path  # Store the path to the original file

                processed_content = ""
                if filename.endswith(".py"):
                    processed_content = read_file_content(original_file_path)
                elif filename.endswith(".ipynb"):
                    try:
                        notebook_content = nbformat.read(
                            open(original_file_path, "r", encoding="utf-8"),
                            as_version=4,
                        )
                        python_code = ""
                        for cell in notebook_content["cells"]:
                            if cell["cell_type"] == "code":
                                python_code += cell["source"] + "\n\n"
                            elif cell["cell_type"] == "markdown":
                                # Comment out Markdown content
                                markdown_lines = cell["source"].split("\n")
                                for line in markdown_lines:
                                    if line.strip():
                                        python_code += f"# {line}\n"
                        processed_content = python_code
                    except Exception as e:
                        logger.error(f"Error processing {original_file_path}: {e}")

                submissions_dict["SID"].append(student_id)
                submissions_dict["S_NAME"].append(
                    student_name.replace("_", " ")
                )  # Replace underscores with spaces in names
                submissions_dict["RAW_FILE"].append(raw_content)
                submissions_dict["PROCESSED_FILE"].append(processed_content)

                logger.info(f"Processed submission for Student ID: {student_id}")
            else:
                logger.warning(f"Filename does not match pattern: {filename}")

    return submissions_dict


# F2
def add_grades_and_comments(submissions_dict, directory_path):
    csv_file_path = find_csv_filename(directory_path)
    if not csv_file_path:
        logger.error("CSV file not found in the specified directory.")
        return None

    try:
        # Read the CSV with no header to manipulate rows separately
        full_data = pd.read_csv(csv_file_path, header=None)
        headers = full_data.iloc[:3]  # First three rows as headers
        data = full_data.iloc[3:]  # Rest of the data
        logger.info(f"Gradebook loaded successfully from {csv_file_path}.")
    except Exception as e:
        logger.error(f"Failed to load the gradebook CSV: {e}")
        return None

    # Check the correct name for the 'ID' column in the first row of headers
    id_column_index = (
        headers.iloc[0]
        .tolist()
        .index(next(col for col in headers.iloc[0] if "ID" in col))
    )

    assignment_name = input("Enter the Assignment Name: ")
    possible_points = int(input("Enter the total points possible: "))

    # Find the index for the first "(read only)" column in the third header row
    read_only_col_index = (
        headers.iloc[2]
        .tolist()
        .index(
            next(col for col in headers.iloc[2] if "(read only)" in str(col).lower())
        )
    )

    # Insert the new assignment and possible points at the correct position
    headers.insert(read_only_col_index, assignment_name, ["", "", possible_points])
    data.insert(
        read_only_col_index,
        assignment_name,
        ["" for _ in range(len(data))],  # Initially assign blank marks for all
    )

    # Assign grades based on the 'ID' column, ensuring all students are included
    for index, row in data.iterrows():
        sid = row[id_column_index]
        # Check if the SID exists in submissions_dict
        if str(sid) in submissions_dict["SID"]:
            # If the student has a submission, assign a random mark
            data.at[index, assignment_name] = random.randint(1, possible_points)

    # Combine the header and data for saving
    updated_gradebook_df = pd.concat([headers, data], ignore_index=True)
    # Set the assignment name as the column name for the newly inserted assignment column
    updated_gradebook_df.iloc[0, read_only_col_index] = assignment_name

    # Save the updated gradebook
    updated_gradebook_path = csv_file_path.replace(".csv", "_updated.csv")
    try:
        updated_gradebook_df.to_csv(updated_gradebook_path, index=False, header=False)
        logger.info(
            f"Updated gradebook with '{assignment_name}' saved to {updated_gradebook_path}."
        )
    except Exception as e:
        logger.error(f"Failed to save the updated gradebook: {e}")
        return None

    # Create list of dictionaries for comments
    comments_list = []
    for sid, sname in zip(submissions_dict["SID"], submissions_dict["S_NAME"]):
        comments_list.append(
            {
                "Assignment": assignment_name,
                "Student ID": sid,
                "Student Name": sname,
                "Points": random.randint(1, possible_points),
                "Comments": "",
            }
        )

    return updated_gradebook_path, comments_list


# F3
'''def prepare_for_canvas_upload(submissions_data, section, assignment_name):
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

    return csv_file_path'''


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
### Points: {percentage_grade:.2f}%
### Comments:
{comments}
"""

        # Write the report to a Markdown file
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(report_content)

        logger.info(f"Report generated for {student_name} ({sid}) at {filepath}")
