import os
import nbformat
import pandas as pd

from src.autograder.logging import logger
import re
import random
import numpy as np
import os
import openai
from dotenv import load_dotenv
from openai import OpenAI
import json
from tenacity import (
    retry,
    wait_random_exponential,
    stop_after_attempt,
    retry_if_exception_type,
)
from pydantic import BaseModel, Field
from typing import List
from pydantic import ValidationError


from typing import Union, List, Dict


class GradingComment(BaseModel):
    points: float
    comments: str


load_dotenv()
prof = """
You are a programming expert tasked with evaluating student submissions for a programming assignment. Your evaluation should strictly adhere to the provided grading rubric. Each submission needs to be graded based on the assignment's requirements, and feedback should be given in the form of points and detailed comments. For each deduction in points, specify the reason based on the rubric. The feedback should be structured as a JSON object with two keys: 'points' and 'comments'. The 'points' key should contain the numeric grade awarded to the submission out of the total points possible, and the 'comments' key should list reasons for each deduction, directly correlating to the rubric's criteria.
"""

model = "gpt-4"
client = OpenAI()


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
    # def generate_structured_rubric_with_chatgpt(rubric_text, total_points):

    prompt = f"""
    The total points for the assignment is {total_points}. Given the rubric text below, your task is to transform it into a structured dictionary format. This dictionary should clearly outline each grading criterion from the rubric, including a detailed description, the specific actions or requirements necessary to meet the criterion, and the points to be deducted for non-compliance. Do not assume any other criterion or points to be deducted other than that is not mentioned in the Rubric Text which is enclosed in triple backticks below. Follow the provided template for each entry in the dictionary:

    Template:
    {{ Total Points: {total_points}
        'criteria_name': {{
            'description': 'Detailed description of the criterion',
            'points': 'Points deducted for not meeting this criterion',
            'action_required': 'Specific action or requirement needed to satisfy the criterion'
        }},
    }}

    Make sure each criterion from the rubric is represented as a separate entry within the dictionary. This structured format will enable a grading system to assess student submissions against each criterion, automatically calculate total deductions, and generate targeted, actionable feedback.

    Please structure the following rubric text into the desired dictionary format:

    Rubric Text:
    ```{rubric_text}```
    """

    # Call the function that interfaces with ChatGPT, passing the constructed prompt
    structured_rubric = get_completion(prompt)

    # Process the response from ChatGPT to obtain the structured rubric
    # Assuming the response is directly usable or requires minimal processing

    return structured_rubric


def add_grades_and_comments(submissions_dict, directory_path):
    csv_file_path = find_csv_filename(directory_path)
    if not csv_file_path:
        logger.error("CSV file not found in the specified directory.")
        return None

    try:
        full_data = pd.read_csv(csv_file_path, header=None)
        headers = full_data.iloc[:3]  # First three rows as headers
        data = full_data.iloc[3:]  # Rest of the data
        logger.info(f"Gradebook loaded successfully from {csv_file_path}.")
    except Exception as e:
        logger.error(f"Failed to load the gradebook CSV: {e}")
        return None

    id_column_index = (
        headers.iloc[0]
        .tolist()
        .index(next(col for col in headers.iloc[0] if "ID" in col))
    )
    assignment_name = input("Enter the Assignment Name: ")
    possible_points = int(input("Enter the total points possible: "))

    read_only_col_index = (
        headers.iloc[2]
        .tolist()
        .index(
            next(col for col in headers.iloc[2] if "(read only)" in str(col).lower())
        )
    )

    headers.insert(read_only_col_index, assignment_name, ["", "", possible_points])
    data.insert(read_only_col_index, assignment_name, ["" for _ in range(len(data))])

    question_file_path = input(
        "Enter the path to the text file containing the question: "
    ).strip('"')
    rubric_file_path = input(
        "Enter the path to the text file containing the rubric: "
    ).strip('"')

    question = read_file_content(question_file_path)
    rubric = read_file_content(rubric_file_path)

    if question is None or rubric is None:
        logger.error("Failed to read question or rubric file.")
        return None
    # rubric = generate_structured_rubric_with_chatgpt(rubric, possible_points)
    # logger.info(f"rubric summary: %s" % rubric)

    comments_list = []
    for index, row in data.iterrows():
        sid = row[id_column_index]
        if str(sid) in submissions_dict["SID"]:
            student_index = submissions_dict["SID"].index(str(sid))
            points, comments = get_points_and_comments_using_GPT4(
                sid,
                submissions_dict["S_NAME"][student_index],
                submissions_dict["PROCESSED_FILE"][student_index],
                assignment_name,
                possible_points,
                question,
                rubric,
            )

            data.at[index, assignment_name] = points
            comments_list.append(
                {
                    "SID": sid,
                    "Name": submissions_dict["S_NAME"][student_index],
                    "Question": question,
                    "Processed File": submissions_dict["PROCESSED_FILE"][student_index],
                    "Points": points,
                    "Comments": comments,
                }
            )

    updated_gradebook_df = pd.concat([headers, data], ignore_index=True)
    updated_gradebook_df.iloc[0, read_only_col_index] = assignment_name

    updated_gradebook_path = csv_file_path.replace(".csv", "_updated.csv")
    try:
        updated_gradebook_df.to_csv(updated_gradebook_path, index=False, header=False)
        logger.info(
            f"Updated gradebook with '{assignment_name}' saved to {updated_gradebook_path}."
        )
    except Exception as e:
        logger.error(f"Failed to save the updated gradebook: {e}")
        return None

    return updated_gradebook_path, comments_list


from tenacity import retry, stop_after_attempt, wait_random_exponential


@retry(
    retry=retry_if_exception_type(
        (
            openai.APIError,
            openai.APIConnectionError,
            openai.RateLimitError,
            openai.Timeout,
        )
    ),
    wait=wait_random_exponential(multiplier=2, max=60),
    stop=stop_after_attempt(10),
)
def get_completion(prompt):
    messages = [
        {
            "role": "system",
            "content": "Activate Grading Mode.",
        },
        {"role": "user", "content": prompt},
    ]
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0,
    )
    response_message = response.choices[0].message.content
    return response_message


def get_points_and_comments_using_GPT4(
    sid,
    student_name,
    processed_file,
    assignment_name,
    possible_points,
    question,
    rubric,
):

    possible_points = float(possible_points)
    prompt = f""" question:###{question}###,\
                  Rubric: \"{rubric}\",
                  Total points: {possible_points}, \
                  ``SUBMISSION TO QUESTION ABOVE.``: ###{processed_file}###,
                  ``Verify submission against rubric included in knowledge file prior to grading. Respond with a json output with keys points (consisting of final grade after deductions) and comments (A string for any deducted points and reason.).``
                  """

    try:
        response_message = get_completion(prompt)
        logger.info(response_message)

        # Use Pydantic for parsing and validation
        grading_info = GradingComment.parse_raw(response_message)

        points = grading_info.points
        # Handle comments being either a list or a dictionary
        if isinstance(grading_info.comments, dict):
            # Convert dictionary comments to a list of strings if needed
            comments = [
                f"{key}: {value}" for key, value in grading_info.comments.items()
            ]
        else:
            # If it's already a list, use it directly
            comments = grading_info.comments

    except ValidationError as e:
        logger.error(f"Validation error: {e} for student id: {sid}")
        points, comments = 0, ["Validation error. Check the data structure."]
    except json.JSONDecodeError as e:
        logger.error(f"JSON decoding error: {e} for student id: {sid}")
        points, comments = 0, [
            "Failed to decode JSON. Check the response_message format."
        ]
    except Exception as e:
        logger.error(f"Unexpected error: {e} for student id: {sid}")
        points, comments = 0, ["An unexpected error occurred."]

    return points, comments
