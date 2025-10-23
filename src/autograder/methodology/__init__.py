import os
import nbformat
import pandas as pd

from src.autograder.logging import logger
import re
# import random  # Unused import
# import numpy as np  # Unused import
import os
from dotenv import load_dotenv
import json

from pydantic import BaseModel, Field
from typing import List
from pydantic import ValidationError
from src.autograder.utils import (
    GradingComment,
    read_file_content,
    find_csv_filename,
    get_completion,
)

from typing import Union, List, Dict


# load_dotenv()
# prof = """You are a programming expert tasked with evaluating student submissions for a programming assignment. Your evaluation should strictly adhere to the provided grading rubric. Each submission needs to be graded based on the assignment's requirements, and feedback should be given in the form of points and detailed comments. For each deduction in points, specify the reason based on the rubric. The feedback should be structured as a JSON object with two keys: 'points' and 'comments'. The 'points' key should contain the numeric grade awarded to the submission out of the total points possible, and the 'comments' key should list reasons for each deduction, directly correlating to the rubric's criteria."""


def add_grades_and_comments(
    submissions_dict,
    directory_path,
    assignment_name,
    possible_points,
    question_file_path,
    rubric_file_path,
    temperature=0,
    selected_model="gpt-3.5-turbo",
    reasoning_level=None,
):
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
    read_only_col_index = (
        headers.iloc[2]
        .tolist()
        .index(
            next(col for col in headers.iloc[2] if "(read only)" in str(col).lower())
        )
    )

    headers.insert(read_only_col_index, assignment_name, ["", "", possible_points])
    data.insert(read_only_col_index, assignment_name, ["" for _ in range(len(data))])

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
                temperature,
                selected_model,
                reasoning_level,
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

    updated_gradebook_path = csv_file_path.replace(".csv", "_updatedM1.csv")
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


def get_points_and_comments_using_GPT4(
    sid,
    student_name,
    processed_file,
    assignment_name,
    possible_points,
    question,
    rubric,
    temperature,
    selected_model,
    reasoning_level=None,
):
    possible_points = float(possible_points)
    prompt = f""" question:###{question}###,\
              Rubric: \"{rubric}\",
              Total points: {possible_points}, \
              ``SUBMISSION TO QUESTION ABOVE.``: ###{processed_file}###,
              ``Verify submission against rubric included in knowledge file prior to grading. 
              
              IMPORTANT: Respond with ONLY a valid JSON object in this exact format:
              {{
                "points": <number>,
                "comments": "<string>"
              }}
              
              Do not include any other text, markdown formatting, or special characters.``
              """

    try:
        response_message = get_completion(prompt, temperature, selected_model, reasoning_level)
        cleaned_response_message = response_message.strip("`").replace("json\n", "")
        
        # Better JSON extraction with error handling
        json_match = re.search(r"\{.*\}", cleaned_response_message, re.DOTALL)
        if json_match:
            json_str = json_match.group(0)
            logger.info("Extracted JSON string: %s", json_str)
            
            # Clean the JSON string to remove control characters
            import unicodedata
            json_str = ''.join(char for char in json_str if unicodedata.category(char)[0] != 'C' or char in '\n\r\t')
            
            # Parse and validate
            grading_info = GradingComment.model_validate_json(json_str)
            return grading_info.points, grading_info.comments
        else:
            logger.error("No JSON found in the response.")
            return 0, ["No valid JSON found in response"]

    except ValidationError as e:
        logger.error(f"Validation error: {e} for student id: {sid}")
        # Try to extract partial data if possible
        try:
            partial_data = json.loads(json_str)
            points = partial_data.get('points', 0)
            comments = partial_data.get('comments', 'Validation error - missing required fields')
            return points, comments
        except:
            return 0, ["Validation error. Check the data structure."]
    except json.JSONDecodeError as e:
        logger.error(f"JSON decoding error: {e} for student id: {sid}")
        return 0, ["Failed to decode JSON. Check the response_message format."]
    except Exception as e:
        logger.error(f"Unexpected error: {e} for student id: {sid}")
        return 0, ["An unexpected error occurred."]

    return points, comments
