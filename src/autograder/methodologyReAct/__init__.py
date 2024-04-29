import os
import nbformat
import pandas as pd

from src.autograder.logging import logger
import re
import random
import numpy as np
import os
from dotenv import load_dotenv
import json
from pydantic import ValidationError
from src.autograder.utils import (
    GradingComment,
    read_file_content,
    find_csv_filename,
    get_completion,
    get_completionReAct,
)
from pydantic import BaseModel, Field


class GradingCommentReAct(BaseModel):
    points: float = Field(..., alias="points")
    comments: str = Field(..., alias="comments")


load_dotenv()
prof = """You are a programming expert tasked with evaluating student submissions for a programming assignment. Your evaluation should strictly adhere to the provided grading rubric."""


def add_grades_and_comments_ReAct(
    submissions_dict,
    directory_path,
    assignment_name,
    possible_points,
    question_file_path,
    rubric_file_path,
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
    prompt_for_guidelines = f"""Given the assignment '{assignment_name}' where students are asked the question: '{question}', and\
    the  total possible points are: {possible_points}.\
        Here is the rubric: ```{rubric}```\
        How would you go about grading this assignment for a final grade and feedback for the student's submission? What steps would you take to achieve this?"""
    try:
        grading_guidelines = get_completionReAct(prof, prompt_for_guidelines)
        logger.info(f"Grading guidelines extracted:{grading_guidelines}")
    except Exception as e:
        logger.error(f"Failed to extract the grading guidelines: {e}")
        return None

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
                grading_guidelines,
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

    updated_gradebook_path = csv_file_path.replace(".csv", "_updatedM3.csv")
    try:
        updated_gradebook_df.to_csv(updated_gradebook_path, index=False, header=False)
        logger.info(
            f"Updated gradebook with '{assignment_name}' saved to {updated_gradebook_path}."
        )
    except Exception as e:
        logger.error(f"Failed to save the updated gradebook: {e}")
        return None

    return updated_gradebook_path, comments_list


def get_points_and_comments_using_GPT4(
    sid,
    student_name,
    processed_file,
    assignment_name,
    possible_points,
    question,
    grading_guidelines,
):
    possible_points = float(possible_points)

    grading_prompt = f"""Follow the listed steps which are enclosed in parenthesis to grade the student submission which is enclosed in triple back ticks.\n
                        \nCompare each listed step to the corresponding part from the submission and evaluate it accordingly. Stick strictly to the steps for evaluation.\n
                        steps: {grading_guidelines}\
                        \nStudent submission: ```{processed_file}```\
                        \nProvide only a JSON output with 'points' and 'comments' as the keys in the response. 'comments' should be a string.""".strip()

    try:
        logger.info("Grading prompt is %s" % grading_prompt)
        response_message = get_completion(grading_prompt)
        json_match = re.search(r"\{.*\}", response_message, re.DOTALL)
        if json_match:
            json_str = json_match.group(0)
        else:
            logger.error(f"Response is: {json_str}")
        logger.info(f"Response after regex: {json_str}")

        # Assuming the response is well-structured JSON, parse it directly
        grading_info = GradingCommentReAct.parse_raw(json_str)
        return grading_info.points, grading_info.comments
    except Exception as e:
        logger.error(f"Error processing grading for student ID {sid}: {e}")
        return 0, "An error occurred during grading."
