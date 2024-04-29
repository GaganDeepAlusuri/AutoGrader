import pandas as pd
from pydantic import BaseModel, Field, ValidationError
import json
import os
from dotenv import load_dotenv
from src.autograder.logging import logger
from dotenv import load_dotenv
from openai import OpenAI
import openai
import json


from typing import Union, List, Dict, Any
from pydantic import ValidationError

from src.autograder.utils import (
    GradingComment,
    read_file_content,
    find_csv_filename,
    get_completionCOTMultiCalls,
)
import re

from typing import Dict, Any
from pydantic import BaseModel


class Criterion(BaseModel):
    description: str
    deduction: float


class DeductionPlan(BaseModel):
    criteria: Dict[str, Criterion]

    @classmethod
    def parse_deduction_plan(cls, json_str: str) -> "DeductionPlan":
        data = json.loads(json_str)
        return cls(criteria=data)


class EvaluationCriterion(BaseModel):
    deduction: float
    comments: str


class SubmissionEvaluation(BaseModel):
    evaluation: Dict[str, EvaluationCriterion]
    total_deduction: float
    final_comments: str


class FinalGrade(BaseModel):
    points: float
    comments: str


load_dotenv()


def generate_deduction_plan(
    question: str, rubric: str, possible_points: int
) -> DeductionPlan:
    prompt_template = f"""
    Assignment question: {question}
    Total marks: {possible_points}
    Rubric: {rubric}
    Understand the question and rubric, then create a deduction plan as described in the rubric.\
    Provide only a JSON output with each criteria as the key with description and deduction inside. 
    Here is an example of expected response:\
   {{
     "Train/Test Split": {{
    "description": "They should use the train test split function",
    "deduction": 0.5
  }},
  "Post-split Data Preprocessing": {{
    "description": "It's very important that they do not fit the test data",
    "deduction": 2.5
  }},
  "Measure Results": {{
    "description": "The students should indicate the RMSE",
    "deduction": 1.0
  }}
   }}
    """
    deduction_plan_response = get_completionCOTMultiCalls(prompt_template)
    json_match = re.search(r"\{.*\}", deduction_plan_response, re.DOTALL)
    if json_match:
        json_str = json_match.group(0)
        logger.info("Extracted JSON string: %s", json_str)
        return DeductionPlan.parse_deduction_plan(json_str)
    else:
        logger.error("No JSON found in the response.")
        return None


def evaluate_submission(
    deduction_plan: DeductionPlan, submission: str
) -> SubmissionEvaluation:
    """
    Evaluates the submission against the rubric and deduction plan.
    """
    prompt_template = f"""
    Deduction plan: {deduction_plan.json()}
    Student's submission: {submission}
    Grade the submission based on the rubric and deduction plan. Provide a JSON output with specific examples and deductions.\
    Here is an example of expected response:\
    {{
  "evaluation": {{
    "Title and Introduction": {{
      "deduction": 0.0,
      "comments": "Good job on including both a title and a brief introduction."
    }},
    "Load Data": {{
      "deduction": 0.5,
      "comments": "You imported libraries that were not used in your submission."
    }}
    // Additional criteria...
  }},
  "total_deduction": 1.0,
  "final_comments": "Overall, well done, but please ensure to only import what you use."
}}
    """
    evaluation_response = get_completionCOTMultiCalls(prompt_template)

    json_match = re.search(r"\{.*\}", evaluation_response, re.DOTALL)
    if json_match:
        json_str = json_match.group(0)
        logger.info("Extracted JSON string: %s", json_str)
        return SubmissionEvaluation.parse_raw(json_str)
    else:
        logger.error("No JSON found in the response.")
        return None


def finalize_grade(evaluation: dict, possible_points: int) -> dict:
    """
    Calculates the final grade and provides detailed feedback.
    """
    prompt_template = f"""
    Evaluation: {evaluation.json()}
    Total possible points: {possible_points}
    Calculate the total grade and provide detailed feedback for each deduction. 
    Here are a few examples of the expected output:\
    {{
  "points": 8.5,
  "comments": "Your submission mostly addresses the key aspects of the assignment question. You loaded the data, explored it, conducted pre-processing, and split the data into training and testing sets. However, there are a few areas where the submission could be improved. First, you scaled the entire dataset before splitting it into training and testing sets, which is not a good practice as it can lead to data leakage. The correct approach is to fit the scaler on the training data and then transform both the training and testing data. Second, you did not provide a rationale for their choice of k in the KNN model. Lastly, your discussion of the results could be more detailed, particularly in comparing the performance of the two models. Therefore, 1.5 points were deducted - 1 point for scaling the entire dataset before splitting and 0.5 points for not providing a rationale for the choice of k in the KNN model."
}}

Example 2 JSON Output:
{{
  "points": 7.5,
  "comments": "Your partially meets the requirements of the assignment. You successfully loaded the data, conducted a train/test split, and fitted both a Linear Regression and KNN model. However, there were several areas where the submission fell short. First, you did not provide a title or introduction for their analysis, resulting in a deduction of 0.5 points. Second, you did not discuss their selection of k value for the KNN model, resulting in a deduction of 1.0 points. Lastly, you did not recap their analysis or discuss the performance of the models using the RMSE metric, resulting in a deduction of 1.0 points. To improve, you should ensure they provide a clear introduction and conclusion for their analysis, and thoroughly discuss your model selection and performance evaluation process."
}}
Provide a JSON output with keys 'points' and 'comments' with a string value.
    """
    final_grade_response = get_completionCOTMultiCalls(prompt_template)
    cleaned_final_grade_response = final_grade_response.strip("`").replace("json\n", "")
    json_match = re.search(r"^\{.*\}$", cleaned_final_grade_response, re.DOTALL)
    if json_match:
        json_str = json_match.group(0).strip()
        logger.info("Extracted JSON string: %s", json_str)
        return FinalGrade.parse_raw(json_str)
    else:
        logger.error("No JSON found in the response.")
        return None


def add_grades_and_comments_COTMultiCalls(
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
    # Generate deduction plan
    deduction_plan = generate_deduction_plan(question, rubric, possible_points)
    logger.info(f"Deduction plan: {deduction_plan.json()}")

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
                deduction_plan,
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

    updated_gradebook_path = csv_file_path.replace(".csv", "_updatedM5.csv")
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
    sid: str,
    student_name: str,
    processed_file: str,
    assignment_name: str,
    possible_points: int,
    question: str,
    rubric: str,
    deduction_plan: dict,
):
    """
    Performs the multi-step grading process using separate LLM calls.
    """

    # Evaluate the submission based on the deduction plan
    evaluation = evaluate_submission(deduction_plan, processed_file)
    logger.info(f"Evaluation: {evaluation.json()}")

    # Calculate the final grade and provide feedback
    final_grade_and_comments = finalize_grade(evaluation, possible_points)
    logger.info(f"Final grade and comments: {final_grade_and_comments.json()}")

    points = final_grade_and_comments.points
    comments = final_grade_and_comments.comments
    logger.info(f"Final grade: {points} and comments: {comments}")

    return points, comments
