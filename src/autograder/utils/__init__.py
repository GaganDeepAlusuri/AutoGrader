import os
from src.autograder.logging import logger
from pydantic import BaseModel
import nbformat
import re


class GradingComment(BaseModel):
    points: float
    comments: str


def read_file_content(file_path):
    """Reads and returns the content of a file."""
    with open(file_path, "r", encoding="utf-8") as file:
        return file.read()


def find_csv_filename(directory):
    """Finds a CSV file in the specified directory."""
    directory = directory.strip('"')
    for file in os.listdir(directory):
        if file.endswith(".csv"):
            return os.path.join(directory, file)
    logger.warning("No CSV file found in the directory.")
    return None


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


from tenacity import (
    retry,
    wait_random_exponential,
    stop_after_attempt,
    retry_if_exception_type,
)
import openai
from openai import OpenAI

client = OpenAI()
model = "gpt-4"


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


def get_completion2(prompt2):
    messages = [
        {
            "role": "system",
            "content": "You are a keyword extractor for Grading assignment. Given a question, extract all relevant keywords in it that are important to the evaluation of the question. Only output keywords seperated by spaces nothing else.",
        },
        {"role": "user", "content": prompt2},
    ]
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages,
        temperature=0,
    )
    response_message = response.choices[0].message.content
    return response_message


def get_completion3(sys_content, prompt3):
    messages = [
        {
            "role": "system",
            "content": sys_content,
        },
        {"role": "user", "content": prompt3},
    ]
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages,
        temperature=0,
    )
    response_message = response.choices[0].message.content
    return response_message
