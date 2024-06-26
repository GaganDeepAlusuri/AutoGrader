import os
from src.autograder.logging import logger
from pydantic import BaseModel
import nbformat
import re

model = "gpt-3.5-turbo"
# client = OpenAI()
from openai import OpenAI

global client
client = None


def set_global_client(api_key):
    global client
    client = OpenAI(api_key=api_key)


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


@retry(
    retry=retry_if_exception_type(
        (
            openai.APIError,
            openai.APIConnectionError,
            openai.RateLimitError,
            openai.Timeout,
        )
    ),
    wait=wait_random_exponential(multiplier=4, max=60),
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


@retry(
    retry=retry_if_exception_type(
        (
            openai.APIError,
            openai.APIConnectionError,
            openai.RateLimitError,
            openai.Timeout,
        )
    ),
    wait=wait_random_exponential(multiplier=4, max=60),
    stop=stop_after_attempt(10),
)
def get_completionCOT(sys_content, prompt):
    messages = [
        {
            "role": "system",
            "content": sys_content,
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


@retry(
    retry=retry_if_exception_type(
        (
            openai.APIError,
            openai.APIConnectionError,
            openai.RateLimitError,
            openai.Timeout,
        )
    ),
    wait=wait_random_exponential(multiplier=4, max=60),
    stop=stop_after_attempt(10),
)
def get_completionCOTMultiCalls(prompt):
    messages = [
        {
            "role": "system",
            "content": "You are an experienced Data Science Professor.",
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


@retry(
    retry=retry_if_exception_type(
        (
            openai.APIError,
            openai.APIConnectionError,
            openai.RateLimitError,
            openai.Timeout,
        )
    ),
    wait=wait_random_exponential(multiplier=4, max=60),
    stop=stop_after_attempt(10),
)
def get_completion_keywords(prompt2):
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


@retry(
    retry=retry_if_exception_type(
        (
            openai.APIError,
            openai.APIConnectionError,
            openai.RateLimitError,
            openai.Timeout,
        )
    ),
    wait=wait_random_exponential(multiplier=4, max=60),
    stop=stop_after_attempt(10),
)
def get_completionReAct(sys_content, prompt3):
    messages = [
        {
            "role": "system",
            "content": sys_content,
        },
        {"role": "user", "content": prompt3},
    ]
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0,
    )
    response_message = response.choices[0].message.content
    return response_message


'''def get_topComments(comments):
    # Define the content of the system message using an f-string variable
    system_content = f"""You are an experienced Data Science Professor. \
To systematically analyze a dataset of TA comments on student assignments and extract the top 5 most frequently mentioned mistakes. \
This will help in identifying areas where students commonly struggle, enabling targeted support and instructional improvement. Follow the below \
steps to do this:
step 1: Read through the comments enclosed in the triple back ticks below to get an overall sense of the feedback themes.
step 2: Note any recurring mistakes or areas of improvement mentioned by the TA. Do not make up any mistakes or areas on your own. Stick strictly to only the provived comments that are enclosed in triple back ticks below.
step 3: Create categories based on the types of mistakes identified (e.g., Model Selection, Data Preprocessing, Result Interpretation).
step 4: Assign each specific mistake mentioned in the comments to its corresponding category.
step 5: Tally the frequency of each mistake within its category.
step 6: Rank the mistakes from most to least common based on their frequency.
step 7: Identify the top 5 mistakes based on the ranking.
step 8: Put them in a python list to display.

Here are the comments:```{comments}```\
Make sure to only respond with the top 5 mistakes as a python list. No other output or introduction or heading is needed. \
Just output the top 5 mistakes as a python list.
"""

    messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": comments},
    ]

    response = client.chat.completions.create(
        model="gpt-4",
        messages=messages,
        temperature=0,
    )
    response_message = response.choices[0].message.content
    return response_message
'''
