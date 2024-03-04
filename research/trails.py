import json
import os
import nbformat
import re
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


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
    print("No CSV file found in the directory.")
    return None


def export_as_json(data, filename):
    with open(filename, "w") as f:
        json.dump(data, f, indent=4)


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
                        print(f"Error processing {original_file_path}: {e}")

                submissions_dict["SID"].append(student_id)
                submissions_dict["S_NAME"].append(
                    student_name.replace("_", " ")
                )  # Replace underscores with spaces in names
                submissions_dict["RAW_FILE"].append(raw_content)
                submissions_dict["PROCESSED_FILE"].append(processed_content)

                print(f"Processed submission for Student ID: {student_id}")
            else:
                print(f"Filename does not match pattern: {filename}")

    return submissions_dict


def evaluate_submissions_with_deepseek_model(
    submissions_dict, rubric, assignment_questions
):
    """
    Evaluates student submissions using the DeepSeek model.

    Args:
    - submissions_dict (dict): Dictionary containing submission data.
    - rubric (str): The grading rubric as a string.
    - assignment_questions (str): Assignment questions.

    Returns:
    - dict: A dictionary with evaluations and grades for each submission.
    """
    # Ensure CUDA is available for GPU acceleration
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load the tokenizer and model with trust_remote_code to support custom architecture
    tokenizer = AutoTokenizer.from_pretrained(
        "deepseek-ai/deepseek-coder-33b-instruct", trust_remote_code=True
    )

    model = AutoModelForCausalLM.from_pretrained(
        "deepseek-ai/deepseek-coder-33b-instruct", trust_remote_code=True
    ).cuda()

    evaluations = []

    for submission in submissions_dict["PROCESSED_FILE"]:
        # Construct the prompt for the DeepSeek model
        prompt = f"You are an expert in evaluating programming assignments. You are given an assignment question, \
a rubric to grade it on. Provide a grade out of 10 for the student's submission adhering to the rubric for any deductions. Give comments for each deduction to present as feedback to the student. \
### Assignment Question\n{assignment_questions}\n### Student Submission\n{submission}\n### Grading Rubric\n{rubric}\n### Evaluation\n"

        messages = [{"role": "user", "content": prompt}]
        # Tokenize the prompt and move to the same device as the model
        inputs = tokenizer.apply_chat_template(
            messages, return_tensors="pt", max_length=1024
        ).to(model.device)

        # Generate evaluation using DeepSeek
        outputs = model.generate(
            inputs,
            max_new_tokens=512,
            do_sample=False,
            top_k=50,
            top_p=0.95,
            num_return_sequences=1,
            eos_token_id=32021,
        )

        # Decode and format the evaluation result
        evaluation_result = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Append the formatted evaluation result
        evaluations.append(evaluation_result)

    # Return the evaluations
    return evaluations


if __name__ == "__main__":
    folder_path = r"D:\canvas files\Assignment 4\submissions (6)"
    gradebook_path = r"D:\canvas files\Assignment 4\gradebook"
    submissions_dict = process_submissions(folder_path)
    print("Student submissions processed.")
    print(f"Submissions dictionary: {submissions_dict}")
    rubric = f"""# Students should have a Title name and brief intro - deduct 0.5 if all missing,

0.25 up to 0.5 for individual items missind

Load Data

# Students should only import what the use - deduct 0.5 if any missing - deduct 0.5 if any imported and not used.

Explore data

# load data - deduct 1.0 if not loaded (if not loaded, this indicates much bigger problems though - and more marks will be deducted for those)

# They should at least check columns (either df.columns, or df.head) to missing values and plot the data - deduct 0.5 if not done

# conduct any pre-split data preprocessing
# They should note somewhere that there are no missing values - or other issues that require preprocessing - deduct 0.5 if not done


# conduct train/test split

# they should use the train test split function - deduct 0.5 if not used
# this is a relatively small dataset, and if students understand this and and train/test splitting, they should favor a smaller test set size - like 80/20 - do not deduct if they do not have this, but add a comment
# also, the student should indicate the random_state so that this is repeatable - do not deduct if missing, but add a comment

# conduct any post-split data preprocessing

scaler = StandardScaler()

 
# they could do this in two lines, with a fit, then transform

# it's very important that they do not fit the test data - deduct 2.5 if they do


# measure results - since this is regression, it's easy -- RMSE
# The students should indicate the RMSE - deduct 1.0 if not done
# It's OK if the student includes R^2, and lists the coefficients, but this is not required - do not deduct if  done


# for KNN, they should discuss their selection of k value. The rule of thumb is to use root of N
# they can argue for another value, or set of values (and do multiple tests)
# if they haven't demonstrated any indication for their selection of k, deduct 1.0



# Discussion results

# They should recap their analysis (they fit and tested two models) - deduct 0.5 if not done
# they should discuss the performance of these models using the RMSE metric to indicate which model fits better - deduct 1.0 if not done"""
q = f"""Fit a linear regression and KNN regressor to the data given to you. Your notebook must include any necessary preprocessing and data exploration. Include a markdown to discuss the rationale for preprocessing or data exploration steps. Add a section at the end of the notebook that discusses the results of your analysis and discuss the models and their performance on the data. 
    """
list = evaluate_submissions_with_deepseek_model(submissions_dict, rubric, q)
print(list)
