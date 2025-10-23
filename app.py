import streamlit as st
from src.autograder.utils import process_submissions
import tempfile
import os
from src.autograder.logging import logger
from src.autograder.methodology import add_grades_and_comments
from src.autograder.utils import process_submissions, set_global_client
from src.autograder.methodologyCOTRAG import (
    add_grades_and_comments_COTRAG,
    generate_data_store,
)
from src.autograder.methodologyReAct import add_grades_and_comments_ReAct
from src.autograder.methodologyCOT import add_grades_and_comments_COT
from src.autograder.methodologyCOTMultiCalls import (
    add_grades_and_comments_COTMultiCalls,
)
import time
from datetime import datetime, timedelta
import pandas as pd
import ast

# plotting
import io
import seaborn as sns
import matplotlib.pyplot as plt

st.markdown(
    """
  <style>
    .block-container.st-emotion-cache-1y4p8pa.ea3mdgi2 {
      margin-right: 25rem !important;
    }
    
    /* Status bar styling */
    .status-container {
      background-color: #f0f2f6;
      padding: 1rem;
      border-radius: 0.5rem;
      border-left: 4px solid #1f77b4;
      margin: 1rem 0;
    }
    
    .status-text {
      font-weight: 600;
      color: #1f77b4;
      margin-bottom: 0.5rem;
    }
    
    .progress-details {
      color: #666;
      font-size: 0.9rem;
    }
  </style>
""",
    unsafe_allow_html=True,
)

from openai import OpenAI


def verify_api_key(api_key):
    client = OpenAI(api_key=api_key)
    try:
        # Attempt to create a completion or another minimal operation
        response = client.completions.create(
            model="gpt-3.5-turbo-instruct",  # Use an appropriate model identifier
            prompt="test",  # Minimal prompt
            max_tokens=1,
        )
        # If the call succeeds, the API key is valid
        set_global_client(api_key)
        return True
    except Exception as e:
        # Handle specific exceptions related to authentication failure
        print(f"Failed to verify API key: {e}")
        return False


def apply_methodology_with_progress(
    methodology,
    submissions,
    gradebook_path,
    assignment_name,
    possible_points,
    question_file_path,
    rubric_file_path,
    temperature,
    selected_model,
    reasoning_level=None,
    progress_bar=None,
    status_text=None
):
    """Apply methodology with progress tracking"""
    if methodology == "M1":
        logger.info(
            "###############################################################################___________Applying Simple GPT prompt_______________############################################################"
        )
        return add_grades_and_comments_with_progress(
            submissions,
            gradebook_path,
            assignment_name,
            possible_points,
            question_file_path,
            rubric_file_path,
            temperature,
            selected_model,
            reasoning_level,
            progress_bar,
            status_text,
        )
    elif methodology == "M2":
        logger.info(
            "###############################################################################___________Applying COT with RAG_______________############################################################"
        )
        return add_grades_and_comments_COTRAG_with_progress(
            submissions,
            gradebook_path,
            assignment_name,
            possible_points,
            question_file_path,
            rubric_file_path,
            temperature,
            selected_model,
            reasoning_level,
            progress_bar,
            status_text,
        )
    elif methodology == "M3":
        logger.info(
            "###############################################################################___________Applying ReAct_______________############################################################"
        )
        return add_grades_and_comments_ReAct_with_progress(
            submissions,
            gradebook_path,
            assignment_name,
            possible_points,
            question_file_path,
            rubric_file_path,
            temperature,
            selected_model,
            reasoning_level,
            progress_bar,
            status_text,
        )
    elif methodology == "M4":
        logger.info(
            "###############################################################################___________Applying COT_______________############################################################"
        )
        return add_grades_and_comments_COT_with_progress(
            submissions,
            gradebook_path,
            assignment_name,
            possible_points,
            question_file_path,
            rubric_file_path,
            temperature,
            selected_model,
            reasoning_level,
            progress_bar,
            status_text,
        )
    elif methodology == "M5":
        logger.info(
            "###############################################################################___________Applying COT with Multi-Calls_______________############################################################"
        )
        return add_grades_and_comments_COTMultiCalls_with_progress(
            submissions,
            gradebook_path,
            assignment_name,
            possible_points,
            question_file_path,
            rubric_file_path,
            temperature,
            selected_model,
            reasoning_level,
            progress_bar,
            status_text,
        )
    else:
        logger.error("Unknown methodology.")
        return None, None


def apply_methodology(
    methodology,
    submissions,
    gradebook_path,
    assignment_name,
    possible_points,
    question_file_path,
    rubric_file_path,
    temperature,
    selected_model,
    reasoning_level=None
):
    if methodology == "M1":
        logger.info(
            "###############################################################################___________Applying Simple GPT prompt_______________############################################################"
        )
        return add_grades_and_comments(
            submissions,
            gradebook_path,
            assignment_name,
            possible_points,
            question_file_path,
            rubric_file_path,
            temperature,
            selected_model,
            reasoning_level,
        )
    elif methodology == "M2":
        logger.info(
            "###############################################################################___________Applying COT with RAG_______________############################################################"
        )
        return add_grades_and_comments_COTRAG(
            submissions,
            gradebook_path,
            assignment_name,
            possible_points,
            question_file_path,
            rubric_file_path,
            temperature,
            selected_model,
            reasoning_level,
        )
    elif methodology == "M3":
        logger.info(
            "###############################################################################___________Applying ReAct_______________############################################################"
        )
        return add_grades_and_comments_ReAct(
            submissions,
            gradebook_path,
            assignment_name,
            possible_points,
            question_file_path,
            rubric_file_path,
            temperature,
            selected_model,
            reasoning_level,
        )
    elif methodology == "M4":
        logger.info(
            "###############################################################################___________Applying COT_______________############################################################"
        )
        return add_grades_and_comments_COT(
            submissions,
            gradebook_path,
            assignment_name,
            possible_points,
            question_file_path,
            rubric_file_path,
            temperature,
            selected_model,
            reasoning_level,
        )
    elif methodology == "M5":
        logger.info(
            "###############################################################################___________Applying COT with Multi-Calls_______________############################################################"
        )
        return add_grades_and_comments_COTMultiCalls(
            submissions,
            gradebook_path,
            assignment_name,
            possible_points,
            question_file_path,
            rubric_file_path,
            temperature,
            selected_model,
            reasoning_level,
        )
    else:
        logger.error("Unknown methodology.")
        return None, None


# Progress-enabled wrapper functions for each methodology
def add_grades_and_comments_with_progress(
    submissions,
    gradebook_path,
    assignment_name,
    possible_points,
    question_file_path,
    rubric_file_path,
    temperature,
    selected_model,
    reasoning_level=None,
    progress_bar=None,
    status_text=None,
):
    """Wrapper for M1 methodology with real-time progress tracking"""
    if progress_bar and status_text:
        status_text.text("🔄 Starting M1 (Simple GPT) grading...")
        progress_bar.progress(35)
    
    # Count total submissions for progress tracking
    total_submissions = len(submissions.get("SID", []))
    if total_submissions > 0:
        status_text.text(f"🔄 Grading {total_submissions} submissions using M1 (Simple GPT)...")
    
    # Use the progress-enabled version
    result = add_grades_and_comments_with_real_progress(
        submissions,
        gradebook_path,
        assignment_name,
        possible_points,
        question_file_path,
        rubric_file_path,
        temperature,
        selected_model,
        reasoning_level,
        progress_bar,
        status_text,
        total_submissions,
    )
    
    if progress_bar and status_text:
        progress_bar.progress(80)
        status_text.text("✅ M1 grading completed!")
    
    return result


def add_grades_and_comments_COTRAG_with_real_progress(
    submissions_dict,
    directory_path,
    assignment_name,
    possible_points,
    question_file_path,
    rubric_file_path,
    temperature=0,
    selected_model="gpt-3.5-turbo",
    reasoning_level=None,
    progress_bar=None,
    status_text=None,
    total_submissions=0,
):
    """M2 methodology with real-time progress tracking"""
    import os
    import pandas as pd
    from src.autograder.logging import logger
    from src.autograder.utils import (
        read_file_content,
        find_csv_filename,
        get_completion_keywords,
    )
    from src.autograder.methodologyCOTRAG import get_points_and_comments_using_GPT4

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
    keywords_from_question = get_completion_keywords(question, temperature, selected_model, reasoning_level)
    logger.info(f"Keywords from question: {keywords_from_question}")
    rubric = read_file_content(rubric_file_path)

    if question is None or rubric is None:
        logger.error("Failed to read question or rubric file.")
        return None

    comments_list = []
    graded_count = 0
    
    for index, row in data.iterrows():
        sid = row[id_column_index]
        if str(sid) in submissions_dict["SID"]:
            student_index = submissions_dict["SID"].index(str(sid))
            
            # Update progress for each submission
            if progress_bar and status_text and total_submissions > 0:
                graded_count += 1
                progress_percentage = 35 + (graded_count / total_submissions) * 45  # 35% to 80%
                progress_bar.progress(int(progress_percentage))
                status_text.text(f"🔄 Grading submission {graded_count}/{total_submissions} - Student {sid}")
            
            points, comments = get_points_and_comments_using_GPT4(
                sid,
                submissions_dict["S_NAME"][student_index],
                submissions_dict["PROCESSED_FILE"][student_index],
                assignment_name,
                possible_points,
                question,
                rubric,
                keywords_from_question,
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

    updated_gradebook_path = csv_file_path.replace(".csv", "_updatedM2.csv")
    try:
        updated_gradebook_df.to_csv(updated_gradebook_path, index=False, header=False)
        logger.info(
            f"Updated gradebook with '{assignment_name}' saved to {updated_gradebook_path}."
        )
    except Exception as e:
        logger.error(f"Failed to save the updated gradebook: {e}")
        return None

    return updated_gradebook_path, comments_list


def add_grades_and_comments_COTRAG_with_progress(
    submissions,
    gradebook_path,
    assignment_name,
    possible_points,
    question_file_path,
    rubric_file_path,
    temperature,
    selected_model,
    reasoning_level=None,
    progress_bar=None,
    status_text=None,
):
    """Wrapper for M2 methodology with real-time progress tracking"""
    if progress_bar and status_text:
        status_text.text("🔄 Starting M2 (COT with RAG) grading...")
        progress_bar.progress(35)
    
    # Count total submissions for progress tracking
    total_submissions = len(submissions.get("SID", []))
    if total_submissions > 0:
        status_text.text(f"🔄 Grading {total_submissions} submissions using M2 (COT with RAG)...")
    
    # Use the real-time progress version
    result = add_grades_and_comments_COTRAG_with_real_progress(
        submissions,
        gradebook_path,
        assignment_name,
        possible_points,
        question_file_path,
        rubric_file_path,
        temperature,
        selected_model,
        reasoning_level,
        progress_bar,
        status_text,
        total_submissions,
    )
    
    if progress_bar and status_text:
        progress_bar.progress(80)
        status_text.text("✅ M2 grading completed!")
    
    return result


def add_grades_and_comments_ReAct_with_real_progress(
    submissions_dict,
    directory_path,
    assignment_name,
    possible_points,
    question_file_path,
    rubric_file_path,
    temperature=0,
    selected_model="gpt-3.5-turbo",
    reasoning_level=None,
    progress_bar=None,
    status_text=None,
    total_submissions=0,
):
    """M3 methodology with real-time progress tracking"""
    import os
    import pandas as pd
    from src.autograder.logging import logger
    from src.autograder.utils import (
        read_file_content,
        find_csv_filename,
    )
    from src.autograder.methodologyReAct import get_completionReAct, get_points_and_comments_using_GPT4

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

    # Generate grading guidelines
    prof = """You are a programming expert tasked with evaluating student submissions for a programming assignment. Your evaluation should strictly adhere to the provided grading rubric. Each submission needs to be graded based on the assignment's requirements, and feedback should be given in the form of points and detailed comments. For each deduction in points, specify the reason based on the rubric. The feedback should be structured as a JSON object with two keys: 'points' and 'comments'. The 'points' key should contain the numeric grade awarded to the submission out of the total points possible, and the 'comments' key should list reasons for each deduction, directly correlating to the rubric's criteria."""
    
    prompt_for_guidelines = f"""Given the assignment '{assignment_name}' where students are asked the question: '{question}', and\
    the  total possible points are: {possible_points}.\
        Here is the rubric: ```{rubric}```\
        How would you go about grading this assignment for a final grade and feedback for the student's submission? What steps would you take to achieve this?"""
    
    try:
        grading_guidelines = get_completionReAct(prof, prompt_for_guidelines, temperature, selected_model, reasoning_level)
        logger.info(f"Grading guidelines extracted:{grading_guidelines}")
    except Exception as e:
        logger.error(f"Failed to extract the grading guidelines: {e}")
        return None

    comments_list = []
    graded_count = 0
    
    for index, row in data.iterrows():
        sid = row[id_column_index]
        if str(sid) in submissions_dict["SID"]:
            student_index = submissions_dict["SID"].index(str(sid))
            
            # Update progress for each submission
            if progress_bar and status_text and total_submissions > 0:
                graded_count += 1
                progress_percentage = 35 + (graded_count / total_submissions) * 45  # 35% to 80%
                progress_bar.progress(int(progress_percentage))
                status_text.text(f"🔄 Grading submission {graded_count}/{total_submissions} - Student {sid}")
            
            points, comments = get_points_and_comments_using_GPT4(
                sid,
                submissions_dict["S_NAME"][student_index],
                submissions_dict["PROCESSED_FILE"][student_index],
                assignment_name,
                possible_points,
                question,
                grading_guidelines,
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


def add_grades_and_comments_ReAct_with_progress(
    submissions,
    gradebook_path,
    assignment_name,
    possible_points,
    question_file_path,
    rubric_file_path,
    temperature,
    selected_model,
    reasoning_level=None,
    progress_bar=None,
    status_text=None,
):
    """Wrapper for M3 methodology with real-time progress tracking"""
    if progress_bar and status_text:
        status_text.text("🔄 Starting M3 (ReAct) grading...")
        progress_bar.progress(35)
    
    # Count total submissions for progress tracking
    total_submissions = len(submissions.get("SID", []))
    if total_submissions > 0:
        status_text.text(f"🔄 Grading {total_submissions} submissions using M3 (ReAct)...")
    
    # Use the real-time progress version
    result = add_grades_and_comments_ReAct_with_real_progress(
        submissions,
        gradebook_path,
        assignment_name,
        possible_points,
        question_file_path,
        rubric_file_path,
        temperature,
        selected_model,
        reasoning_level,
        progress_bar,
        status_text,
        total_submissions,
    )
    
    if progress_bar and status_text:
        progress_bar.progress(80)
        status_text.text("✅ M3 grading completed!")
    
    return result


def add_grades_and_comments_COT_with_real_progress(
    submissions_dict,
    directory_path,
    assignment_name,
    possible_points,
    question_file_path,
    rubric_file_path,
    temperature=0,
    selected_model="gpt-3.5-turbo",
    reasoning_level=None,
    progress_bar=None,
    status_text=None,
    total_submissions=0,
):
    """M4 methodology with real-time progress tracking"""
    import os
    import pandas as pd
    from src.autograder.logging import logger
    from src.autograder.utils import (
        read_file_content,
        find_csv_filename,
    )
    from src.autograder.methodologyCOT import get_points_and_comments_using_GPT4

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

    comments_list = []
    graded_count = 0
    
    for index, row in data.iterrows():
        sid = row[id_column_index]
        if str(sid) in submissions_dict["SID"]:
            student_index = submissions_dict["SID"].index(str(sid))
            
            # Update progress for each submission
            if progress_bar and status_text and total_submissions > 0:
                graded_count += 1
                progress_percentage = 35 + (graded_count / total_submissions) * 45  # 35% to 80%
                progress_bar.progress(int(progress_percentage))
                status_text.text(f"🔄 Grading submission {graded_count}/{total_submissions} - Student {sid}")
            
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

    updated_gradebook_path = csv_file_path.replace(".csv", "_updatedM4.csv")
    try:
        updated_gradebook_df.to_csv(updated_gradebook_path, index=False, header=False)
        logger.info(
            f"Updated gradebook with '{assignment_name}' saved to {updated_gradebook_path}."
        )
    except Exception as e:
        logger.error(f"Failed to save the updated gradebook: {e}")
        return None

    return updated_gradebook_path, comments_list


def add_grades_and_comments_COTMultiCalls_with_real_progress(
    submissions_dict,
    directory_path,
    assignment_name,
    possible_points,
    question_file_path,
    rubric_file_path,
    temperature=0,
    selected_model="gpt-3.5-turbo",
    reasoning_level=None,
    progress_bar=None,
    status_text=None,
    total_submissions=0,
):
    """M5 methodology with real-time progress tracking"""
    import os
    import pandas as pd
    from src.autograder.logging import logger
    from src.autograder.utils import (
        read_file_content,
        find_csv_filename,
    )
    from src.autograder.methodologyCOTMultiCalls import (
        generate_deduction_plan,
        get_points_and_comments_using_GPT4,
        DeductionPlan,
        Criterion,
    )

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
    logger.info("Generating deduction plan for M5 methodology...")
    deduction_plan = generate_deduction_plan(question, rubric, possible_points, temperature, selected_model, reasoning_level)
    logger.info(f"Deduction plan generated with {len(deduction_plan.criteria)} criteria")

    if question is None or rubric is None:
        logger.error("Failed to read question or rubric file.")
        return None
    
    # Check if deduction plan is valid
    if not deduction_plan or not deduction_plan.criteria:
        logger.warning("Deduction plan is empty or invalid. Proceeding with empty plan.")
        # Create a minimal fallback deduction plan
        deduction_plan = DeductionPlan(criteria={
            "General Requirements": Criterion(
                description="Basic assignment requirements",
                deduction=1.0
            )
        })

    comments_list = []
    graded_count = 0
    
    for index, row in data.iterrows():
        sid = row[id_column_index]
        if str(sid) in submissions_dict["SID"]:
            student_index = submissions_dict["SID"].index(str(sid))
            
            # Update progress for each submission
            if progress_bar and status_text and total_submissions > 0:
                graded_count += 1
                progress_percentage = 35 + (graded_count / total_submissions) * 45  # 35% to 80%
                progress_bar.progress(int(progress_percentage))
                status_text.text(f"🔄 Grading submission {graded_count}/{total_submissions} - Student {sid}")
            
            points, comments = get_points_and_comments_using_GPT4(
                sid,
                submissions_dict["S_NAME"][student_index],
                submissions_dict["PROCESSED_FILE"][student_index],
                assignment_name,
                possible_points,
                question,
                rubric,
                deduction_plan,
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


def add_grades_and_comments_COT_with_progress(
    submissions,
    gradebook_path,
    assignment_name,
    possible_points,
    question_file_path,
    rubric_file_path,
    temperature,
    selected_model,
    reasoning_level=None,
    progress_bar=None,
    status_text=None,
):
    """Wrapper for M4 methodology with real-time progress tracking"""
    if progress_bar and status_text:
        status_text.text("🔄 Starting M4 (COT) grading...")
        progress_bar.progress(35)
    
    # Count total submissions for progress tracking
    total_submissions = len(submissions.get("SID", []))
    if total_submissions > 0:
        status_text.text(f"🔄 Grading {total_submissions} submissions using M4 (COT)...")
    
    # Use the real-time progress version
    result = add_grades_and_comments_COT_with_real_progress(
        submissions,
        gradebook_path,
        assignment_name,
        possible_points,
        question_file_path,
        rubric_file_path,
        temperature,
        selected_model,
        reasoning_level,
        progress_bar,
        status_text,
        total_submissions,
    )
    
    if progress_bar and status_text:
        progress_bar.progress(80)
        status_text.text("✅ M4 grading completed!")
    
    return result


def add_grades_and_comments_COTMultiCalls_with_progress(
    submissions,
    gradebook_path,
    assignment_name,
    possible_points,
    question_file_path,
    rubric_file_path,
    temperature,
    selected_model,
    reasoning_level=None,
    progress_bar=None,
    status_text=None,
):
    """Wrapper for M5 methodology with real-time progress tracking"""
    if progress_bar and status_text:
        status_text.text("🔄 Starting M5 (COT Multi-Calls) grading...")
        progress_bar.progress(35)
    
    # Count total submissions for progress tracking
    total_submissions = len(submissions.get("SID", []))
    if total_submissions > 0:
        status_text.text(f"🔄 Grading {total_submissions} submissions using M5 (COT Multi-Calls)...")
    
    # Use the real-time progress version
    result = add_grades_and_comments_COTMultiCalls_with_real_progress(
        submissions,
        gradebook_path,
        assignment_name,
        possible_points,
        question_file_path,
        rubric_file_path,
        temperature,
        selected_model,
        reasoning_level,
        progress_bar,
        status_text,
        total_submissions,
    )
    
    if progress_bar and status_text:
        progress_bar.progress(80)
        status_text.text("✅ M5 grading completed!")
    
    return result


def add_grades_and_comments_with_real_progress(
    submissions_dict,
    directory_path,
    assignment_name,
    possible_points,
    question_file_path,
    rubric_file_path,
    temperature=0,
    selected_model="gpt-3.5-turbo",
    reasoning_level=None,
    progress_bar=None,
    status_text=None,
    total_submissions=0,
):
    """M1 methodology with real-time progress tracking"""
    import os
    import nbformat
    import pandas as pd
    from src.autograder.logging import logger
    import re
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

    comments_list = []
    graded_count = 0
    
    for index, row in data.iterrows():
        sid = row[id_column_index]
        if str(sid) in submissions_dict["SID"]:
            student_index = submissions_dict["SID"].index(str(sid))
            
            # Update progress for each submission
            if progress_bar and status_text and total_submissions > 0:
                graded_count += 1
                progress_percentage = 35 + (graded_count / total_submissions) * 45  # 35% to 80%
                progress_bar.progress(int(progress_percentage))
                status_text.text(f"🔄 Grading submission {graded_count}/{total_submissions} - Student {sid}")
            
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
    """Helper function for M1 grading"""
    from src.autograder.utils import get_completion, GradingComment
    import re
    import json
    from pydantic import ValidationError
    
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


com_list = list()


def format_elapsed_time(seconds):
    """Format elapsed time in a human-readable format"""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        remaining_seconds = seconds % 60
        return f"{minutes}m {remaining_seconds:.1f}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        remaining_seconds = seconds % 60
        return f"{hours}h {minutes}m {remaining_seconds:.1f}s"


def process_submissions_ui():
    if not verify_api_key(api_key):
        st.error("Invalid API Key provided. Please enter a valid API key.")
        return  # Stop execution if the API key is invalid
    
    # Create status containers with enhanced styling
    st.markdown("### 📊 Grading Progress")
    status_container = st.container()
    
    with status_container:
        col1, col2 = st.columns([3, 1])
        with col1:
            status_text = st.empty()
            progress_details = st.empty()
        with col2:
            progress_bar = st.progress(0)
    
    # Initialize timer
    start_time = time.time()
    timer_display = st.empty()
    
    # Main processing logic
    with tempfile.TemporaryDirectory() as temp_dir:
        # Step 1: Setup files
        status_text.text("📁 Setting up files...")
        progress_details.text("Creating temporary directories and saving uploaded files...")
        progress_bar.progress(10)
        timer_display.text("⏱️ Elapsed: 0s")
        
        submissions_dir = os.path.join(temp_dir, "submissions")
        os.makedirs(submissions_dir)
        for uploaded_file in uploaded_files:
            file_path = os.path.join(submissions_dir, uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getvalue())

        gradebook_dir = os.path.join(temp_dir, "gradebook")
        os.makedirs(gradebook_dir)
        gradebook_file_path = os.path.join(gradebook_dir, uploaded_gradebook.name)
        with open(gradebook_file_path, "wb") as f:
            f.write(uploaded_gradebook.getvalue())

        question_path = os.path.join(temp_dir, "question.txt")
        with open(question_path, "w", encoding="utf-8") as f:
            f.write(question)

        rubric_path = os.path.join(temp_dir, "rubric.txt")
        with open(rubric_path, "w", encoding="utf-8") as f:
            f.write(rubric)

        # Step 2: Process submissions
        elapsed = time.time() - start_time
        status_text.text("📋 Processing submissions...")
        progress_details.text("Reading and parsing student submission files...")
        progress_bar.progress(20)
        timer_display.text(f"⏱️ Elapsed: {format_elapsed_time(elapsed)}")
        submissions_dict = process_submissions(submissions_dir)

        # Step 3: Apply methodology with progress tracking
        elapsed = time.time() - start_time
        status_text.text(f"🤖 Applying {methodology} methodology...")
        progress_details.text(f"Starting {methodology} grading process...")
        progress_bar.progress(30)
        timer_display.text(f"⏱️ Elapsed: {format_elapsed_time(elapsed)}")
        
        updated_gradebook_path, comments_list = apply_methodology_with_progress(
            methodology=methodology,
            submissions=submissions_dict,
            gradebook_path=gradebook_dir,
            assignment_name=assignment_name,
            possible_points=possible_points,
            question_file_path=question_path,
            rubric_file_path=rubric_path,
            temperature=temperature,
            selected_model=selected_model,
            reasoning_level=reasoning_level,
            progress_bar=progress_bar,
            status_text=status_text,
        )
        com_list = comments_list

        # Step 4: Clean and prepare results
        elapsed = time.time() - start_time
        status_text.text("🧹 Cleaning and preparing results...")
        progress_details.text("Processing grades and comments for download...")
        progress_bar.progress(85)
        timer_display.text(f"⏱️ Elapsed: {format_elapsed_time(elapsed)}")
        
        # New part to clean the grades starts here
        if updated_gradebook_path and os.path.exists(updated_gradebook_path):
            gradebook_df = pd.read_csv(updated_gradebook_path)
            # Apply the clean_grade function to the assignment_name column
            gradebook_df[assignment_name] = gradebook_df[assignment_name].apply(
                clean_grade
            )
            # Write the cleaned DataFrame back to the CSV
            gradebook_df.to_csv(updated_gradebook_path, index=False)

            with open(updated_gradebook_path, "rb") as file:
                st.session_state["gradebook_content"] = file.read()

        # Preparing comments content
        if comments_list:
            comments_df = pd.DataFrame(comments_list)
            comments_df["Points"] = comments_df["Points"].apply(clean_grade)
            comments_df["Comments"] = comments_df["Comments"].apply(clean_comment)
            st.session_state["comments_content"] = comments_df.to_csv(
                index=False
            ).encode("utf-8")

        # Step 5: Complete
        total_elapsed = time.time() - start_time
        status_text.text("✅ Grading complete!")
        progress_details.text("All submissions have been graded successfully!")
        progress_bar.progress(100)
        timer_display.text(f"⏱️ Total Time: {format_elapsed_time(total_elapsed)}")

        # Display success or error messages with timing information
        if updated_gradebook_path and comments_list:
            # Calculate timing statistics
            total_submissions = len(submissions_dict.get("SID", []))
            avg_time_per_submission = total_elapsed / total_submissions if total_submissions > 0 else 0
            
            # Display comprehensive timing summary
            st.success("🎉 Grading complete! Download your results below.")
            
            # Create timing summary
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Time", format_elapsed_time(total_elapsed))
            with col2:
                st.metric("Submissions Graded", total_submissions)
            with col3:
                st.metric("Avg Time/Submission", f"{avg_time_per_submission:.1f}s")
            
            # Display methodology-specific timing info
            st.info(f"📊 **{methodology} Methodology Performance:**\n"
                   f"- Total grading time: {format_elapsed_time(total_elapsed)}\n"
                   f"- Submissions processed: {total_submissions}\n"
                   f"- Average time per submission: {avg_time_per_submission:.1f} seconds")
            
            st.balloons()
        else:
            st.error(
                "❌ Failed to process submissions. Please check the logs for errors."
            )


# Define the function to clean the grade values
def clean_grade(value):
    try:
        # First check if value is a string that needs to be converted to a tuple
        if isinstance(value, str) and value.startswith("('points',"):
            # Convert string to actual tuple
            value = ast.literal_eval(value)
        if isinstance(value, tuple):
            # If it's a tuple, return the second element (the grade)
            return value[1]
        return value
    except Exception as e:
        # Log the error
        print(f"Error converting grade: {e}")
        return value


def clean_comment(value):
    try:
        # First check if value is a string that needs to be converted to a tuple
        if isinstance(value, str) and value.startswith("('comments',"):
            # Convert string to actual tuple
            value = ast.literal_eval(value)
        if isinstance(value, tuple):
            # If it's a tuple, return the second element (the comment)
            return value[1]
        return value
    except Exception as e:
        # Log the error
        print(f"Error cleaning comment: {e}")
        return value


st.title("GradePilot")

# Define session state variables for storing file data if not already done
if "gradebook_content" not in st.session_state:
    st.session_state["gradebook_content"] = None
if "comments_content" not in st.session_state:
    st.session_state["comments_content"] = None

# Sidebar: File Uploaders and Assignment Details
with st.sidebar:
    st.subheader("Upload Files")
    uploaded_files = st.file_uploader(
        "Upload Student Submissions", accept_multiple_files=True, key="files"
    )
    uploaded_gradebook = st.file_uploader(
        "Upload Gradebook", type=["xlsx", "csv"], key="gradebook"
    )

    st.subheader("Assignment Details and Grading Methodology")
    assignment_name = st.text_input("Assignment Name")
    possible_points = st.number_input("Possible Points", min_value=0)
    methodology_options = ["M1", "M2", "M3", "M4", "M5"]
    methodology = st.selectbox(
        "Select Grading Methodology", options=methodology_options
    )
    model_options = ["gpt-5", "gpt-4.1", "gpt-4.1-mini", "gpt-4.1-nano", "gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo"]
    selected_model = st.selectbox(
        "Select AI Model", options=model_options, index=1
    )
    
    # Handle model-specific settings
    if selected_model == "gpt-5":
        temperature = 1  # GPT-5 only supports temperature=1 (default)
        st.info("ℹ️ GPT-5 uses fixed temperature=1 (default)")
        
        # Reasoning level selector for GPT-5
        reasoning_options = ["minimal", "low", "medium", "high"]
        reasoning_level = st.selectbox(
            "Select Reasoning Level", 
            options=reasoning_options, 
            index=2,  # Default to "medium"
            help="minimal: Fastest, simple tasks. low: Good balance. medium: Default, balanced. high: Best quality for complex tasks."
        )
        
        # Set defaults for verbosity and output tokens
        output_verbosity = "medium"  # Default verbosity
        max_output_tokens = 2048     # Default max tokens
    elif selected_model in ["gpt-4.1", "gpt-4.1-mini", "gpt-4.1-nano"]:
        # GPT-4.1 models support custom temperature but do not support reasoning levels
        temperature = st.slider("Model Temperature", min_value=0.0, max_value=2.0, value=0.0, step=0.1)
        reasoning_level = None  # GPT-4.1 models do not support reasoning levels
        
        if selected_model == "gpt-4.1":
            st.info("ℹ️ GPT-4.1: Advanced model with enhanced capabilities")
        elif selected_model == "gpt-4.1-mini":
            st.info("ℹ️ GPT-4.1mini: Optimized for speed and efficiency")
        elif selected_model == "gpt-4.1-nano":
            st.info("ℹ️ GPT-4.1-nano: Ultra-fast, lightweight model")
    else:
        # Standard models (gpt-4o, gpt-4-turbo, gpt-3.5-turbo)
        temperature = st.slider("Model Temperature", min_value=0.0, max_value=2.0, value=0.0, step=0.1)
        reasoning_level = None  # Not applicable for standard models
    st.subheader("API Key")
    api_key = st.text_input("Enter your OpenAI API Key", type="password")
# Main Area: Question, Rubric, and Process Button
question = st.text_area("Question", "Enter the question here...", height=200)
rubric = st.text_area("Rubric", "Enter the rubric here...", height=300)

# buttons
if st.button("Process Submissions") and uploaded_files and uploaded_gradebook:
    if verify_api_key(
        api_key
    ):  # Assuming verify_api_key returns True if the key is valid
        process_submissions_ui()
    else:
        st.error(
            "The API key provided is invalid. Please check your key and try again."
        )


# Display download buttons if content is available
if st.session_state.get("gradebook_content"):
    st.download_button(
        label="Download Updated Gradebook",
        data=st.session_state["gradebook_content"],
        file_name="updated_gradebook.csv",
        mime="text/csv",
    )

if st.session_state.get("comments_content"):
    st.download_button(
        label="Download Comments",
        data=st.session_state["comments_content"],
        file_name="comments.csv",
        mime="text/csv",
    )


if st.session_state.get("gradebook_content"):
    gradebook_df = pd.read_csv(
        io.StringIO(st.session_state["gradebook_content"].decode("utf-8"))
    )
    # st.write(gradebook_df)
    grades_df = pd.to_numeric(gradebook_df[assignment_name], errors="coerce").dropna()
    st.write("Summary statistics for assignment: " + assignment_name)
    with st.expander("See summary statistics"):
        st.table(grades_df.describe())

    # Set theme
    sns.set_theme(style="whitegrid")

    # Create enhanced boxplot
    plt.figure(figsize=(10, 6))  # Set figure size for better readability
    ax = sns.boxplot(
        x=grades_df,
        orient="h",
        palette="Greens",  # Use a palette for color
        showmeans=True,
        meanprops={
            "marker": "D",  # Use a diamond shape for the mean
            "markerfacecolor": "red",  # Highlight mean marker
            "markeredgecolor": "black",
            "markersize": "10",
        },
        linewidth=2.5,  # Thicker box lines
        fliersize=5,  # Adjust outlier marker size
    )

    # Add annotations for mean and median
    mean_val = grades_df.mean()
    median_val = grades_df.median()
    plt.text(
        mean_val, 0.5, f"Mean: {mean_val:.2f}", color="red", ha="center", va="center"
    )
    plt.text(
        median_val,
        0.3,
        f"Median: {median_val:.2f}",
        color="blue",
        ha="center",
        va="center",
    )

    # Title and labels
    plt.title(f"Grades Distribution for {assignment_name}", fontsize=16)
    plt.xlabel("Grades", fontsize=14)

    # Display the plot
    st.pyplot(plt.gcf())

    # Clear the plot
    plt.clf()
