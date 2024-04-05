import streamlit as st
from src.autograder.utils import process_submissions
from main import apply_methodology
import tempfile
import os
import shutil
from src.autograder.logging import logger
from src.autograder.methodology import add_grades_and_comments
from src.autograder.utils import process_submissions
from src.autograder.methodologyCOTRAG import (
    add_grades_and_comments_COTRAG,
    generate_data_store,
)
from src.autograder.methodologyReAct import add_grades_and_comments_ReAct
from src.autograder.methodologyCOT import add_grades_and_comments_COT
from src.autograder.methodologyCOTMultiCalls import (
    add_grades_and_comments_COTMultiCalls,
)
import pandas as pd


def apply_methodology(
    methodology,
    submissions,
    gradebook_path,
    assignment_name,
    possible_points,
    question_file_path,
    rubric_file_path,
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
        )
    else:
        logger.error("Unknown methodology.")
        return None, None


st.title("AutoGrader")

# Define session state variables for storing file data if not already done
if "gradebook_content" not in st.session_state:
    st.session_state["gradebook_content"] = None
if "comments_content" not in st.session_state:
    st.session_state["comments_content"] = None

methodology_options = ["M1", "M2", "M3", "M4", "M5"]
methodology = st.selectbox("Select Grading Methodology", options=methodology_options)

uploaded_files = st.file_uploader(
    "Upload Student Submissions", accept_multiple_files=True
)
uploaded_gradebook = st.file_uploader("Upload Gradebook", type=["xlsx", "csv"])

assignment_name = st.text_input("Assignment Name")
possible_points = st.number_input("Possible Points", min_value=0)
question = st.text_area("Question")
rubric = st.text_area("Rubric")


def process_submissions_ui():
    with st.spinner("Processing... Please wait."):
        with tempfile.TemporaryDirectory() as temp_dir:
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

            submissions_dict = process_submissions(submissions_dir)

            updated_gradebook_path, comments_list = apply_methodology(
                methodology=methodology,
                submissions=submissions_dict,
                gradebook_path=gradebook_dir,
                assignment_name=assignment_name,
                possible_points=possible_points,
                question_file_path=question_path,
                rubric_file_path=rubric_path,
            )

            # Handle results and store in session state
            if updated_gradebook_path and comments_list:
                # Reading the gradebook content
                with open(updated_gradebook_path, "rb") as file:
                    st.session_state["gradebook_content"] = file.read()

                # Preparing comments content
                comments_df = pd.DataFrame(comments_list)
                st.session_state["comments_content"] = comments_df.to_csv(
                    index=False
                ).encode("utf-8")

                st.success("Grading complete!")
                st.balloons()
            else:
                st.error(
                    "Failed to process submissions. Please check the logs for errors."
                )


if st.button("Process Submissions") and uploaded_files and uploaded_gradebook:
    process_submissions_ui()

# Display download buttons if content is available
if st.session_state["gradebook_content"]:
    st.download_button(
        label="Download Updated Gradebook",
        data=st.session_state["gradebook_content"],
        file_name="updated_gradebook.csv",
        mime="text/csv",
    )

if st.session_state["comments_content"]:
    st.download_button(
        label="Download Comments",
        data=st.session_state["comments_content"],
        file_name="comments.csv",
        mime="text/csv",
    )
