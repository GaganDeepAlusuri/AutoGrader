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
  </style>
""",
    unsafe_allow_html=True,
)


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
                st.session_state["comments_content"] = comments_df.to_csv(
                    index=False
                ).encode("utf-8")

            # Display success or error messages
            if updated_gradebook_path and comments_list:
                st.success("Grading complete!")
                st.balloons()
            else:
                st.error(
                    "Failed to process submissions. Please check the logs for errors."
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


st.title("AutoGrader")

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
# Main Area: Question, Rubric, and Process Button
question = st.text_area("Question", "Enter the question here...", height=200)
rubric = st.text_area("Rubric", "Enter the rubric here...", height=300)

# buttons
if st.button("Process Submissions") and uploaded_files and uploaded_gradebook:
    process_submissions_ui()

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
plt.text(mean_val, 0.5, f"Mean: {mean_val:.2f}", color="red", ha="center", va="center")
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
