from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

# from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OpenAIEmbeddings
import os
import shutil
import nbformat
import pandas as pd
from src.autograder.logging import logger
import re
import random
import numpy as np
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
    get_completionCOT,
    get_completion_keywords,
)


from pydantic import BaseModel, Field


class GradingCommentCOT(BaseModel):
    points: float = Field(..., alias="points")
    comments: str = Field(..., alias="comments")


load_dotenv()

CHROMA_PATH = "chroma"
DATA_PATH = "src/autograder/data"


def add_grades_and_comments_COT(
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
    keywords_from_question = get_completion_keywords(question)
    logger.info(f"Keywords from question: {keywords_from_question}")
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
                keywords_from_question,
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


############ Setting up the vector DB ################


def generate_data_store():
    # Check if the Chroma Vector Store already exists
    if os.path.exists(CHROMA_PATH):
        logger.info("Chroma Vector Store already exists. Skipping embedding process.")
        return

    documents = load_documents()
    chunks = split_text(documents)
    save_to_chroma(chunks)


def load_documents():
    loader = DirectoryLoader(
        DATA_PATH, glob="*.md"
    )  # Now adjusted to include both Markdown and PDF files
    documents = loader.load()
    return documents


def split_text(documents: List[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300, chunk_overlap=100, length_function=len, add_start_index=True
    )
    chunks = text_splitter.split_documents(documents)
    logger.info(f"Split {len(documents)} documents into {len(chunks)} chunks.")
    document = chunks[10]
    print(document.page_content)
    print(document.metadata)
    return chunks


def save_to_chroma(chunks: list[Document]):

    # Create a new DB from the documents.
    db = Chroma.from_documents(
        chunks, OpenAIEmbeddings(), persist_directory=CHROMA_PATH
    )
    db.persist()
    print(f"Saved {len(chunks)} chunks to {CHROMA_PATH}.")


from langchain.vectorstores import Chroma


def query_vector_store(query: str, top_k=3):
    """
    Queries the vector store for documents relevant to the provided query string.

    Args:
        query (str): The query string to search for relevant documents.
        top_k (int): The number of top relevant documents to retrieve.

    Returns:
        A list of document contents that are most relevant to the query.
    """
    embedding_function = OpenAIEmbeddings()
    # Load the Chroma VectorStore
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    # Perform the similarity search with relevance scores.
    results = db.similarity_search_with_relevance_scores(query, k=top_k)
    if not results or results[0][1] < 0.7:
        print("Unable to find matching results with high relevance.")
        return []

    # Format the results into a string of context texts.
    context_texts = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    return context_texts


def get_points_and_comments_using_GPT4(
    sid,
    student_name,
    processed_file,
    assignment_name,
    possible_points,
    question,
    rubric,
    keywords_from_question,
):
    # Query the vector store for relevant contexts based on the assignment question's keywords.
    relevant_contexts = query_vector_store(keywords_from_question, top_k=3)

    # Check if relevant contexts were found. If not, prepare a default message.
    context_str = (
        relevant_contexts if relevant_contexts else "No relevant context was found."
    )
    logger.info(f"Context Retrived:{context_str}")
    user_message = f"Evaluate the student's submission and provide only a JSON output with points and comments:{processed_file}"
    # Construct the prompt for GPT-4 with the relevant contexts, assignment question, rubric, and the student's submission
    prompt_template = f"""
First, consider the following contexts relevant to the assignment: {context_str}

Given the assignment question: {question}
And the grading rubric: {rubric}

Step 1: Verify the submission addresses the key aspects of the assignment question. If it does, note how effectively. If not, identify missing elements.

Step 2: Compare the submission against the grading rubric criteria. For each criterion, decide whether it's met, partially met, or not met, providing specific examples from the submission.

Step 3: Based on the evaluation in steps 1 and 2, calculate the total deductions. Remember, the total possible points are {possible_points}.

Step 4: Provide detailed feedback for each deduction, correlating comments with specific rubric criteria to guide the student's learning and improvement.

Respond with a JSON output with keys 'points' (consisting of final grade after deductions) and 'comments' (A string for any deducted points and reason.).

Below are a few examples of the kind of expected outputs:

Example 1 JSON Output:
{{
  "points": 8.5,
  "comments": "Your submission mostly addresses the key aspects of the assignment question. You loaded the data, explored it, conducted pre-processing, and split the data into training and testing sets. However, there are a few areas where the submission could be improved. First, you scaled the entire dataset before splitting it into training and testing sets, which is not a good practice as it can lead to data leakage. The correct approach is to fit the scaler on the training data and then transform both the training and testing data. Second, you did not provide a rationale for their choice of k in the KNN model. Lastly, your discussion of the results could be more detailed, particularly in comparing the performance of the two models. Therefore, 1.5 points were deducted - 1 point for scaling the entire dataset before splitting and 0.5 points for not providing a rationale for the choice of k in the KNN model."
}}

Example 2 JSON Output:
{{
  "points": 7.5,
  "comments": "Your partially meets the requirements of the assignment. You successfully loaded the data, conducted a train/test split, and fitted both a Linear Regression and KNN model. However, there were several areas where the submission fell short. First, you did not provide a title or introduction for their analysis, resulting in a deduction of 0.5 points. Second, you did not discuss their selection of k value for the KNN model, resulting in a deduction of 1.0 points. Lastly, you did not recap their analysis or discuss the performance of the models using the RMSE metric, resulting in a deduction of 1.0 points. To improve, you should ensure they provide a clear introduction and conclusion for their analysis, and thoroughly discuss your model selection and performance evaluation process."
}}
""".strip()
    try:
        response_message = get_completionCOT(prompt_template, user_message)
        logger.info(f"Response: {response_message}")

        # Assuming the response is well-structured JSON, parse it directly
        grading_info = GradingCommentCOT.parse_raw(response_message)
        return grading_info.points, grading_info.comments
    except Exception as e:
        logger.error(f"Error processing grading for student ID {sid}: {e}")
        return 0, "An error occurred during grading."
