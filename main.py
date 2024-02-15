from src.autograder.utils import (
    process_submissions,
    add_grades_and_comments,
    prepare_for_canvas_upload,
    generate_student_reports,
)
import json

if __name__ == "__main__":
    folder_path = input("Enter the path to the folder containing student submissions: ")
    gradebook_path = input("Enter the path to the folder containing the Gradebook: ")
    submissions_dict = process_submissions(folder_path)
    print(submissions_dict)

    # Add percentage grades and comments
    add_grades_and_comments(submissions_dict, gradebook_path)

    """# Print the modified dictionary with grades and comments
    print(modified_submissions_dict)

    # Export the modified dictionary as JSON
    output_file_path = "D:\\AutoGrader\\src\\autograder\\submissions\\output_file.json"
    with open(output_file_path, "w", encoding="utf-8") as json_file:
        json.dump(modified_submissions_dict, json_file, ensure_ascii=False, indent=4)

    print(f"Data saved as JSON to {output_file_path}")

    # Prompt user for Section, Assignment ID, and Assignment Name
    section = input("Enter the Section: ")
    # assignment_name = input("Enter the Assignment Name: ")

    # Use the inputs in calling the `prepare_for_canvas_upload` function
    csv_path = prepare_for_canvas_upload(
        modified_submissions_dict, section, assignment_name
    )
    print(f"CSV file created for Canvas upload: {csv_path}")

    generate_student_reports(
        modified_submissions_dict, r"D:\AutoGrader\reports", assignment_name
    )"""
