"""
This script concatenates code from a specified list of files into a single output file.

Usage:
python selective_code_concatenator.py [input_list_file] [output_file]

Example:
python selective_code_concatenator.py utils/current_workspace_code.txt selective_codebase.txt

"""


import os
import argparse

def selective_codebase_to_text(input_list_file, output_file):
    """
    Reads a list of file paths from input_list_file, concatenates their content
    into output_file in the specified format, and estimates token count.

    Args:
        input_list_file (str): Path to the file containing a list of file paths to process.
                               Each path in this file should be prefixed with '--- /'.
        output_file (str): Path to the output text file.
    """
    formatted_contents = []
    total_estimated_tokens = 0
    files_processed_count = 0
    files_not_found_count = 0

    try:
        with open(input_list_file, 'r', encoding='utf-8') as f_list:
            for line_number, line in enumerate(f_list, 1):
                line = line.strip()
                if line.startswith("--- /"):
                    # The actual path is after "--- /"
                    # Example: "--- /path/to/file.py" -> "path/to/file.py"
                    relative_file_path = line[len("--- /"):].strip()
                    
                    if not relative_file_path:
                        print(f"Warning: Empty path found in {input_list_file} at line {line_number}.")
                        continue

                    # Assuming the script is run from the project root,
                    # and paths in current_workspace_code.txt are relative to the project root.
                    file_path_to_read = relative_file_path

                    try:
                        with open(file_path_to_read, 'r', encoding='utf-8', errors='ignore') as f_content:
                            content = f_content.read()
                        
                        tokens = content.split()  # Basic split by whitespace
                        total_estimated_tokens += len(tokens)

                        # Re-add the original "--- /path" prefix for the output
                        formatted_contents.append(line) 
                        formatted_contents.append(content)
                        formatted_contents.append("")  # Add a newline for separation
                        files_processed_count += 1
                    except FileNotFoundError:
                        print(f"Warning: File not found: {file_path_to_read} (listed in {input_list_file} at line {line_number}). Skipping.")
                        files_not_found_count +=1
                    except Exception as e:
                        print(f"Error reading file {file_path_to_read}: {e}. Skipping.")
                        files_not_found_count +=1
                elif line: # Non-empty line that doesn't match the format
                    print(f"Warning: Invalid line format in {input_list_file} at line {line_number}: '{line}'. Skipping.")

    except FileNotFoundError:
        print(f"Error: Input list file not found: {input_list_file}")
        return
    except Exception as e:
        print(f"Error processing input list file {input_list_file}: {e}")
        return

    if not formatted_contents:
        print("No files were processed or no valid paths found in the input list.")
        if files_processed_count == 0 and files_not_found_count > 0:
             print(f"Total files listed but not found: {files_not_found_count}")
        return

    try:
        with open(output_file, 'w', encoding='utf-8') as out_f:
            out_f.write("\n\n".join(formatted_contents))
        print(f"Successfully processed {files_processed_count} file(s).")
        if files_not_found_count > 0:
            print(f"Number of files listed but not found/processed due to errors: {files_not_found_count}")
        print(f"Selected codebase concatenated and saved to {output_file}")
        print(f"Estimated total tokens in the selected codebase: {total_estimated_tokens}")
    except Exception as e:
        print(f"Error writing to output file {output_file}: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Concatenates code from a specified list of files.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "input_list_file",
        nargs='?',
        default="utils/current_workspace_code.txt",
        help="Path to the file containing a list of file paths to process.\n"
             "Each line should be in the format: '--- /path/to/your/file.py'\n"
             "(default: utils/current_workspace_code.txt)"
    )
    parser.add_argument(
        "output_file",
        nargs='?',
        default="selective_codebase.txt",
        help="Path to the output text file where concatenated code will be saved.\n"
             "(default: selective_codebase.txt)"
    )

    args = parser.parse_args()
    selective_codebase_to_text(args.input_list_file, args.output_file) 