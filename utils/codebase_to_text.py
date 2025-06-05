"""
This script concatenates code from a specified project directory into a single output file.

Usage:
python codebase_to_text.py [project_dir] [output_file]

Example:
python codebase_to_text.py . codebase.txt
"""

import os
import argparse

DEFAULT_IGNORE_DIRS = [".git", "__pycache__", "node_modules", "venv", ".vscode", ".idea", "build", "dist", "env"]
DEFAULT_IGNORE_FILES = [".DS_Store"]
DEFAULT_INCLUDE_EXTENSIONS = ['.py', '.pyi', '.ipynb', '.json', '.yaml', '.yml', '.toml', '.md', '.rst', '.sh', '.html', '.css', '.js', '.sql', '.cfg', '.ini']
DEFAULT_INCLUDE_FILENAMES = ['makefile', 'dockerfile', '.gitignore', 'procfile'] # Lowercase for set comparison

def codebase_to_text(project_dir, output_file="codebase.txt", 
                     ignore_dirs=None, ignore_files=None, 
                     include_extensions=None, include_filenames=None):
    """
    Traverses a project folder structure, concatenates specified codebase files,
    and saves it to a text file in the specified format.

    Args:
        project_dir (str): The path to the project directory.
        output_file (str): The path to the output text file.
        ignore_dirs (list, optional): Directory names to ignore. Defaults to common ones.
        ignore_files (list, optional): Specific file names to ignore. Defaults to common ones.
        include_extensions (list, optional): File extensions to include (e.g., ['.py', '.md']). Defaults to Python project files.
        include_filenames (list, optional): Specific filenames to include (e.g., ['Makefile']). Defaults to common ones.
    """
    if ignore_dirs is None:
        ignore_dirs = DEFAULT_IGNORE_DIRS
    if ignore_files is None:
        ignore_files = DEFAULT_IGNORE_FILES
    if include_extensions is None:
        include_extensions = DEFAULT_INCLUDE_EXTENSIONS
    if include_filenames is None:
        include_filenames = DEFAULT_INCLUDE_FILENAMES

    # Convert to sets for efficient lookup, extensions should be lowercased and include the dot
    include_extensions_set = {ext.lower() for ext in include_extensions}
    # Filenames should be lowercased
    include_filenames_set = {fname.lower() for fname in include_filenames}
    # ignore_files should also be a set and lowercased for comparison consistency, though os.walk provides exact names
    ignore_files_set = {fname.lower() for fname in ignore_files}

    formatted_contents = []
    total_estimated_tokens = 0
    project_dir_abs = os.path.abspath(project_dir)

    for root, dirs, files in os.walk(project_dir_abs):
        # Filter out ignored directories (case-sensitive for directory names)
        # Also filter hidden directories unless explicitly allowed elsewhere (though typically handled by ignore_dirs like .git)
        dirs[:] = [d for d in dirs if d not in ignore_dirs and not (d.startswith('.') and d not in [fn for fn in include_filenames if '/' not in fn])]

        for file in files:
            file_lower = file.lower()

            if file_lower in ignore_files_set:
                continue

            filename_stem, file_ext_with_dot = os.path.splitext(file_lower)
            
            is_included = False
            if file_ext_with_dot in include_extensions_set:
                is_included = True
            elif file_lower in include_filenames_set: # Check full filename if extension didn't match or no extension
                is_included = True
            
            if not is_included:
                continue

            file_path = os.path.join(root, file) # Use original file name for path joining
            relative_file_path = os.path.relpath(file_path, project_dir_abs)

            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                
                tokens = content.split() 
                total_estimated_tokens += len(tokens)

                formatted_contents.append(f"--- /{relative_file_path.replace(os.sep, '/')}")
                formatted_contents.append(content)
                formatted_contents.append("") 
            except Exception as e:
                print(f"Error reading file {file_path}: {e}")

    with open(output_file, 'w', encoding='utf-8') as out_f:
        out_f.write("\n\n".join(formatted_contents))
    print(f"Codebase concatenated and saved to {output_file}")
    print(f"Estimated total tokens in codebase (from included files): {total_estimated_tokens}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Concatenate codebase into a single text file.")
    parser.add_argument("project_dir", nargs='?', default=".", help="The path to the project directory (default: current directory '.').")
    parser.add_argument("output_file", nargs='?', default="codebase.txt", help="The path to the output text file (default: codebase.txt).")
    parser.add_argument(
        "--ignore-dirs",
        nargs="*",
        default=DEFAULT_IGNORE_DIRS,
        help=f"Directory names to ignore. Default: {DEFAULT_IGNORE_DIRS}"
    )
    parser.add_argument(
        "--ignore-files",
        nargs="*",
        default=DEFAULT_IGNORE_FILES,
        help=f"File names to ignore. Default: {DEFAULT_IGNORE_FILES}"
    )
    parser.add_argument(
        "--include-extensions",
        nargs="*",
        default=DEFAULT_INCLUDE_EXTENSIONS,
        help=f"File extensions to include (e.g., .py .md). Default: {DEFAULT_INCLUDE_EXTENSIONS}"
    )
    parser.add_argument(
        "--include-filenames",
        nargs="*",
        default=DEFAULT_INCLUDE_FILENAMES,
        help=f"Specific filenames to include (e.g., Makefile .gitignore). Default: {DEFAULT_INCLUDE_FILENAMES}"
    )

    args = parser.parse_args()
    codebase_to_text(args.project_dir, args.output_file, 
                     args.ignore_dirs, args.ignore_files, 
                     args.include_extensions, args.include_filenames)
