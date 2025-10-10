import os
import sys

def write_folder_structure(directory, output_file, ignore_patterns):
    """
    Write the folder structure to the output file.
    """
    for root, dirs, files in os.walk(directory):
        if any(ignored in root for ignored in ignore_patterns):
            continue
        
        relative_path = os.path.relpath(root, directory)
        if relative_path == ".":
            relative_path = os.path.basename(directory)
        output_file.write(f"{relative_path}/\n")
        
        filtered_files = [file for file in files if not any(file.endswith(ext) for ext in ignore_patterns)]
        for file in sorted(filtered_files):
            output_file.write(f"  {file}\n")


def write_files_content(directory, output_file, ignore_patterns):
    """
    Write the content of files to the output file.
    """
    for root, dirs, files in os.walk(directory):
        if any(ignored in root for ignored in ignore_patterns):
            continue
        
        filtered_files = [file for file in files if not any(file.endswith(ext) for ext in ignore_patterns)]
        for file in sorted(filtered_files):
            file_path = os.path.join(root, file)
            relative_path = os.path.relpath(file_path, directory)
            
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
                output_file.write(f"\n{'-' * 80}\n")
                output_file.write(f"File: {relative_path}\n")
                output_file.write(f"{'-' * 80}\n")
                output_file.write(content)
                output_file.write("\n")
            except Exception as e:
                output_file.write(f"\n{'-' * 80}\n")
                output_file.write(f"File: {relative_path}\n")
                output_file.write(f"{'-' * 80}\n")
                output_file.write(f"Error reading file: {e}\n")


def main():
    """
    Main function to write folder structure and file contents to a .txt file.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    ignore_patterns = [
        "env", "migrations", "__pycache__", ".bak", ".dat", ".dir", 
        "db.sqlite3", ".txt", "prompt.py", ".git", ".pyc", ".venv",
        ".idea", ".vscode", "node_modules", 'prompt.py', ".json", 'preprocessing', 'models_output', '.csv'
    ]
    
    target_directory = script_dir
    
    output_file_name = "codebase.txt"
    
    with open(output_file_name, "w", encoding="utf-8") as output_file:
        output_file.write(f"Project Analysis: {os.path.basename(target_directory)}\n")
        output_file.write(f"Generated on: {sys.platform} operating system\n")
        output_file.write("=" * 80 + "\n\n")
        
        output_file.write("Folder Structure:\n")
        output_file.write("=" * 80 + "\n")
        write_folder_structure(target_directory, output_file, ignore_patterns)
        
        output_file.write("\n\nFile Contents:\n")
        output_file.write("=" * 80 + "\n")
        write_files_content(target_directory, output_file, ignore_patterns)
    
    print(f"Folder structure and file contents written to {output_file_name}")
    print(f"Output file location: {os.path.join(script_dir, output_file_name)}")


if __name__ == "__main__":
    main()