import os

# Directory where photomacros is located
root_dir = os.path.abspath('../photomacros')

# List all Python files in 'photomacros' and its subdirectories (including 'modeling'), but exclude 'sklearn-env' folder
def generate_autodoc_directives():
    for root, dirs, files in os.walk(root_dir):
        # Skip 'sklearn-env' folder if it's inside 'modeling'
        if 'sklearn-env' in root:
            continue  # Skip this directory and its subdirectories

        # Only process directories that are 'photomacros' or 'modeling' and subdirectories
        if 'photomacros' in root or 'modeling' in root:
            for file in files:
                if file.endswith('.py') and file != '__init__.py':  # Avoid __init__.py
                    # Create the module path
                    module_path = os.path.relpath(os.path.join(root, file), start=root_dir)
                    module_name = module_path.replace(os.path.sep, '.')[:-3]  # Remove .py extension
                    
                    # Print the automodule directive
                    print(f'.. automodule:: {module_name}')
                    print('    :members:')
                    print('    :undoc-members:')
                    print('    :show-inheritance:')
                    print()  # Blank line to separate entries

if __name__ == "__main__":
    generate_autodoc_directives()