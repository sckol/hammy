#!/usr/bin/python3
import sys
import os
from jinja2 import Environment, FileSystemLoader

def process_jinja_files(directory_path):
    template_loader = FileSystemLoader(searchpath=directory_path)
    jinja_env = Environment(loader=template_loader)

    for file_name in os.listdir(directory_path):
        if file_name.endswith(".sql.jinja2"):
            template_path = os.path.join(directory_path, file_name)
            output_path = os.path.join(directory_path, file_name.replace(".jinja2", ""))
            
            # Render the template
            template = jinja_env.get_template(file_name)
            rendered_content = template.render()

            # Write the rendered content to the output file
            with open(output_path, "w") as output_file:
                output_file.write(rendered_content)

            print(f"Processed: {template_path} -> {output_path}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: process_jinja <directory>")
        sys.exit(1)
    process_jinja_files(sys.argv[1])