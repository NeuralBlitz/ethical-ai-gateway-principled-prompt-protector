# scripts/update_readme.py
import re
import sys

def update_readme_hf_url(readme_path, hf_space_url):
    """
    Updates the README.md file with the live Hugging Face Space URL.
    """
    try:
        with open(readme_path, 'r') as f:
            content = f.read()

        # Regex to find and replace the Hugging Face Spaces URL
        # Looks for the line containing "Hugging Face Spaces" badge and updates the URL
        # This regex is specific to the markdown badge format: [![...](URL)]
        pattern = r"(\[!\[Hugging Face Spaces\]\((.*?)\)\]\()(https://huggingface.co/spaces/[\w-]+/[\w-]+)(\))"
        replacement_url = f"https://huggingface.co/spaces/{hf_space_url}"
        
        # We need to correctly capture the parts of the original markdown link.
        # Let's simplify and assume the badge URL always points to the space.
        # Find the line that starts with [![Hugging Face Spaces]
        
        updated_content = content
        
        # More robust: find the line with the badge and modify it
        lines = content.splitlines()
        for i, line in enumerate(lines):
            if "[![Hugging Face Spaces](" in line:
                # This line contains the badge, replace the URL part
                updated_content = re.sub(r"\]\((https://huggingface.co/spaces/[\w-]+/[\w-]+)\)", f")]({replacement_url})", content)
                break
        
        if updated_content == content:
            print(f"Warning: Hugging Face Space URL pattern not found in {readme_path}. Appending URL.")
            updated_content += f"\n\nLive on Hugging Face Spaces: {replacement_url}\n"


        with open(readme_path, 'w') as f:
            f.write(updated_content)
        print(f"Updated {readme_path} with Hugging Face Space URL: {replacement_url}")

    except Exception as e:
        print(f"Error updating README.md: {e}")
        sys.exit(1)

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python update_readme.py <hf_space_repo_id>")
        sys.exit(1)
    
    readme_file = "README.md" # Assuming README.md is in the root
    hf_space_repo_id = sys.argv[1] # e.g., ethical-ai-gateway/prompt-protector-demo

    update_readme_hf_url(readme_file, hf_space_repo_id)
