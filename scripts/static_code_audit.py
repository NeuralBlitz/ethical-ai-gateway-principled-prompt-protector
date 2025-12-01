# scripts/static_code_audit.py
import os
import sys

def run_static_ethical_audit(src_directory):
    """
    Performs a basic static analysis of code for hardcoded problematic patterns.
    This acts as a SentiaGuard pre-gate for the code itself.
    """
    problematic_keywords = [
        "bomb", "weaponize", "kill_all", "exploit_vulnerability", "circumvent_ethics",
        "hide_data", "mislead_user", "gender_bias", "racial_slur"
    ]
    
    flagged_files = {}

    print(f"Running static ethical audit on {src_directory}...")

    for root, _, files in os.walk(src_directory):
        for file_name in files:
            if file_name.endswith(".py"): # Only check Python files for now
                file_path = os.path.join(root, file_name)
                try:
                    with open(file_path, 'r') as f:
                        content = f.read()
                        found_issues = []
                        for keyword in problematic_keywords:
                            if keyword in content:
                                found_issues.append(f"Contains problematic keyword: '{keyword}'")
                        if found_issues:
                            flagged_files[file_path] = found_issues
                except Exception as e:
                    print(f"Error reading {file_path}: {e}")

    if flagged_files:
        print("\n--- Static Ethical Audit FAILED ---")
        for file, issues in flagged_files.items():
            print(f"File: {file}")
            for issue in issues:
                print(f"  - {issue}")
        print("\nDeployment blocked due to detected ethical concerns in code. (Φ1, Φ4, Φ5)")
        sys.exit(1) # Fail the CI/CD job
    else:
        print("\n--- Static Ethical Audit PASSED --- No problematic keywords found. ---")
        return True

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python static_code_audit.py <src_directory>")
        sys.exit(1)
    run_static_ethical_audit(sys.argv[1])
