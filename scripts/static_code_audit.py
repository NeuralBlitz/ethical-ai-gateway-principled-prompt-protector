import os
import sys

def run_static_ethical_audit(src_directory):
    problematic_keywords = ["bomb", "weaponize", "kill_all", "exploit_vulnerability", "circumvent_ethics", "hide_data", "mislead_user", "gender_bias", "racial_slur"]
    flagged_files = {}
    for root, _, files in os.walk(src_directory):
        for file_name in files:
            if file_name.endswith(".py"):
                file_path = os.path.join(root, file_name)
                try:
                    with open(file_path, "r") as f:
                        content = f.read()
                        found_issues = [k for k in problematic_keywords if k in content]
                        if found_issues:
                            flagged_files[file_path] = found_issues
                except Exception:
                    pass
    if flagged_files:
        print("Static Ethical Audit FAILED")
        sys.exit(1)
    print("Static Ethical Audit PASSED")
    return True

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python static_code_audit.py <src_directory>")
        sys.exit(1)
    run_static_ethical_audit(sys.argv[1])
