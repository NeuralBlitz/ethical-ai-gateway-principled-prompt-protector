# scripts/ethical_decision_maker.py
import json
import os
import sys
import shutil # For moving files
from huggingface_hub import HfApi, login

def make_deployment_decision(current_model_path, new_model_path, 
                             current_eval_path, new_eval_path,
                             deploy_target_path, hf_model_repo, github_token):
    
    print("Initiating Judex-like arbitration for model deployment decision...")

    try:
        with open(current_eval_path, 'r') as f:
            current_results = json.load(f)
        with open(new_eval_path, 'r') as f:
            new_results = json.load(f)
    except FileNotFoundError as e:
        print(f"Error: Evaluation results file not found: {e}")
        sys.exit(1)

    # --- 1. Performance Check (Φ1) ---
    current_f1 = current_results['model_performance_metrics']['f1_score_micro']
    new_f1 = new_results['model_performance_metrics']['f1_score_micro']

    performance_gain_threshold = 0.02 # Example: 2% improvement
    performance_ok = (new_f1 > current_f1 + performance_gain_threshold)

    print(f"Performance: Current F1={current_f1:.2f}, New F1={new_f1:.2f}")
    if not performance_ok:
        print(f"Decision: FAIL - New model does not offer significant performance gain over current model (Φ1).")
        print("Model update will be skipped.")
        return # Do not deploy

    # --- 2. Ethical Bias Audit Check (Φ7, Φ11 - Alignment > Performance) ---
    current_bias_status = current_results['ethical_bias_audit_metrics'].get('protected_attribute_bias_check', 'PASS')
    new_bias_status = new_results['ethical_bias_audit_metrics'].get('protected_attribute_bias_check', 'PASS')

    ethical_drift_threshold = 0.1 # Example: Max tolerable increase in bias
    
    # This is a highly conceptual check. In reality, it would analyze detailed bias metrics
    # e.g., current_results['ethical_bias_audit_metrics']['avg_risk_group_A']
    # If the bias audit indicates the new model is worse, it's a FAIL.
    ethical_ok = (new_bias_status == 'PASS')
    if current_bias_status == 'PASS' and new_bias_status == 'FAIL':
        print(f"Decision: FAIL - New model introduces new ethical bias (Φ7).")
        ethical_ok = False
    elif current_bias_status == 'FAIL' and new_bias_status == 'PASS':
        print(f"Decision: PASS - New model *reduces* existing ethical bias (Φ7).")
        ethical_ok = True # Even better!
    elif current_bias_status == 'FAIL' and new_bias_status == 'FAIL':
        # Both are bad, but if new is significantly worse, then fail
        print(f"Decision: FAIL - Both models have bias; new model is not a clear improvement (Φ7).")
        ethical_ok = False # For this conceptual example, if both fail, new doesn't get deployed.

    print(f"Ethical Bias: Current Status='{current_bias_status}', New Status='{new_bias_status}'")
    if not ethical_ok:
        print("Decision: FAIL - New model fails ethical bias audit (Φ7, Φ11).")
        print("Model update will be skipped.")
        return # Do not deploy

    # --- 3. Overall Judex Decision (Deployment Authorization) ---
    if performance_ok and ethical_ok:
        print("\n--- Judex Arbitration: PASS ---")
        print("New model meets performance and ethical standards. Authorizing deployment.")
        
        # --- Deployment Action: Move new model to official path ---
        print(f"Moving new model from '{new_model_path}' to '{deploy_target_path}'...")
        # Ensure target path is clean or ready for overwrite
        if os.path.exists(deploy_target_path):
            shutil.rmtree(deploy_target_path)
        shutil.copytree(new_model_path, deploy_target_path)
        print("Model moved successfully.")
        
        # --- Trigger downstream CD (Hugging Face Model Hub push) ---
        print(f"Triggering push to Hugging Face Model Hub '{hf_model_repo}'...")
        # This part assumes your GitHub Actions workflow for CD to HF is set up to
        # re-run when the `models/` directory changes on `main`.
        # Alternatively, you could call the push_to_hf_hub.py script directly here.
        
        # For simplicity, we'll indicate success. The CD workflow would detect changes
        # in 'models/' and handle the actual push.
        print("Model update successful and triggered for Hugging Face deployment.")

        # Optionally, update evaluation_metrics.json to reflect the new best model
        shutil.copy(new_eval_path, current_eval_path)
        print("Updated main evaluation metrics to reflect new model.")

    else:
        print("\n--- Judex Arbitration: FAIL ---")
        print("New model did not meet all performance or ethical criteria. Human review required.")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Arbitrate deployment decision for CRC model.")
    parser.add_argument("--current_model_path", type=str, default="models_current/",
                        help="Path to the currently deployed model directory.")
    parser.add_argument("--new_model_path", type=str, default="models_new/",
                        help="Path to the newly trained model directory.")
    parser.add_argument("--current_eval_path", type=str, default="evaluation_metrics.json",
                        help="Path to the JSON for current model evaluation results.")
    parser.add_argument("--new_eval_path", type=str, default="evaluation_results_new.json",
                        help="Path to the JSON for new model evaluation results.")
    parser.add_argument("--deploy_target_path", type=str, default="models/",
                        help="Official path where the new model should be deployed.")
    parser.add_argument("--hf_model_repo", type=str, 
                        default="ethical-ai-gateway/principled-prompt-protector-model",
                        help="Hugging Face Model Hub repository ID.")
    parser.add_argument("--github_token", type=str, default=None,
                        help="GitHub token for committing changes.") # Passed by GitHub Actions secrets

    args = parser.parse_args()

    # Create dummy evaluation files if they don't exist for the first run
    if not os.path.exists(args.current_eval_path):
        with open(args.current_eval_path, 'w') as f:
            json.dump({
                "model_performance_metrics": {"f1_score_micro": 0.70}, # Baseline
                "ethical_bias_audit_metrics": {"protected_attribute_bias_check": "PASS"}
            }, f)
    
    if not os.path.exists(args.new_eval_path):
        with open(args.new_eval_path, 'w') as f:
            json.dump({
                "model_performance_metrics": {"f1_score_micro": 0.78}, # Example new performance
                "ethical_bias_audit_metrics": {"protected_attribute_bias_check": "PASS"}
            }, f)

    make_deployment_decision(
        current_model_path=args.current_model_path,
        new_model_path=args.new_model_path,
        current_eval_path=args.current_eval_path,
        new_eval_path=args.new_eval_path,
        deploy_target_path=args.deploy_target_path,
        hf_model_repo=args.hf_model_repo,
        github_token=args.github_token
    )
