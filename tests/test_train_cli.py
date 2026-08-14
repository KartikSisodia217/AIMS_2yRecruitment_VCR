import subprocess
import sys
import os

def test_train_cli_image_dir_arg():
    """Verify that train.py accepts the --image_dir argument."""
    # Run the train script with --help
    script_path = os.path.join("src", "train.py")
    env = os.environ.copy()
    env["PYTHONPATH"] = os.path.abspath(".")
    result = subprocess.run([sys.executable, script_path, "--help"], capture_output=True, text=True, env=env)
    
    # Check that the script runs successfully
    assert result.returncode == 0
    
    # Check that --image_dir is present in the help output
    assert "--image_dir" in result.stdout, "Expected --image_dir argument not found in train.py help output"
