#!/usr/bin/env python3
"""
Minimal isolated test runner for BrainPy - only isolates known problematic tests.
"""

import subprocess
import sys
import os

# Only the specific tests that cause state pollution issues
ISOLATED_TESTS = [
    "brainpy/_src/math/object_transform/tests/test_jit.py::TestClsJIT::test_class_jit1",
    "brainpy/_src/optimizers/tests/test_scheduler.py::TestMultiStepLR::test20",
    "brainpy/_src/optimizers/tests/test_scheduler.py::TestMultiStepLR::test21", 
    "brainpy/_src/optimizers/tests/test_scheduler.py::TestMultiStepLR::test22",
    "brainpy/_src/optimizers/tests/test_scheduler.py::TestMultiStepLR::test23",
    "brainpy/_src/optimizers/tests/test_scheduler.py::TestCosineAnnealingLR::test1",
    "brainpy/_src/optimizers/tests/test_scheduler.py::TestCosineAnnealingWarmRestarts::test1",
    "brainpy/_src/tests/test_base_classes.py::TestDynamicalSystem::test_delay",
    "brainpy/_src/tests/test_dyn_runner.py::TestDSRunner::test_DSView",
]

# Files that contain problematic tests (need to be run separately)
ISOLATED_FILES = [
    "brainpy/_src/math/object_transform/tests/test_base.py",  # causes state pollution
]

def run_isolated_test(test_path):
    """Run a single problematic test in isolation."""
    print(f"Running isolated: {test_path}")
    
    # GitHub Actions specific settings
    is_github_actions = os.getenv('IS_GITHUB_ACTIONS') == '1'
    base_cmd = [sys.executable, "-m", "pytest"]
    test_args = ["-v", "--tb=short", "-x"]
    if is_github_actions:
        test_args.extend(["--maxfail=1", "-q"])
    
    cmd = base_cmd + [test_path] + test_args
    result = subprocess.run(cmd)
    return result.returncode == 0

def main():
    """Run tests with minimal isolation - only isolate known problematic tests."""
    print("Starting minimal isolated test runner for BrainPy...")
    
    # GitHub Actions specific settings  
    is_github_actions = os.getenv('IS_GITHUB_ACTIONS') == '1'
    base_cmd = [sys.executable, "-m", "pytest"]
    test_args = ["-v", "--tb=short"]
    if is_github_actions:
        test_args.extend(["--maxfail=5", "-q"])
    
    # Build ignore list - only ignore files that contain problematic tests
    ignore_patterns = []
    all_problematic_files = set(ISOLATED_FILES)
    
    # Add files from isolated tests to ignore list
    for test in ISOLATED_TESTS:
        if "::" in test:
            file_path = test.split("::")[0]
            all_problematic_files.add(file_path)
    
    for file_path in all_problematic_files:
        ignore_patterns.append(f"--ignore={file_path}")
    
    # Run main test suite (excluding problematic files)
    print("\n" + "="*60)
    print("RUNNING MAIN TEST SUITE (excluding problematic files)")
    print("="*60)
    
    cmd = base_cmd + ["brainpy/_src/"] + test_args + ignore_patterns
    main_result = subprocess.run(cmd)
    main_passed = main_result.returncode == 0
    
    # Run isolated problematic files
    print("\n" + "="*60)
    print("RUNNING ISOLATED PROBLEMATIC FILES")
    print("="*60)
    
    isolated_results = []
    for file_path in sorted(all_problematic_files):
        if os.path.exists(file_path):
            print(f"Running isolated file: {file_path}")
            cmd = base_cmd + [file_path] + test_args + ["-x"]
            result = subprocess.run(cmd)
            isolated_results.append(result.returncode == 0)
        else:
            print(f"Skipping non-existent file: {file_path}")
            isolated_results.append(True)
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    all_passed = main_passed and all(isolated_results)
    
    print(f"Main test suite: {'✓ PASSED' if main_passed else '✗ FAILED'}")
    print(f"Isolated files: {'✓ PASSED' if all(isolated_results) else '✗ FAILED'} ({len([r for r in isolated_results if r])}/{len(isolated_results)})")
    
    if not all_passed:
        print("\nSome tests failed. Re-run individually to debug.")
        return 1
    else:
        print("\nAll tests passed!")
        return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)