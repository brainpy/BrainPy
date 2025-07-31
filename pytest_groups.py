#!/usr/bin/env python3
"""
Minimal isolated test runner for BrainPy - only isolates known problematic tests.
"""

import subprocess
import sys
import os
import time

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

# Additional files that need isolation in CI environments
CI_ISOLATED_FILES = [
    "brainpy/_src/math/object_transform/tests/test_autograd.py",  # unhashable Array type in CI
    "brainpy/_src/math/object_transform/tests/test_controls.py",  # unhashable Array type in CI
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
    start_time = time.time()
    
    # GitHub Actions specific settings  
    is_github_actions = os.getenv('IS_GITHUB_ACTIONS') == '1'
    base_cmd = [sys.executable, "-m", "pytest"]
    test_args = ["-v", "--tb=short"]
    if is_github_actions:
        test_args.extend(["--maxfail=5"])
    
    # Build ignore list - only ignore files that contain problematic tests
    ignore_patterns = []
    all_problematic_files = set(ISOLATED_FILES)
    
    # In CI environments, also isolate CI-specific problematic files
    if is_github_actions:
        all_problematic_files.update(CI_ISOLATED_FILES)
    
    # Add files from isolated tests to ignore list
    for test in ISOLATED_TESTS:
        if "::" in test:
            file_path = test.split("::")[0]
            all_problematic_files.add(file_path)
    
    for file_path in all_problematic_files:
        ignore_patterns.append(f"--ignore={file_path}")
    
    print("=" * 80)
    print("BrainPy Test Suite (with state pollution isolation)")
    print("=" * 80)
    
    # Run main test suite (excluding problematic files)
    print(f"\n{'Running main test suite...':<60} ", end="", flush=True)
    
    cmd = base_cmd + ["brainpy/_src/"] + test_args + ignore_patterns
    main_start = time.time()
    main_result = subprocess.run(cmd, capture_output=True, text=True)
    main_time = time.time() - main_start
    main_passed = main_result.returncode == 0
    
    if main_passed:
        print(f"PASSED ({main_time:.1f}s)")
    else:
        print(f"FAILED ({main_time:.1f}s)")
        if not is_github_actions:
            # Extract key info from pytest output
            lines = main_result.stdout.split('\n')
            failed_lines = [line for line in lines if 'FAILED' in line][:5]  # Show first 5 failures
            if failed_lines:
                print("\n   Recent failures:")
                for line in failed_lines:
                    print(f"   {line}")
    
    # Run isolated problematic files
    isolated_results = []
    for file_path in sorted(all_problematic_files):
        if os.path.exists(file_path):
            file_name = file_path.split("/")[-1]
            print(f"{'Isolated: ' + file_name:<60} ", end="", flush=True)
            
            cmd = base_cmd + [file_path] + test_args + ["-x"]
            iso_start = time.time()
            result = subprocess.run(cmd, capture_output=True, text=True)
            iso_time = time.time() - iso_start
            passed = result.returncode == 0
            isolated_results.append(passed)
            
            if passed:
                print(f"PASSED ({iso_time:.1f}s)")
            else:
                print(f"FAILED ({iso_time:.1f}s)")
                if not is_github_actions:
                    lines = result.stdout.split('\n')
                    failed_lines = [line for line in lines if 'FAILED' in line][:3]
                    if failed_lines:
                        print("   Failures:")
                        for line in failed_lines:
                            print(f"   {line}")
        else:
            isolated_results.append(True)
    
    # Final summary in pytest style
    total_time = time.time() - start_time
    print("\n" + "=" * 80)
    
    all_passed = main_passed and all(isolated_results)
    total_groups = 1 + len([f for f in all_problematic_files if os.path.exists(f)])
    passed_groups = (1 if main_passed else 0) + sum(isolated_results)
    failed_groups = total_groups - passed_groups
    
    if all_passed:
        print(f"{'=' * 25} {passed_groups} passed in {total_time:.1f}s {'=' * 25}")
    else:
        status_parts = []
        if failed_groups > 0:
            status_parts.append(f"{failed_groups} failed")
        if passed_groups > 0:
            status_parts.append(f"{passed_groups} passed")
        
        status = ", ".join(status_parts)
        print(f"{'=' * 20} {status} in {total_time:.1f}s {'=' * 20}")
        
        if not all_passed:
            print("\nFailed test groups:")
            if not main_passed:
                print("  - Main test suite")
            for file_path, passed in zip(sorted(all_problematic_files), isolated_results):
                if os.path.exists(file_path) and not passed:
                    print(f"  - {file_path}")
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)