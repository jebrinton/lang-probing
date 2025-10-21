"""
Script simple para verificar la estructura del proyecto (sin dependencias externas)
"""

import os
import sys

def check_file_exists(filepath, description):
    """Check if a file exists"""
    if os.path.exists(filepath):
        print(f"✓ {description}")
        return True
    else:
        print(f"✗ {description} - NOT FOUND: {filepath}")
        return False

def main():
    print("=" * 60)
    print("SAE Probing System - Structure Verification")
    print("=" * 60)
    
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(base_dir)
    
    checks = []
    
    # Source files
    print("\nSource files:")
    checks.append(check_file_exists("src/__init__.py", "src/__init__.py"))
    checks.append(check_file_exists("src/config.py", "src/config.py"))
    checks.append(check_file_exists("src/utils.py", "src/utils.py"))
    checks.append(check_file_exists("src/data.py", "src/data.py"))
    checks.append(check_file_exists("src/activations.py", "src/activations.py"))
    checks.append(check_file_exists("src/probe.py", "src/probe.py"))
    checks.append(check_file_exists("src/features.py", "src/features.py"))
    checks.append(check_file_exists("src/ablation.py", "src/ablation.py"))
    
    # Test files
    print("\nTest files:")
    checks.append(check_file_exists("tests/__init__.py", "tests/__init__.py"))
    checks.append(check_file_exists("tests/test_data.py", "tests/test_data.py"))
    checks.append(check_file_exists("tests/test_probe.py", "tests/test_probe.py"))
    checks.append(check_file_exists("tests/test_features.py", "tests/test_features.py"))
    
    # Scripts
    print("\nScripts:")
    checks.append(check_file_exists("scripts/verify_setup.py", "scripts/verify_setup.py"))
    checks.append(check_file_exists("scripts/train_probes.py", "scripts/train_probes.py"))
    checks.append(check_file_exists("scripts/find_features.py", "scripts/find_features.py"))
    checks.append(check_file_exists("scripts/run_ablation.py", "scripts/run_ablation.py"))
    
    # Documentation
    print("\nDocumentation:")
    checks.append(check_file_exists("README.md", "README.md"))
    checks.append(check_file_exists("requirements.txt", "requirements.txt"))
    
    # Examples
    print("\nExample files:")
    checks.append(check_file_exists("examples/tense_past_english.txt", "examples/tense_past_english.txt"))
    checks.append(check_file_exists("examples/number_plural_english.txt", "examples/number_plural_english.txt"))
    
    # Directories
    print("\nDirectories:")
    for dir_name in ["outputs/probes", "outputs/features", "outputs/ablations", "logs"]:
        if os.path.exists(dir_name):
            print(f"✓ {dir_name}/")
            checks.append(True)
        else:
            print(f"✗ {dir_name}/ - NOT FOUND")
            checks.append(False)
    
    # Summary
    print("\n" + "=" * 60)
    total = len(checks)
    passed = sum(checks)
    print(f"Summary: {passed}/{total} checks passed")
    print("=" * 60)
    
    if passed == total:
        print("\n✓ All files and directories present!")
        print("\nNext steps:")
        print("  1. Install dependencies: pip install -r requirements.txt")
        print("  2. Verify setup: python scripts/verify_setup.py")
        print("  3. Train probes: python scripts/train_probes.py")
        return 0
    else:
        print(f"\n⚠ {total - passed} items missing")
        return 1

if __name__ == "__main__":
    sys.exit(main())

