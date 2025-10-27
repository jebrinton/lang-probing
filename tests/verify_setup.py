"""
Script para verificar que el sistema está correctamente configurado

Usage:
    python scripts/verify_setup.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import logging

from src.config import PROBES_DIR, FEATURES_DIR, ABLATIONS_DIR, LOGS_DIR

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')


def check_directories():
    """Verifica que los directorios necesarios existen"""
    logging.info("Checking directories...")
    
    required_dirs = [
        PROBES_DIR,
        FEATURES_DIR,
        ABLATIONS_DIR,
        LOGS_DIR
    ]
    
    all_exist = True
    for dir_path in required_dirs:
        if os.path.exists(dir_path):
            logging.info(f"  ✓ {dir_path}")
        else:
            logging.warning(f"  ✗ {dir_path} (will be created when needed)")
            all_exist = False
    
    return all_exist


def check_ud_treebanks():
    """Verifica que los treebanks de UD están disponibles"""
    logging.info("\nChecking UD treebanks...")
    
    from src.config import UD_BASE_FOLDER, LANGUAGES
    from src.data import get_ud_filepath
    
    if not os.path.exists(UD_BASE_FOLDER):
        logging.error(f"  ✗ UD base folder not found: {UD_BASE_FOLDER}")
        return False
    
    logging.info(f"  ✓ UD base folder: {UD_BASE_FOLDER}")
    
    all_found = True
    for language in LANGUAGES:
        train_file = get_ud_filepath(language, split='train')
        test_file = get_ud_filepath(language, split='test')
        
        if train_file:
            logging.info(f"  ✓ {language} (train)")
        else:
            logging.warning(f"  ✗ {language} (train file not found)")
            all_found = False
        
        if test_file:
            logging.info(f"  ✓ {language} (test)")
        else:
            logging.info(f"  ⚠ {language} (test file not found - optional)")
    
    return all_found


def check_dependencies():
    """Verifica que las dependencias necesarias están instaladas"""
    logging.info("\nChecking Python dependencies...")
    
    required_packages = [
        'torch',
        'numpy',
        'sklearn',
        'pyconll',
        'nnsight',
        'transformers',
        'huggingface_hub',
        'tqdm',
        'joblib'
    ]
    
    all_installed = True
    for package in required_packages:
        try:
            __import__(package)
            logging.info(f"  ✓ {package}")
        except ImportError:
            logging.error(f"  ✗ {package} (not installed)")
            all_installed = False
    
    return all_installed


def check_model_access():
    """Verifica que se puede acceder al modelo y SAE"""
    logging.info("\nChecking model and SAE access...")
    
    from src.config import MODEL_ID, SAE_ID
    
    try:
        from huggingface_hub import hf_hub_download
        from transformers import AutoTokenizer
        
        # Check model access
        logging.info(f"  Checking model: {MODEL_ID}")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        logging.info(f"  ✓ Model accessible")
        
        # Check SAE access
        logging.info(f"  Checking SAE: {SAE_ID}")
        sae_filename = "llama-3-8b-layer16.pt"
        sae_path = hf_hub_download(repo_id=SAE_ID, filename=sae_filename)
        logging.info(f"  ✓ SAE accessible")
        
        return True
        
    except Exception as e:
        logging.error(f"  ✗ Error accessing model/SAE: {str(e)}")
        return False


def check_gpu():
    """Verifica disponibilidad de GPU"""
    logging.info("\nChecking GPU availability...")
    
    import torch
    
    if torch.cuda.is_available():
        n_gpus = torch.cuda.device_count()
        logging.info(f"  ✓ CUDA available ({n_gpus} GPU(s))")
        
        for i in range(n_gpus):
            props = torch.cuda.get_device_properties(i)
            logging.info(f"    GPU {i}: {props.name} ({props.total_memory / 1e9:.1f} GB)")
        
        return True
    else:
        logging.warning("  ⚠ No CUDA GPU available (will use CPU - very slow)")
        return False


def test_data_loading():
    """Prueba que se pueden cargar datos"""
    logging.info("\nTesting data loading...")
    
    try:
        from src.data import get_ud_filepath, get_available_concepts, ProbingDataset
        
        # Try to load English-PUD
        train_file = get_ud_filepath("English-PUD", split='train')
        
        if not train_file:
            logging.warning("  ⚠ Could not load English-PUD for testing")
            return False
        
        # Get concepts
        concepts = get_available_concepts(train_file)
        logging.info(f"  ✓ Found {len(concepts)} concept types in English-PUD")
        
        # Try to create a dataset
        if concepts:
            concept_key = list(concepts.keys())[0]
            concept_value = list(concepts[concept_key])[0]
            
            dataset = ProbingDataset(train_file, concept_key, concept_value)
            logging.info(f"  ✓ Created dataset with {len(dataset)} samples for {concept_key}:{concept_value}")
            
            return True
        else:
            logging.warning("  ⚠ No concepts found")
            return False
            
    except Exception as e:
        logging.error(f"  ✗ Error testing data loading: {str(e)}")
        return False


def main():
    """Run all verification checks"""
    logging.info("=" * 60)
    logging.info("SAE Probing System - Setup Verification")
    logging.info("=" * 60)
    
    checks = [
        ("Directories", check_directories),
        ("UD Treebanks", check_ud_treebanks),
        ("Python Dependencies", check_dependencies),
        ("GPU", check_gpu),
        ("Model/SAE Access", check_model_access),
        ("Data Loading", test_data_loading),
    ]
    
    results = {}
    for name, check_func in checks:
        try:
            results[name] = check_func()
        except Exception as e:
            logging.error(f"\nError in {name} check: {str(e)}")
            results[name] = False
    
    # Summary
    logging.info("\n" + "=" * 60)
    logging.info("Summary")
    logging.info("=" * 60)
    
    all_passed = True
    for name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        logging.info(f"{name:25s}: {status}")
        if not passed:
            all_passed = False
    
    logging.info("=" * 60)
    
    if all_passed:
        logging.info("\n✓ All checks passed! System is ready to use.")
        logging.info("\nNext steps:")
        logging.info("  1. Train probes: python scripts/train_probes.py")
        logging.info("  2. Find features: python scripts/find_features.py")
        logging.info("  3. Run ablation: python scripts/run_ablation.py --help")
    else:
        logging.warning("\n⚠ Some checks failed. Please review the errors above.")
        logging.info("\nNote: Some failures (like test file availability) are non-critical.")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())

