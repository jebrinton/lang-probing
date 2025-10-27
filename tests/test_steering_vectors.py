#!/usr/bin/env python3
"""
Script de prueba para validar la generación de steering vectors
con un subconjunto pequeño (1 idioma, 1 concepto, pocas capas)
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import logging
from src.config import STEERING_VECTORS_DIR, LOGS_DIR, MAX_SAMPLES_FOR_STEERING
from src.utils import ensure_dir, setup_logging



def test_steering_vectors():
    """Prueba la generación de steering vectors con un subconjunto pequeño"""
    
    logging.info("="*60)
    logging.info("PRUEBA DE NOVA DE STEERING VECTORS")
    logging.info("="*60)
    
    # Importar el script principal
    from scripts.generate_steering_vectors import main
    
    # Crear directorio de salida para la prueba
    test_output_dir = os.path.join(STEERING_VECTORS_DIR, "nova")
    ensure_dir(test_output_dir)
    
    # Simular argumentos para la prueba
    class TestArgs:
        def __init__(self):
            self.languages = ["English", "Spanish", "Turkish", "Arabic", "German", "Chinese", "French", "Japanese"]  # Solo un idioma
            self.concepts = ["Tense", "Number"]     # Solo un concepto
            self.layers = "0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31"            # Solo 1 capa para prueba rápida
            self.output_dir = test_output_dir
            self.batch_size = 16           # Batch muy pequeño para prueba
            self.max_samples = 4000     # Límite pequeño para prueba
    
    args = TestArgs()
    
    logging.info(f"Configuración de prueba:")
    logging.info(f"  Idiomas: {args.languages}")
    logging.info(f"  Conceptos: {args.concepts}")
    logging.info(f"  Capas: {args.layers}")
    logging.info(f"  Directorio de salida: {args.output_dir}")
    logging.info(f"  Batch size: {args.batch_size}")
    logging.info(f"  Max sentences: {args.max_samples}")
    
    try:
        # Ejecutar el script principal
        main(args)
        logging.info("✓ Prueba completada exitosamente!")
        
        # Verificar que se generaron archivos
        vectors_dir = os.path.join(args.output_dir, "vectors")
        if os.path.exists(vectors_dir):
            vector_files = [f for f in os.listdir(vectors_dir) if f.endswith('.pkl')]
            logging.info(f"✓ Se generaron {len(vector_files)} archivos de steering vectors")
            
            # Mostrar algunos archivos generados
            for i, file in enumerate(vector_files[:3]):  # Mostrar solo los primeros 3
                logging.info(f"  - {file}")
            if len(vector_files) > 3:
                logging.info(f"  ... y {len(vector_files) - 3} más")
        else:
            logging.warning("✗ No se encontró el directorio de vectores")
        
        # Verificar archivo de estadísticas
        stats_file = os.path.join(args.output_dir, "steering_stats.json")
        if os.path.exists(stats_file):
            logging.info(f"✓ Archivo de estadísticas generado: {stats_file}")
        else:
            logging.warning("✗ No se encontró el archivo de estadísticas")
            
    except Exception as e:
        logging.error(f"✗ Error durante la prueba: {str(e)}")
        import traceback
        logging.error(traceback.format_exc())
        return False
    
    logging.info("="*60)
    logging.info("PRUEBA COMPLETADA")
    logging.info("="*60)
    return True


if __name__ == "__main__":
    # Setup logging para la prueba
    setup_logging(LOGS_DIR, 'test_steering_vectors.log')
    success = test_steering_vectors()
    if success:
        print("\n✓ Prueba exitosa! El sistema de steering vectors está funcionando correctamente.")
        print(f"Revisa los logs en: {os.path.join(LOGS_DIR, 'test_steering_vectors.log')}")
        print(f"Archivos generados en: {os.path.join(STEERING_VECTORS_DIR, 'nova')}")
    else:
        print("\n✗ La prueba falló. Revisa los logs para más detalles.")
        sys.exit(1)
