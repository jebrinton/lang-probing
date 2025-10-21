# Lang Probing System - Implementation Summary

## Resumen Ejecutivo

Se ha implementado exitosamente un sistema completo para entrenar probes lineales sobre activaciones de capas MLP, identificar features correlacionadas con conceptos gramaticales, y realizar experimentos de ablación.

**Estado**: ✓ Implementación completa y actualizada  
**Fecha**: Octubre 2025  
**Ubicación**: `/projectnb/mcnet/jbrin/lang-probing/`

## Componentes Implementados

### 1. Módulos Core (src/)

#### ✓ config.py
- Configuración centralizada del sistema
- Parámetros del modelo (Llama-3.1-8B, SAE layer 16)
- Idiomas: English, Spanish, Turkish (multi-treebank support)
- Conceptos: Tense, Number, Gender
- Hiperparámetros de entrenamiento

#### ✓ utils.py
- `setup_model()`: Carga modelo, tokenizer y SAE
- `get_device_info()`: Detección automática de GPU/CPU
- Utilidades para JSON y directorios

#### ✓ data.py
- `get_all_treebank_files()`: Busca todos los archivos .conllu para un idioma
- `get_training_files()`: Obtiene archivos de entrenamiento (train + dev)
- `get_test_files()`: Obtiene archivos de test
- `get_available_concepts()`: Extrae conceptos disponibles de múltiples archivos
- `ProbingDataset`: Dataset con soporte multi-treebank y etiquetado automático
- `balance_dataset()`: Balanceo por undersampling
- `concept_filter()`: Filtrado de sentences por concepto
- `get_ud_filepath()`: DEPRECATED - Función legacy para compatibilidad

#### ✓ activations.py
- `extract_mlp_activations()`: Extracción batch de activaciones MLP desde cualquier capa
- `extract_sae_activations()`: DEPRECATED - Función legacy con warning
- Mean pooling weighted by attention mask
- Extracción desde `model.model.layers[layer_num].output[0]`

#### ✓ probe.py
- `train_probe()`: Entrena LogisticRegression con sklearn
- `evaluate_probe()`: Calcula accuracy
- `save_probe()` / `load_probe()`: Persistencia con joblib
- `get_probe_predictions()`: Predicciones, probabilidades y logits
- `get_probe_info()`: Metadatos del probe

#### ✓ features.py
- `find_top_correlating_features()`: Top k features por peso absoluto
- `find_top_positive_negative_features()`: Separación por signo
- `get_shared_features_across_languages()`: Features compartidas
- `analyze_feature_overlap()`: Análisis de overlap entre conjuntos
- Utilidades para extracción de índices y pesos

#### ✓ ablation.py
- `ablate_features()`: Ablación (set to 0) de features específicas
- `activate_features()`: Activación (incremento) de features
- `progressive_ablation()`: Ablación incremental
- `test_feature_necessity()`: Test estadístico de necesidad

### 2. Tests Unitarios (tests/)

#### ✓ test_data.py
- Tests de carga de archivos UD
- Tests de extracción de conceptos
- Tests de ProbingDataset
- Tests de balanceo

#### ✓ test_probe.py
- Tests de entrenamiento con datos sintéticos
- Tests de evaluación
- Tests de save/load (persistencia)
- Tests de predicciones y metadatos

#### ✓ test_features.py
- Tests de identificación de top features
- Tests de separación positivas/negativas
- Tests de análisis de overlap
- Tests de features compartidas

### 3. Scripts de Ejecución (scripts/)

#### ✓ verify_setup.py
Verificación completa del sistema:
- Directorios
- UD treebanks
- Dependencias Python
- Disponibilidad de GPU
- Acceso a modelo/SAE
- Test de carga de datos

#### ✓ check_structure.py
Verificación rápida de estructura sin dependencias:
- 24 checks de archivos y directorios
- No requiere dependencias externas

#### ✓ train_probes.py
Script principal para entrenamiento:
- Loop sobre idiomas y conceptos
- Soporte multi-treebank: concatena automáticamente todos los treebanks por idioma
- Entrenamiento: archivos train.conllu + dev.conllu
- Evaluación: archivos test.conllu
- Extracción automática de conceptos disponibles
- Balanceo de datasets
- Entrenamiento y evaluación
- Guardado automático de probes
- Logging detallado con información de treebanks utilizados
- Opciones: `--languages`, `--concepts`, `--min_samples`, `--overwrite`

#### ✓ find_features.py
Análisis de features correlacionadas:
- Procesamiento por concepto
- Identificación de top k features
- Cálculo de features compartidas
- Guardado en JSON
- Opciones: `--k`, `--concept_key`, `--concept_value`

#### ✓ run_ablation.py
Experimentos de ablación:
- Tres modos: simple, progressive, necessity
- Carga automática de probes y features
- Procesamiento de archivos de ejemplos
- Guardado de resultados detallados
- Opciones: `--language`, `--concept_key`, `--concept_value`, `--examples`, `--k`, `--mode`

### 4. Documentación

#### ✓ README.md
Documentación completa con:
- Estructura del proyecto
- Guía de configuración inicial
- Pipeline de uso paso a paso
- Formato de archivos de datos
- Troubleshooting

#### ✓ requirements.txt
Dependencias del proyecto:
- torch, numpy, scikit-learn
- transformers, nnsight, pyconll
- huggingface_hub, tqdm, joblib
- pytest

#### ✓ IMPLEMENTATION_SUMMARY.md (este archivo)
Resumen de implementación

### 5. Archivos de Ejemplo (examples/)

#### ✓ tense_past_english.txt
- 5 ejemplos positivos (past tense)
- 5 ejemplos negativos (present tense)

#### ✓ number_plural_english.txt
- 5 ejemplos positivos (plural)
- 5 ejemplos negativos (singular)

## Estructura de Directorios

```
/projectnb/mcnet/jbrin/lang-probing/
├── src/                           ✓ 8 módulos (config.py actualizado con rutas absolutas)
├── tests/                         ✓ 4 archivos de tests
├── scripts/                       ✓ 5 scripts ejecutables (actualizados con rutas absolutas)
├── examples/                      ✓ 2 archivos de ejemplo
├── outputs/                       ✓ Ruta absoluta: /projectnb/mcnet/jbrin/lang-probing/outputs
│   ├── probes/                   ✓ (para .joblib)
│   ├── features/                 ✓ (para .json)
│   └── ablations/                ✓ (para .json)
├── logs/                         ✓ Ruta absoluta: /projectnb/mcnet/jbrin/lang-probing/logs
├── README.md                     ✓ (actualizado)
├── requirements.txt              ✓
└── IMPLEMENTATION_SUMMARY.md     ✓ (actualizado)
```

**Total**: 24 archivos/directorios verificados ✓

## Pipeline Completo

### Paso 1: Entrenar Probes
```bash
python scripts/train_probes.py
```
- Input: Múltiples UD treebanks por idioma (.conllu)
- Concatena automáticamente: train.conllu + dev.conllu para entrenamiento
- Usa test.conllu para evaluación
- Output: Probes entrenados (.joblib)
- Formato: `{language}_{concept_key}_{concept_value}.joblib`

### Paso 2: Encontrar Features
```bash
python scripts/find_features.py --k 100
```
- Input: Probes (.joblib)
- Output: Features correlacionadas (.json)
- Archivos:
  - `{concept_key}_{concept_value}.json` (por idioma)
  - `{concept_key}_{concept_value}_shared.json` (compartidas)

### Paso 3: Ablación
```bash
python scripts/run_ablation.py \
    --language English \
    --concept_key Tense \
    --concept_value Past \
    --examples examples/tense_past_english.txt \
    --k 10 \
    --mode simple
```
- Input: Probes, features, ejemplos
- Output: Resultados de ablación (.json)

## Características Técnicas

### Modelo y SAE
- **Modelo**: Llama-3.1-8B (meta-llama/Llama-3.1-8B)
- **SAE**: jbrinkma/sae-llama-3-8b-layer16
- **Layer**: 16
- **Dictionary size**: 32,768 features

### Probing Methodology
- **Pooling**: Mean pooling weighted by attention mask
- **Probe**: LogisticRegression
  - `class_weight="balanced"`
  - `solver="liblinear"`
  - `C=0.1`
  - `max_iter=5000`
- **Feature selection**: Peso absoluto de coeficientes

### Data Processing
- **Source**: Universal Dependencies 2.16
- **Multi-treebank**: Concatena automáticamente todos los treebanks por idioma
- **Training data**: train.conllu + dev.conllu
- **Test data**: test.conllu
- **Balancing**: Undersampling de clase mayoritaria
- **Minimum samples**: 128 (configurable)
- **Batch size**: 16 (configurable)

## Soporte Multi-Treebank

### Funcionalidad Implementada
El sistema ahora soporta automáticamente múltiples treebanks por idioma:

- **Detección automática**: Busca todos los directorios `UD_{language}-*`
- **Concatenación inteligente**: 
  - Entrenamiento: Combina archivos `*train.conllu` + `*dev.conllu`
  - Evaluación: Usa archivos `*test.conllu`
- **Manejo de conceptos**: Usa la unión de todos los valores de conceptos disponibles
- **Logging detallado**: Muestra qué treebanks se están utilizando

### Ejemplo de Uso
Para English, el sistema automáticamente encuentra y usa:
- `UD_English-EWT/`, `UD_English-GUM/`, `UD_English-PUD/`, etc.
- Combina datos de todos los treebanks para mayor robustez

### Beneficios
- **Mayor cantidad de datos**: Más ejemplos para entrenamiento
- **Mejor generalización**: Diversidad de dominios y estilos
- **Robustez**: Menos dependencia de un solo treebank

## Extensiones Futuras

### Attribution Patching (Planificado)
Para análisis causal de features:
1. Crear `src/attribution.py` adaptando código de `multilingual-features`
2. Implementar gradient-based attribution en `features.py`
3. Comparar con método actual basado en pesos

### Escalabilidad
- Añadir más idiomas (fácil: editar `LANGUAGES` en config.py)
- Añadir más conceptos (automático: detectados en treebanks)
- El soporte multi-treebank ya está implementado

## Verificación de Implementación

✓ Estructura completa: 24/24 checks passed  
✓ Todos los módulos core implementados  
✓ Tests unitarios completos  
✓ Scripts ejecutables listos  
✓ Documentación comprehensiva  
✓ Archivos de ejemplo proporcionados  

## Próximos Pasos para el Usuario

1. **Instalar dependencias**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Verificar setup completo**:
   ```bash
   python scripts/verify_setup.py
   ```

3. **Entrenar probes** (ejemplo pequeño):
   ```bash
   python scripts/train_probes.py \
       --languages English \
       --concepts Tense
   ```

4. **Encontrar features**:
   ```bash
   python scripts/find_features.py --k 50
   ```

5. **Ejecutar ablación**:
   ```bash
   python scripts/run_ablation.py \
       --language English \
       --concept_key Tense \
       --concept_value Past \
       --examples examples/tense_past_english.txt \
       --k 10
   ```

## Notas Importantes

- El sistema requiere GPU con suficiente memoria (~24GB recomendado para Llama-3-8B)
- Los UD treebanks deben estar en `/projectnb/mcnet/jbrin/.cache/ud/ud-treebanks-v2.16/`
- Algunos conceptos pueden no estar disponibles en todos los idiomas (el sistema los salta automáticamente)
- Turkish no tiene género, como se esperaba
- El sistema es totalmente extensible y modular

## Contacto y Soporte

Para problemas o preguntas, revisar:
1. README.md (troubleshooting section)
2. Logs en `logs/`
3. Ejecutar `python scripts/verify_setup.py` para diagnóstico

---

**Implementación completada exitosamente** ✓

