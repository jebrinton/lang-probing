# Quick Start Guide

## Configuración Inicial (Una sola vez)

### 1. Verificar estructura
```bash
cd /projectnb/mcnet/jbrin/lang-probing
python scripts/check_structure.py
```

### 2. Instalar dependencias
```bash
pip install -r requirements.txt
```

### 3. Verificar setup completo
```bash
python scripts/verify_setup.py
```

## Uso Básico

### Entrenar un probe para un concepto específico

```bash
# Entrenar probe para Tense:Past en inglés
python scripts/train_probes.py \
    --languages English \
    --concepts Tense
```

### Entrenar probes para todos los conceptos configurados

```bash
# Procesa English, Spanish, Turkish (multi-treebank)
# para Tense, Number, Gender
python scripts/train_probes.py
```

### Encontrar top features correlacionadas

```bash
# Top 100 features para todos los conceptos
python scripts/find_features.py --k 100

# Top 50 features para un concepto específico
python scripts/find_features.py \
    --k 50 \
    --concept_key Tense \
    --concept_value Past
```

### Ejecutar experimento de ablación simple

```bash
python scripts/run_ablation.py \
    --language English \
    --concept_key Tense \
    --concept_value Past \
    --examples examples/tense_past_english.txt \
    --k 10 \
    --mode simple
```

### Ejecutar ablación progresiva

```bash
python scripts/run_ablation.py \
    --language English \
    --concept_key Number \
    --concept_value Plur \
    --examples examples/number_plural_english.txt \
    --k 20 \
    --mode progressive
```

### Test de necesidad de features

```bash
python scripts/run_ablation.py \
    --language English \
    --concept_key Tense \
    --concept_value Past \
    --examples examples/tense_past_english.txt \
    --k 10 \
    --mode necessity
```

## Crear tus propios ejemplos

Crea un archivo de texto con el formato:

```
positive
Sentence with the concept present.
Another sentence with the concept.

negative
Sentence without the concept.
Another sentence without the concept.
```

Guárdalo en `examples/` y úsalo con `--examples`.

## Añadir más idiomas

1. Edita `src/config.py`:
```python
LANGUAGES = ["English", "Spanish", "Turkish", "French"]
```

2. Asegúrate de que los treebanks UD existen en:
```
/projectnb/mcnet/jbrin/.cache/ud/ud-treebanks-v2.16/UD_French-*/
```
El sistema automáticamente encontrará y usará todos los treebanks disponibles.

3. Ejecuta el pipeline normalmente

## Añadir más conceptos

Los conceptos se detectan automáticamente desde los treebanks UD. Simplemente añádelos a la configuración:

```python
CONCEPTS = ["Tense", "Number", "Gender", "Case", "Mood"]
```

## Ver resultados

### Probes entrenados
```bash
ls /projectnb/mcnet/jbrin/lang-probing/outputs/probes/
# English_Tense_Past.joblib
# Spanish_Tense_Past.joblib
# Turkish_Tense_Past.joblib
# etc.
```

### Features identificadas
```bash
ls /projectnb/mcnet/jbrin/lang-probing/outputs/features/
# Tense_Past.json
# Tense_Past_shared.json
# Number_Plur.json
# etc.
```

### Resultados de ablación
```bash
ls /projectnb/mcnet/jbrin/lang-probing/outputs/ablations/
# English_Tense_Past_simple_k10.json
# Spanish_Number_Plur_progressive_k20.json
# etc.
```

## Tips

- **GPU recomendada**: El modelo requiere ~24GB de VRAM
- **Batch size**: Reduce `BATCH_SIZE` en `src/config.py` si tienes problemas de memoria
- **Mínimo de muestras**: Ajusta `--min_samples` si un concepto tiene pocas instancias
- **Logs**: Revisa `logs/` si algo falla

## Comandos útiles

```bash
# Ver probes disponibles
ls -lh outputs/probes/

# Contar cuántos probes se entrenaron
ls outputs/probes/*.joblib | wc -l

# Ver features compartidas para Tense:Past
cat outputs/features/Tense_Past_shared.json | head -20

# Ejecutar tests
python -m pytest tests/

# Ver logs recientes
tail -f logs/train_probes.log
```

## Troubleshooting Rápido

### Error: "No training files found"
Los treebanks UD no existen para ese idioma. Verifica:
```bash
ls /projectnb/mcnet/jbrin/.cache/ud/ud-treebanks-v2.16/UD_English-*/
```

### Error: "Not enough samples"
El concepto tiene pocas instancias. Opciones:
- Reduce `--min_samples 64`
- Prueba otro concepto
- Usa otro treebank

### Error: "CUDA out of memory"
Reduce batch size en `src/config.py`:
```python
BATCH_SIZE = 8  # o 4
```

### Error: "Probe already exists"
Usa `--overwrite` para reentrenar:
```bash
python scripts/train_probes.py --overwrite
```

