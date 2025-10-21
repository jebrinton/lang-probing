# Lang Probing System

Sistema para entrenar probes lineales sobre activaciones de capas MLP y analizar features correlacionadas con conceptos gramaticales.

## Estructura del Proyecto

```
lang-probing/
├── src/                    # Código fuente
│   ├── config.py          # Configuración global
│   ├── data.py            # Carga de datos desde UD treebanks
│   ├── activations.py     # Extracción de activaciones MLP
│   ├── probe.py           # Entrenamiento de probes
│   ├── features.py        # Análisis de features correlacionadas
│   ├── ablation.py        # Experimentos de ablación
│   └── utils.py           # Utilidades generales
├── tests/                 # Tests unitarios
├── scripts/               # Scripts de ejecución
│   ├── verify_setup.py    # Verificar configuración
│   ├── train_probes.py    # Entrenar probes
│   ├── find_features.py   # Encontrar top features
│   └── run_ablation.py    # Experimentos de ablación
├── outputs/               # Resultados
│   ├── probes/           # Probes entrenados (.joblib)
│   ├── features/         # Top features (.json)
│   └── ablations/        # Resultados de ablaciones (.json)
└── logs/                 # Logs de ejecución
```

## Configuración Inicial

### 1. Verificar el Setup

```bash
cd /projectnb/mcnet/jbrin/lang-probing
python scripts/verify_setup.py
```

Este script verifica:
- Directorios necesarios
- Disponibilidad de UD treebanks
- Dependencias de Python
- Acceso a GPU
- Acceso al modelo y SAE
- Carga de datos

### 2. Configuración

Edita `src/config.py` para ajustar:
- `LANGUAGES`: Idiomas a procesar (nombres simples: English, Spanish, Turkish)
- `CONCEPTS`: Conceptos gramaticales a analizar
- `BATCH_SIZE`, `SEED`, etc.

## Pipeline de Uso

### Paso 1: Entrenar Probes

Entrena probes lineales para todos los conceptos en todos los idiomas:

```bash
python scripts/train_probes.py
```

Opciones:
- `--languages LANG1 LANG2`: Procesar idiomas específicos
- `--concepts CONCEPT1 CONCEPT2`: Procesar conceptos específicos
- `--min_samples N`: Mínimo de muestras requeridas (default: 128)
- `--overwrite`: Sobreescribir probes existentes

Ejemplo:
```bash
python scripts/train_probes.py --languages English --concepts Tense Number
```

Los probes se guardan en `outputs/probes/` con el formato:
`{language}_{concept_key}_{concept_value}.joblib`

### Paso 2: Encontrar Top Features

Analiza los probes entrenados para identificar las features SAE más correlacionadas:

```bash
python scripts/find_features.py
```

Opciones:
- `--k N`: Número de top features (default: 100)
- `--concept_key KEY --concept_value VALUE`: Procesar un concepto específico

Ejemplo:
```bash
python scripts/find_features.py --k 50 --concept_key Tense --concept_value Past
```

Los resultados se guardan en `outputs/features/`:
- `{concept_key}_{concept_value}.json`: Features por idioma
- `{concept_key}_{concept_value}_shared.json`: Features compartidas entre idiomas

### Paso 3: Experimentos de Ablación

Ablata features específicas y mide el efecto en la clasificación:

```bash
python scripts/run_ablation.py \
    --language English \
    --concept_key Tense \
    --concept_value Past \
    --examples examples/tense_past_english.txt \
    --k 10 \
    --mode simple
```

Opciones:
- `--language LANG`: Idioma a probar
- `--concept_key KEY`: Clave del concepto
- `--concept_value VALUE`: Valor del concepto
- `--examples FILE`: Archivo con ejemplos (ver formato abajo)
- `--k N`: Número de top features a ablatar
- `--mode MODE`: Modo de ablación:
  - `simple`: Ablación simple en todos los ejemplos
  - `progressive`: Ablación progresiva (añade features una a una)
  - `necessity`: Test de necesidad de features

#### Formato del Archivo de Ejemplos

```
positive
The boy walked to school yesterday.
She ate breakfast this morning.

negative
The boy walks to school every day.
She eats breakfast every morning.
```

## Ejecutar Tests

```bash
cd /projectnb/mcnet/jbrin/sae_probing

# Ejecutar todos los tests
python -m pytest tests/

# Ejecutar tests específicos
python -m pytest tests/test_data.py
python -m pytest tests/test_probe.py
python -m pytest tests/test_features.py

# Ejecutar con unittest
python tests/test_data.py
```

## Estructura de Datos

### Probes

Los probes son modelos `LogisticRegression` de sklearn guardados con joblib.

### Features

Archivo JSON con estructura:
```json
{
  "English": [
    [12345, 0.8542],  // [feature_index, weight]
    [67890, 0.7321],
    ...
  ],
  "Spanish": [
    [12345, 0.7892],
    [11223, 0.6543],
    ...
  ]
}
```

### Features Compartidas

```json
{
  "12345": {
    "count": 3,
    "languages": ["English", "Spanish", "Turkish"],
    "avg_weight": 0.8123,
    "std_weight": 0.0432
  }
}
```

### Resultados de Ablación

```json
{
  "language": "English",
  "concept_key": "Tense",
  "concept_value": "Past",
  "k": 10,
  "experiments": [
    {
      "sentence": "The boy walked to school.",
      "label": "positive",
      "original_logit": 2.345,
      "ablated_logit": -0.123,
      "logit_change": -2.468,
      "original_prob": 0.912,
      "ablated_prob": 0.469,
      "prob_change": -0.443
    }
  ]
}
```

## Idiomas y Conceptos por Defecto

### Idiomas
- English (multi-treebank: EWT, GUM, PUD, etc.)
- Spanish (multi-treebank: AnCora, GSD, PUD, etc.)
- Turkish (multi-treebank: BOUN, GB, IMST, PUD, etc.)

### Conceptos
- Tense (e.g., Past, Present, Future)
- Number (e.g., Sing, Plur)
- Gender (e.g., Masc, Fem, Neut)

## Notas Técnicas

- **Modelo**: Llama-3.1-8B
- **SAE**: jbringma/sae-llama-3-8b-layer16 (32768 features)
- **MLP Activaciones**: Extraídas de `model.model.layers[layer_num].output[0]`
- **Pooling**: Mean pooling de activaciones sobre tokens (weighted by attention mask)
- **Probe**: LogisticRegression con class_weight="balanced", C=0.1
- **Correlación**: Basada en pesos absolutos del probe
- **Rutas**: Configuradas en `src/config.py` con base `/projectnb/mcnet/jbrin/lang-probing`

## Extensiones Futuras

Para implementar attribution patching (análisis causal de features):

1. Crear `src/attribution.py` adaptando código de `multilingual-features`
2. Añadir función en `features.py` para gradient-based attribution
3. Comparar resultados con correlación basada en pesos

## Troubleshooting

### Error: UD treebank not found
Verifica que `/projectnb/mcnet/jbrin/.cache/ud/ud-treebanks-v2.16/` existe y contiene los treebanks necesarios.

### Error: CUDA out of memory
Reduce `BATCH_SIZE` en `src/config.py`.

### Error: Not enough samples
Algunos conceptos pueden no tener suficientes ejemplos. El sistema requiere al menos 128 ejemplos balanceados por defecto (ajustable con `--min_samples`).

### Error: Model/SAE access
Verifica que tienes acceso a HuggingFace y los modelos necesarios. Puede requerir configurar `HF_TOKEN`.

