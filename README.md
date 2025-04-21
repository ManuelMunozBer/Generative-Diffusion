# Generative‑Diffusion

![Python](https://img.shields.io/badge/python-3.8%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

Toolkit modular para **modelos de difusión generativos** (imágenes color)
con soporte para:

* **Procesos** VE‑SDE, VP‑SDE, SubVP‑SDE  
* **Samplers** Euler‑Maruyama, Predictor–Corrector, Probability‑Flow ODE,
  Exponential‑Integrator  
* **Noise schedules** lineal, coseno, constante  
* **Control** de generación (class‑conditional, imputación)  
* **Métricas** FID, IS, BPD

## Instalación rápida

```bash
pip install generative-diffusion           # desde PyPI
# ó desde el repo
pip install -e .[dev]
```

## Ejemplo mínimo

```python
from generative_diffusion import ModelFactory
from generative_diffusion.score_networks.unet_score_network import UNetScore

model = ModelFactory.create(
    score_model_class = UNetScore,
    is_conditional    = False,
    sde_name          = "ve_sde",
    sampler_name      = "euler_maruyama",
)

samples = model.generate(n_samples=4, n_steps=200)
print(samples.shape)   # (4, 3, H, W)
```

## Estructura de carpetas

```
generative_diffusion/   <-- código del paquete
demo_notebooks/         <-- ejemplos de uso
checkpoints/            <-- pesos entrenados opcionales
pyproject.toml
README.md
```

## Desarrollo

* **Tests**: `pytest -q`
* **Formateo**: `black .`
* **Linter**: `ruff check . --fix`
* **CI**: ver `.github/workflows/ci.yml`

---

## Licencia

MIT