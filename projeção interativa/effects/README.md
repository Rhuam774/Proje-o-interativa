# Sistema de Efeitos - Projeção Interativa

## Estrutura de Pastas

```
effects/
├── simple/          # Efeitos visuais simples (plexus, grid, etc)
├── games/           # Jogos interativos (insetos, etc)
├── immersive/       # Visualizações imersivas (vetores, etc)
└── README.md        # Este arquivo
```

## Como Criar um Novo Efeito

### 1. Escolha a Categoria

- **simple/**: Efeitos visuais que não requerem interação complexa
- **games/**: Jogos interativos com pontuação, objetivos, etc
- **immersive/**: Visualizações avançadas que usam processamento de imagem

### 2. Crie o Arquivo do Efeito

Crie um arquivo Python na pasta apropriada. O nome do arquivo deve ser descritivo e usar snake_case.

Exemplo: `effects/simple/my_new_effect.py`

### 3. Implemente a Classe do Efeito

```python
import cv2
import numpy as np
import math
import random

class MyNewEffect:
    def __init__(self):
        """Inicializa o efeito"""
        self.name = "Meu Novo Efeito"
        self.category = "simple"  # simple, games, ou immersive
        self.description = "Descrição curta do efeito"
        
        # Variáveis do efeito (opcional)
        self.some_var = 0
    
    def apply(self, frame, trail_canvas, cam_trail_canvas, H, yolo_enabled, trackers, trackers_lock):
        """
        Aplica o efeito ao frame atual
        
        Parâmetros:
        - frame: Frame atual da câmera (numpy array)
        - trail_canvas: Canvas do projetor para desenhar (numpy array)
        - cam_trail_canvas: Canvas do monitor para desenhar (numpy array)
        - H: Matriz de homografia para transformação de perspectiva
        - yolo_enabled: Boolean indicando se visão IA está ativa
        - trackers: Dicionário de rastreadores ativos
        - trackers_lock: Lock para acesso seguro ao dicionário de rastreadores
        
        Retorna: None (desenha diretamente nos canvases)
        """
        # Exemplo: desenhar algo no canvas do projetor
        height, width = trail_canvas.shape[:2]
        
        # Seu código de efeito aqui
        # ...
        
        pass
    
    def get_config_ui(self, parent_frame, save_callback):
        """
        Retorna widgets de configuração do efeito (opcional)
        
        Parâmetros:
        - parent_frame: Frame tkinter onde adicionar os widgets
        - save_callback: Função para chamar quando configuração mudar
        
        Retorna: Lista de widgets tkinter (ou None se não houver configuração)
        """
        # Exemplo: criar sliders para configurar o efeito
        import tkinter as tk
        from tkinter import ttk
        
        widgets = []
        
        # Seu código de UI aqui
        # ...
        
        return widgets
    
    def cleanup(self):
        """
        Limpa recursos do efeito (opcional)
        Chamado quando o efeito é desativado
        """
        # Limpeza de recursos aqui
        pass
```

### 4. Importações Disponíveis

Você pode usar:
- `cv2` (OpenCV)
- `numpy` (como np)
- `math`
- `random`
- `threading` (se necessário)

### 5. Acesso aos Rastreadores

Para acessar os rastreadores ativos:

```python
with trackers_lock:
    for tid, tdata in trackers.items():
        history = tdata.get("history", [])
        # Processar histórico do rastreador
        for cx, cy, w, h in history:
            # Fazer algo com as coordenadas
            pass
```

### 6. Transformação de Coordenadas

Para transformar coordenadas da câmera para o projetor:

```python
if H is not None:
    pt_cam = np.float32([[[x, y]]])
    pt_proj = cv2.perspectiveTransform(pt_cam, H)
    x_proj, y_proj = int(pt_proj[0][0][0]), int(pt_proj[0][0][1])
else:
    x_proj, y_proj = x, y
```

### 7. Cores HSV Dinâmicas

Para criar cores baseadas em ângulo/direção:

```python
angle = math.atan2(dy, dx)
hue = int((angle + math.pi) / (2 * math.pi) * 180)
color_hsv = np.uint8([[[hue, 255, 255]]])
color_bgr = cv2.cvtColor(color_hsv, cv2.COLOR_HSV2BGR)[0][0]
```

## Convenções

- Use snake_case para nomes de arquivos e variáveis
- Use PascalCase para nomes de classes
- Adicione docstrings explicando o propósito de cada método
- Mantenha o código limpo e comentado
- Teste o efeito antes de adicionar ao sistema

## Exemplos

Veja os efeitos existentes em:
- `effects/simple/plexus.py` - Exemplo de efeito visual simples
- `effects/games/insects.py` - Exemplo de jogo interativo
- `effects/immersive/vectors.py` - Exemplo de visualização imersiva

## Suporte

Para dúvidas ou problemas, consulte a documentação do projeto ou entre em contato com a equipe de desenvolvimento.
