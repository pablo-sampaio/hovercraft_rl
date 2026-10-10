# `render` — visualizador (pygame)

Desenha a pista, o veículo, o rastro e um HUD. É usado por `env.render()` e pelo modo jogo.

## Visões (tecla **V**)

| Nome | Descrição |
|---|---|
| `"global"` — **Global (top-down)** | mapa fixo, com a pista inteira na tela e norte para cima |
| `"ego"` — **Egocêntrica (heading-up / chase cam)** | câmera presa ao veículo e girando com ele, frente sempre para cima; inclui um minimapa |

## Elementos

- **Pista:** asfalto, zebras vermelho/branco nas bordas (limite `half_width`), linha base tracejada e largada quadriculada.
- **Hovercraft:**
  - saia preta e convés laranja;
  - cabine azul e nariz claro, que indicam a frente;
  - dois ventiladores na popa;
  - jato de ar proporcional a $\tau_u$;
  - lemes que defletem com $\tau_r$.
  - Na visão global, o desenho é ampliado até um tamanho mínimo (`min_pixels`), para a orientação ficar sempre legível.
- **Seta azul:** velocidade no mundo, projetada 0.5 s à frente. Quando há derrapagem, ela diverge do nariz.
- **Debug (tecla G):** ponto de referência em amarelo e pontos de lookahead da observação em lilás.
- **HUD (tecla H):** tempo, % da volta, $|V|, u, v, r, \psi$, $e_{lat}$, voltas (última e melhor), recompensa e barras de $\tau_u, \tau_r$.
- **Banner "FORA DA PISTA"** quando `info["off_track"]`.

Zoom: teclas **+/−**.

## API

```python
from hovercraft_rl.render import make_renderer
r = make_renderer(physics, track, render_mode="human", view="ego", show_debug=True)
r.render(state_dict, action, info)   # "human": desenha na janela; "rgb_array": retorna (H, W, 3) uint8
r.reset(); r.close()
```

| kwarg | Padrão | Descrição |
|---|---|---|
| `render_mode` | `"human"` | `"human"` ou `"rgb_array"` (sem janela) |
| `view` | `"global"` | visão inicial |
| `width`, `height` | 960, 720 | tamanho da tela (px) |
| `fps` | 50 | limite de quadros no modo humano (tempo real com $h = 0.02$ s) |
| `show_debug`, `show_hud` | False, True | overlays |
| `process_events` | True | consome os eventos da janela; o jogo usa `False` e trata os eventos ele mesmo |

## Outro modelo físico

1. Herde `BaseTrackRenderer` e implemente `draw_vehicle` e `hud_lines`. Se preciso, implemente também `vehicle_pose` e `draw_hud_extra`.
2. Registre em `RENDERERS` (`render/__init__.py`).

A câmera, a pista e o HUD básico são reaproveitados.
