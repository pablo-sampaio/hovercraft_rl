# `track` — pistas

Uma pista é uma **linha base** fechada (polilinha) com uma **meia-largura** fixa `half_width`. O hovercraft está dentro da pista enquanto $|e_{lat}| \le$ `half_width`.

## Arquivo (`data/<nome>.json`)

```json
{"name": "oval", "half_width": 3.0, "smooth_iterations": 0, "points": [[x0, y0], [x1, y1], ...]}
```

- `points` são os vértices do polígono. O último vértice liga ao primeiro, então não repita o ponto inicial.
- O sentido da pista é a ordem dos pontos, e a largada fica em `points[0]`.
- `smooth_iterations` aplica a suavização de Chaikin, útil para polígonos com poucos vértices.

Pistas incluídas (geradas por `python -m hovercraft_rl.track.generate`, espaçamento ≈ 1 m, todas no sentido anti-horário):

| Nome | Comprimento | Raio mín. | Características |
|---|---|---|---|
| `circle` | 188 m | 30 m | curva constante |
| `oval` | 246 m | 20 m | 2 retas de 60 m + 2 curvas de 180° |
| `rounded_rect` | 263 m | 10 m | 4 curvas de 90° fechadas |
| `kidney` | 224 m | 7 m | curvas côncavas e convexas |
| `gp` | 380 m | 3.9 m | circuito com "S" e grampos |

Para criar uma pista nova, basta salvar um JSON neste formato, ou usar `Track(points, half_width).save(path)`.

## API principal (`Track`)

| Método | Retorno |
|---|---|
| `Track.load(nome_ou_caminho)` | pista |
| `project(x, y, hint_s=None)` | `Projection(x_ref, y_ref, s, lateral_error, heading, segment)` |
| `point_at(s)`, `heading_at(s)` | ponto e direção da linha base no arco `s` (escalar ou vetor) |
| `progress_delta(s_prev, s_new)` | avanço com sinal, tratando a passagem pela largada |
| `boundaries()` | bordas esquerda e direita (para desenho) |
| `length`, `half_width`, `min_curvature_radius()` | dados geométricos |

### Ponto de referência

- O ponto de referência é a **projeção ortogonal sobre o segmento de reta** mais próximo. Portanto, em geral não coincide com um vértice.
- `lateral_error` é positivo à esquerda do sentido da pista.
- Com `hint_s` (o `s` do passo anterior), a busca considera só segmentos a até `search_window` metros de arco (padrão: 15 m).
  - Isso evita que, em grampos ou trechos próximos, a referência "salte" para outro trecho da pista.
  - Também evita que o progresso conte atalhos.
- Uma pista só é bem definida se o raio de curvatura mínimo for maior que `half_width`.
