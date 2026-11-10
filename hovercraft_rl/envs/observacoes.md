# Guia sobre os Tipos de Observações

As opções `world_frame` e `hovercraft_frame` descrevem os mesmos elementos do
ambiente em eixos diferentes. Foram criadas para estudar o efeito do
referencial em experimentos.

(Ps.: as opções anteriores `reference_error` e `body_frame` continuam disponíveis, por enquanto, e não foram alteradas).

## 1. Distinção Essencial

Seja **H** a posição do hovercraft e **P** a projeção de H na linha central da
pista, escolhida pelo ambiente. As duas opções usam P como origem, mas adotam
eixos com orientações diferentes:

- `world_frame`: move a origem para P, mantendo a orientação dos eixos X e Y
  do mundo.
- `hovercraft_frame`: eixo X aponta para a frente do veículo e eixo Y para a
  esquerda; move a origem para P e gira os eixos junto com o veículo.


## 2. Comparativo Geral das Observações

Os valores são colocados na seguinte ordem:

| Grupo | `world_frame` | `hovercraft_frame` |
|---|---|---|
| Posição | Coordenadas de H relativas a P, em metros, nos eixos do mundo | As mesmas coordenadas, expressas nos eixos do veículo |
| Erro lateral (opcional, incluído por padrão) | Distância com sinal à linha central, dividida pela meia-largura | O mesmo valor |
| Direção do hovercraft | `(cos ψ, sin ψ)` | Omitida por padrão; opcionalmente `(1, 0)` |
| Direção local da pista | `(cos θ, sin θ)` | `(cos(θ − ψ), sin(θ − ψ))` |
| Velocidades | `u`, `v`, `r` | Os mesmos valores |
| Pontos à frente | Coordenadas de cada ponto Li relativas a P, nos eixos do mundo | As mesmas coordenadas, expressas nos eixos do veículo |
| Direções entre pontos | Opcionalmente, pares `(cos α, sin α)` | Opcionalmente, `(cos(α − ψ), sin(α − ψ))` |

Os valores de posição e dos pontos à frente são coordenadas relativas à origem
P, não coordenadas absolutas do mundo. Os pares de cosseno e seno, por sua vez,
representam direções. Cada grupo será explicado nas seções a seguir.


## 3. Posição

Em `world_frame`, a origem é movida para P e os eixos continuam alinhados aos
do mundo. Por isso, as coordenadas observadas são
`(x_H - x_P, y_H - y_P)`: a posição de H medida a partir de P.

Em `hovercraft_frame`, além de mover a origem para P, os eixos são girados
junto com o veículo. Primeiro se obtêm as coordenadas de H relativas a P nos
eixos do mundo, `(x, y)`. Em seguida, elas são expressas nos eixos do veículo
usando `ψ`, o ângulo do veículo em relação ao eixo X do mundo:

```text
coordenada à frente   =  cos(ψ) x + sin(ψ) y
coordenada à esquerda = -sin(ψ) x + cos(ψ) y
```

## 4. Erro Lateral

O erro lateral é positivo à esquerda do **sentido da pista**, não
necessariamente à esquerda do veículo. Esse erro é normalizado pela meia-largura da pista. Assim, valores próximos de ±1 indicam que o veículo está próximo do limite permitido. 

Para omitir esse sinal da observação, configure `include_lateral_error=False`. 

## 5. Direção do Hovercraft

Note que `world_frame` fornece a orientação **absoluta** do veículo. Por padrão,
`hovercraft_frame` não oferece informação análoga, porque a orientação do veículo é fixa, 
coincidindo sempre com o novo eixo x calculado neste tipo de observação. 

Assim, os dois vetores de observação diferem em tamanho por dois valores apenas, por padrão. 

Para deixar os formatos comparáveis, com o mesmo tamanho, habilite `include_hovercraft_heading` na opção relativa. Ela adiciona constantes que representam a direção do hovercraft (alinhada ao eixo x).

## 6. Direção local da pista

A direção local é a direção do segmento da linha central associado à
projeção P. Como a linha central da pista é representada por segmentos de reta, usa-se a própria direção do segmento onde P for projetado (não é feito nenhum cálculo mais avançado de
a tangente de uma curva suave). Se esse segmento vai do ponto A ao ponto B, calculam-se
`dx = x_B - x_A` e `dy = y_B - y_A`; então:

```text
θ = atan2(dy, dx)
(cos θ, sin θ) = (dx / L, dy / L), onde L = √(dx² + dy²)
```

Assim, a observação guarda duas componentes que indicam a orientação, não as
coordenadas de um ponto[^nota1]. Em `hovercraft_frame`, a mesma direção é expressa
relativamente à direção do veículo, com ângulo `θ − ψ`. 

[^nota1]: Isso tem alguma semelhança com o cálculo
das direções entre pontos à frente (seção 9). Em ambos os casos, normaliza-se uma
diferença de coordenadas. A diferença é que aqui se usa o segmento da pista
associado a P; na direção entre entre pontos, usa-se a corda que liga dois pontos à frente.

## 7. Velocidades

São três valores do estado do veículo, fornecidos sem alteração pelas duas
observações:
- `u`: velocidade longitudinal (para a frente), em m/s;
- `v`: velocidade lateral do veículo, em m/s;
- `r`: velocidade angular, em rad/s.


## 8. Pontos à frente

Parte importante da observação são informações sobre "pontos à frente" da trajetória, que devem ajudar a guiar o hovercraft. Sem essas informações, o hovercraft não consegue mover-se de maneira preparatória para o trajeto à frente.

Os pontos são calculados sobre a linha central, a distâncias de arco a partir de P.
Por padrão, são criados 5 pontos a 5, 10, 15, 20 e 25 metros à frente da posição do veículo.
Mas a quantidade e as distâncias dos pontos são configuráveis.

As coordenadas fornecidas são de cada ponto Li relativas a P, e não relativas
diretamente ao veículo H. 

Há duas formas de configurar os pontos:

```python
# Três pontos, espaçados a cada 4 m (a 4, 8 e 12 m de P):
{"n_lookahead": 3, "lookahead_spacing": 4.0}

# Três pontos, a 2, 7 e 15 m de P:
{"lookahead_distances": [2, 7, 15]}
```

Não combine a lista de pontos com os parâmetros de quantidade ou de espaçamento uniforme.


## 9. Direções entre pontos

Com `include_directions=True`, a observação também contém a direção de L1
para L2, L2 para L3 e assim por diante. Para N pontos há `max(N − 1, 0)`
direções; cada direção acrescenta duas componentes, cosseno e seno. Para
calcular cada uma, subtrai-se a coordenada do primeiro ponto da coordenada do
segundo e divide-se a diferença pelo seu comprimento. Essas direções descrevem
as **cordas entre pontos**, não a direção do segmento da pista em cada ponto.

Essas direções podem ser calculadas a partir das coordenadas e, portanto, não
acrescentam informação geométrica nova. Porém, entregá-las prontas pode ajudar
o aprendizado.


## 10. Configuração do experimento

Em `examples/train_sac.py`, escolha uma observação e seus parâmetros:

```python
OBSERVATION = "hovercraft_frame"  # ou "world_frame"
OBSERVATION_PARAMS = {
    "lookahead_distances": [3, 8, 15, 25],
    "include_directions": True,
    "include_hovercraft_heading": False,
    "include_lateral_error": True,  # False remove o sinal de erro lateral normalizado
}
```

Use `{}` para os padrões. O nome e os parâmetros definidos assim são salvos no `config.json` 
e usados pela avaliação. Parâmetros também podem ser passados diretamente ao ambiente:

```python
env = make_env(
    "gp",
    observation="world_frame",
    observation_params={"n_lookahead": 3, "lookahead_spacing": 4.0},
)
```

## 11. Quantidade de sinais

Por padrão, com 5 pontos e direções entre pontos desligadas, os vetores têm 20 valores em
`world_frame` e 18 em `hovercraft_frame`. 

Em geral, têm respectivamente
`10 + 2N + 2 max(N − 1, 0)` e `8 + 2N + 2 max(N − 1, 0)` valores quando as
direções estão habilitadas; sem elas, elimine o último termo. Incluir a direção
constante do hovercraft soma dois valores ao `hovercraft_frame`.
Definir `include_lateral_error=False` subtrai um valor em qualquer uma das
duas opções; os demais sinais mantêm a mesma ordem relativa.
