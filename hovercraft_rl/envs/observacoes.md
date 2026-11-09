# Observações comparáveis: mundo e hovercraft

As opções `world_frame` e `hovercraft_frame` descrevem os mesmos elementos do
ambiente em eixos diferentes. Foram criadas para estudar o efeito do
referencial em experimentos; as opções anteriores `reference_error` e
`body_frame` continuam disponíveis e não foram alteradas.

## Referenciais e origem

Seja **H** a posição do hovercraft e **P** a projeção de H na linha central da
pista, escolhida pelo ambiente. As duas opções usam P como origem:

- `world_frame`: eixos X e Y fixos do mundo.
- `hovercraft_frame`: eixo X aponta para a frente do veículo e eixo Y para a
  esquerda. Esses eixos giram com o veículo, mas a origem continua sendo P,
  **não** H.

O ambiente usa a referência anterior para procurar P em uma região próxima da
pista durante o movimento. Portanto, em pistas com trechos próximos, P não é
necessariamente o ponto globalmente mais próximo.

Se `ψ` é o ângulo do hovercraft em relação ao eixo X do mundo, um vetor `(x, y)`
nos eixos do mundo é representado nos eixos do veículo como:

```text
frente   =  cos(ψ) x + sin(ψ) y
esquerda = -sin(ψ) x + cos(ψ) y
```

Por exemplo, se P → H é `(2, 1)` metros no mundo e o hovercraft aponta a 90°
do eixo X, nos eixos do veículo o mesmo vetor é `(1, -2)` metros.

## Conteúdo da observação

Os valores são colocados na seguinte ordem:

| Grupo | `world_frame` | `hovercraft_frame` |
|---|---|---|
| Posição | Componentes de P → H, em metros, nos eixos do mundo | Mesmo vetor nos eixos do veículo |
| Erro lateral (opcional, incluído por padrão) | Distância com sinal à linha central, dividida pela meia-largura | O mesmo valor |
| Direção do hovercraft | `(cos ψ, sin ψ)` | Omitida por padrão; opcionalmente `(1, 0)` |
| Direção local da pista | `(cos θ, sin θ)` | `(cos(θ − ψ), sin(θ − ψ))` |
| Velocidades | `u`, `v`, `r` | Os mesmos valores |
| Pontos à frente | Vetores P → Li nos eixos do mundo | Os mesmos vetores nos eixos do veículo |
| Direções entre pontos | Opcionalmente, pares `(cos α, sin α)` | Opcionalmente, `(cos(α − ψ), sin(α − ψ))` |

O erro lateral é positivo à esquerda do **sentido da pista**, não
necessariamente à esquerda do veículo. Valores próximos de ±1 indicam que o
veículo está próximo do limite permitido. A direção θ é a do segmento associado
à projeção; em um vértice, a escolha desse segmento afeta a direção.
Para omitir esse sinal, configure `include_lateral_error=False`. O vetor de
posição P → H continua presente, mas a observação deixa de fornecer diretamente
o erro lateral normalizado pela largura.

`u` e `v` são as velocidades longitudinal e lateral no referencial do veículo
(m/s), e `r` é a velocidade angular (rad/s). A orientação do veículo no seu
próprio referencial é constante; incluí-la não adiciona informação.

Note que `world_frame` fornece a orientação **absoluta** do veículo e
`hovercraft_frame` não. Portanto, apesar das features geométricas análogas, os
dois vetores não contêm exatamente a mesma informação: a invariância à rotação
global é uma diferença intencional. Com o padrão de omitir a direção constante,
eles também diferem em tamanho por dois valores. Para comparar os formatos com
o mesmo tamanho, habilite `include_hovercraft_heading` na opção relativa.

## Pontos à frente

Os pontos são obtidos sobre a linha central, a distâncias de arco a partir de P.
O padrão é 5 pontos a 5, 10, 15, 20 e 25 metros. As coordenadas fornecidas são
os vetores **de P até cada ponto**, e não do veículo até os pontos. Para obter
H → Li, subtraia o vetor P → H do vetor P → Li no mesmo referencial.

Há duas formas de configurar os pontos:

```python
# Dois pontos, a 4 e 8 m de P:
{"n_lookahead": 2, "lookahead_spacing": 4.0}

# Três pontos, a 2, 7 e 15 m de P:
{"lookahead_distances": [2, 7, 15]}

# Nenhum ponto:
{"lookahead_distances": []}
```

Não combine a lista com quantidade ou espaçamento uniforme. As distâncias
devem ser finitas, positivas e estritamente crescentes; **todas devem ser
menores que o comprimento da pista**, inclusive ao avaliar em outra pista.
No modo uniforme, `n_lookahead` deve ser inteiro não negativo e o espaçamento,
positivo e finito.

### Direções entre pontos

Com `include_directions=True`, o vetor também contém a direção de L1 → L2,
L2 → L3 e assim por diante. Para N pontos há `max(N − 1, 0)` direções; cada
direção acrescenta duas componentes, cosseno e seno. Elas descrevem as
**cordas entre pontos**, não a tangente da pista em cada ponto.

Essas direções podem ser calculadas a partir das coordenadas e, portanto, não
acrescentam informação geométrica nova. Entregá-las prontas pode ou não ajudar
o aprendizado: é uma hipótese a testar.

Se dois pontos consecutivos coincidirem, sua direção não está definida. Nesse
caso é emitido um aviso e o par é `(0, 0)` — um marcador, não um par válido de
seno e cosseno. Isso pode ocorrer em pistas que se cruzam ou se sobrepõem, mesmo
quando todas as distâncias são menores que o comprimento da pista.

## Configuração do experimento

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

Use `{}` para os padrões. Em vez da lista, pode-se usar `n_lookahead` e
`lookahead_spacing`. `include_hovercraft_heading` só altera o vetor
`hovercraft_frame`: em `world_frame` a orientação absoluta já é sempre
incluída. O nome e os parâmetros são salvos no `config.json` e usados pela
avaliação. Parâmetros também podem ser passados diretamente ao ambiente:

```python
env = make_env(
    "gp",
    observation="world_frame",
    observation_params={"n_lookahead": 3, "lookahead_spacing": 4.0},
)
```

Por padrão, com 5 pontos e direções desligadas, os vetores têm 20 valores em
`world_frame` e 18 em `hovercraft_frame`. Em geral, têm respectivamente
`10 + 2N + 2 max(N − 1, 0)` e `8 + 2N + 2 max(N − 1, 0)` valores quando as
direções estão habilitadas; sem elas, elimine o último termo. Incluir a direção
constante soma dois valores ao `hovercraft_frame`.
Definir `include_lateral_error=False` subtrai um valor em qualquer uma das
duas opções; os demais sinais mantêm a mesma ordem relativa.

Para comparar os referenciais, mantenha pista, parâmetros de lookahead,
direções opcionais, física e hiperparâmetros iguais. Repita os treinos com
várias seeds e avalie em condições iguais. Não se pode atribuir uma diferença
em relação ao `body_frame` antigo apenas ao referencial: a origem e outras
features também diferem.
