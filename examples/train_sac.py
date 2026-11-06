"""Treina um agente SAC (stable-baselines3) no ambiente do hovercraft.

Como usar
---------
1. Edite o bloco "Configuração do experimento" logo abaixo (pista, observação, física, etc.).
2. Execute este arquivo (botão "Run" do VS Code ou ``python examples/train_sac.py``).
3. Ao final, tudo é salvo em uma pasta nova dentro de ``examples/runs-sac/``:
   - ``model.zip``         : o agente treinado;
   - ``vecnormalize.pkl``  : estatísticas de normalização das observações (necessário para usar o modelo);
   - ``config.json``       : parâmetros do ambiente (lido por ``evaluate.py`` para recriar o mesmo ambiente);
   - ``checkpoints/``      : cópias intermediárias do modelo.
4. Veja o agente em ação com ``python examples/evaluate.py <pasta-do-treino>``.
5. Acompanhe as curvas de aprendizado com ``tensorboard --logdir examples/runs-sac/tb``.

Para comparar experimentos (ex.: observações ou pistas diferentes), rode o script várias vezes
mudando UMA coisa por vez no bloco de configuração e mantenha o resto igual. Cada execução cria
uma pasta com nome único (que inclui pista, observação, física, data/hora e seed), e o TensorBoard
mostra todas as execuções juntas para comparação.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize

from hovercraft_rl.envs import make_env

# --------------------------------------------------------------------------------------
# Configuração do experimento (edite aqui para testar outros regimes de treinamento).

# --- Ambiente ---
# Pista, opções: "circle", "gp", "kidney", "oval" ou "rounded_rect" (arquivos em hovercraft_rl/track/data).
TRACK = "gp"

# Observação que o agente recebe (ver hovercraft_rl/envs/observations.py):
#   "body_frame"      : tudo relativo ao veículo (erro lateral, ângulo em relação à pista, u, v, r, ...);
#                       não depende da posição absoluta, então tende a generalizar melhor entre pistas.
#   "reference_error" : erro de posição/orientação em relação ao ponto de referência na pista.
OBSERVATION = "body_frame"

# Onde o veículo começa cada episódio: "fixed" (sempre na largada) ou "random" (ponto aleatório da pista).
START_MODE = "random"

# Limite de passos por episódio (5000 passos = 100 s simulados); ao atingir, o episódio é truncado.
MAX_EPISODE_STEPS = 5_000

# --- Treinamento ---
SEED = 0                    # semente aleatória; varie-a para medir a variabilidade dos resultados
N_ENVS = 1                  # nº de ambientes em paralelo (1 é suficiente para o SAC)
TOTAL_TIMESTEPS = 150_000   # duração do treino, em passos de ambiente
CHECKPOINT_EVERY = 50_000  # salva uma cópia do modelo a cada tantos passos

# --- Modelo físico do hovercraft ---
# BETA_V, BETA_U e BETA_R são coeficientes de arrasto (atrito com o ar): quanto maior, mais o
# veículo "freia" sozinho nas respectivas velocidades (v = lateral, u = frontal, r = rotação).
PHYSICS = "3dof+drag"
BETA_V = 1.00
BETA_U = 0.15
BETA_R = 0.70

# --- Hiperparâmetros principais do SAC (os demais usam os valores default do stable_baselines3) ---
GAMMA = 0.99          # fator de desconto: quanto o agente valoriza recompensas futuras (0 a 1)
LEARNING_RATE = 3e-4  # passo de aprendizado das redes neurais
BATCH_SIZE = 256      # nº de transições amostradas do buffer de replay a cada atualização

# --- Arquitetura das MLPs: neurônios de cada camada oculta ---
# Ex.: [256, 256] = duas camadas de 256 neurônios. Actor = política; critic = estima o valor das ações.
ACTOR_HIDDEN = [256, 256]
CRITIC_HIDDEN = [256, 256]

# Pasta onde as execuções são salvas.
RUNS_DIR = Path(__file__).resolve().parent / "runs-sac"
# --------------------------------------------------------------------------------------


def main() -> None:
    # Nome único por execução (inclui data/hora): evita sobrescrever treinos anteriores.
    run_name = (
        f"sac_{TRACK}_{OBSERVATION}_bv{BETA_V:g}_bu{BETA_U:g}_br{BETA_R:g}_"
        f"{datetime.now():%Y%m%d_%H%M%S_%f}_s{SEED}"
    )
    run_dir = RUNS_DIR / run_name
    checkpoints_dir = run_dir / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=False)

    # Argumentos para criação do ambiente usando o make_env (do módulo hovercraft_rl). 
    # Ação "continuous": o agente escolhe diretamente valores contínuos de força/torque.
    env_kwargs = {
        "track": TRACK,
        "physics": PHYSICS,
        "physics_params": {"beta_v": BETA_V, "beta_u": BETA_U, "beta_r": BETA_R},
        "observation": OBSERVATION,
        "start": START_MODE,
        "max_episode_steps": MAX_EPISODE_STEPS,
        "action_mode": "continuous",
    }

    # Salvo para o evaluate.py recriar o mesmo ambiente (e carregar o algoritmo certo) depois.
    config = {
        "algo": "sac", 
        "env_kwargs": env_kwargs
    }
    (run_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    # make_vec_env cria N_ENVS cópias do ambiente. VecNormalize normaliza as observações
    # (média 0, desvio 1), o que ajuda muito o aprendizado; as estatísticas são salvas ao final.
    vec_env = make_vec_env(make_env, n_envs=N_ENVS, seed=SEED, env_kwargs=env_kwargs)
    vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=False, gamma=GAMMA)
    try:
        # Para outro algoritmo (PPO, TD3, ...), troque SAC por ele e ajuste os hiperparâmetros.
        # "MlpPolicy" = redes totalmente conectadas (MLPs); verbose=1 imprime o progresso.
        model = SAC(
            "MlpPolicy",
            vec_env,
            seed=SEED,
            gamma=GAMMA,
            learning_rate=LEARNING_RATE,
            batch_size=BATCH_SIZE,
            policy_kwargs={"net_arch": {"pi": ACTOR_HIDDEN, "qf": CRITIC_HIDDEN}},
            verbose=1,
            tensorboard_log=str(RUNS_DIR / "tb"),
        )
        checkpoint = CheckpointCallback(
            save_freq=max(CHECKPOINT_EVERY // N_ENVS, 1),
            save_path=str(checkpoints_dir),
            save_vecnormalize=True,
        )
        # Treina de fato; tb_log_name nomeia a curva no TensorBoard.
        model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=checkpoint, tb_log_name=run_name)
        # Salve o modelo E a normalização: um não funciona sem o outro.
        model.save(str(run_dir / "model"))
        vec_env.save(str(run_dir / "vecnormalize.pkl"))
    finally:
        vec_env.close()

    print(f"Treino concluído. Artefatos salvos em: {run_dir}")
    print(f"Avalie com: python examples/evaluate.py {run_dir}")


if __name__ == "__main__":
    main()
