import os
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor, VecNormalize
from stable_baselines3.common.logger import configure

from gym_env_carla import CarlaRLEnv


def make_env(rank, seed=0):
    """
    Crée un environnement CARLA avec reset automatique.
    rank : index de l'environnement (utile si multi-env)
    seed : seed pour reproducibilité
    """
    def _init():
        env = CarlaRLEnv()
        obs, _ = env.reset(seed=seed + rank)
        print(f"[Env {rank}] Reset done, starting training...")
        return env
    return _init


if __name__ == "__main__":
    NUM_ENVS = 1
    TOTAL_TIMESTEPS = 300_000

    base_dir = os.path.dirname(os.path.abspath(__file__))
    logdir = os.path.join(base_dir, "logs")
    checkpoint_dir = os.path.join(base_dir, "checkpoints")
    os.makedirs(logdir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)

    print("🔹 Initialisation de l'environnement vectorisé...")
    env = DummyVecEnv([make_env(i) for i in range(NUM_ENVS)])
    env = VecMonitor(env)
    env = VecNormalize(env, norm_obs=False, norm_reward=True)

    final_model_path = os.path.join(base_dir, "ppo_carla_final.zip")

    # Chargement ou nouveau modèle
    if os.path.exists(final_model_path):
        print("🔄 Reprise du modèle existant...")
        model = PPO.load(final_model_path, env)
    else:
        print("🚀 Nouveau modèle PPO (CnnPolicy)")
        model = PPO(
            policy="CnnPolicy",
            env=env,
            learning_rate=3e-4,
            n_steps=1024,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            clip_range=0.2,
            ent_coef=0.01,
            verbose=1,
            tensorboard_log=logdir,
        )

    # Sauvegardes régulières
    checkpoint_callback = CheckpointCallback(
        save_freq=5000,
        save_path=checkpoint_dir,
        name_prefix="ppo_carla"
    )

    logger = configure(logdir, ["stdout", "csv", "tensorboard"])
    model.set_logger(logger)

    try:
        print("🏁 Démarrage de l'entraînement...")
        model.learn(
            total_timesteps=TOTAL_TIMESTEPS,
            callback=checkpoint_callback,
            progress_bar=True
        )
    finally:
        env.close()

    model.save(final_model_path)
    print("🎉 Modèle sauvegardé :", final_model_path)
