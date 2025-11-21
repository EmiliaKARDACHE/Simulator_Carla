import os
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor, VecNormalize
from stable_baselines3.common.logger import configure

from gym_env_carla import CarlaRLEnv

def make_env(rank, seed=0, camera_mode="third_person"):
    def _init():
        print(f"🛠 Création de l'environnement {rank}")
        env = CarlaRLEnv(curriculum_level=1, camera_mode=camera_mode)
        obs, _ = env.reset(seed=seed + rank)
        print(f"✅ Environnement {rank} prêt, image initiale {obs.shape}")
        return env
    return _init

if __name__ == "__main__":
    print("🚀 Lancement du script PPO CARLA...")

    NUM_ENVS = 1  # Windows + CARLA
    TOTAL_TIMESTEPS = 300_000

    base_dir = os.path.dirname(os.path.abspath(__file__))
    logdir = os.path.join(base_dir, "logs", "carla_autopilot")
    checkpoint_dir = os.path.join(base_dir, "checkpoints_carla")
    os.makedirs(logdir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)

    env = DummyVecEnv([make_env(i) for i in range(NUM_ENVS)])
    env = VecMonitor(env)
    env = VecNormalize(env, norm_obs=False, norm_reward=True)

    checkpoint_callback = CheckpointCallback(
        save_freq=5000,
        save_path=checkpoint_dir,
        name_prefix="ppo_carla"
    )

    model = PPO(
        policy="CnnPolicy",
        env=env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        clip_range=0.2,
        ent_coef=0.01,
        verbose=1,
        tensorboard_log=logdir,
    )

    new_logger = configure(logdir, ["stdout", "csv", "tensorboard"])
    model.set_logger(new_logger)

    try:
        model.learn(
            total_timesteps=TOTAL_TIMESTEPS,
            callback=checkpoint_callback,
            progress_bar=True
        )
    finally:
        env.close()
        print("🛑 Entraînement terminé, environnement fermé")

    model.save(os.path.join(base_dir, "ppo_carla_final"))
    print("🎉 Modèle enregistré :", os.path.join(base_dir, "ppo_carla_final.zip"))
