import gymnasium as gym
from gymnasium import spaces
import numpy as np
import carla
import random
import time
import cv2

class CarlaRLEnv(gym.Env):
    """Environnement CARLA avec caméra third-person et Spectator fluide"""

    def __init__(self, curriculum_level=1, camera_mode="third_person"):
        super().__init__()
        print("🔌 Initialisation de l'environnement CARLA...")

        self.client = carla.Client('localhost', 2000)
        self.client.set_timeout(10.0)
        try:
            self.world = self.client.get_world()
        except Exception as e:
            print("❌ Impossible de récupérer le monde CARLA :", e)
            raise e

        self.curriculum_level = curriculum_level
        self.camera_mode = camera_mode

        # Observation et action
        self.observation_space = spaces.Box(low=0, high=255, shape=(84, 84, 3), dtype=np.uint8)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

        self.vehicle = None
        self.camera_sensor = None
        self.image = np.zeros((84, 84, 3), dtype=np.uint8)

        # Position actuelle du spectator pour interpolation
        self.spectator_transform = None
        self.smooth_factor = 0.1  # plus petit = plus lent, plus fluide

    def reset(self, seed=None, options=None):
        print("♻️ Reset de l'environnement CARLA...")

        if self.vehicle is not None:
            self.vehicle.destroy()
            self.vehicle = None
        if self.camera_sensor is not None:
            self.camera_sensor.destroy()
            self.camera_sensor = None

        blueprint_library = self.world.get_blueprint_library()
        vehicle_bp = blueprint_library.find("vehicle.tesla.model3")
        spawn_point = random.choice(self.world.get_map().get_spawn_points())
        self.vehicle = self.world.spawn_actor(vehicle_bp, spawn_point)
        print("🚗 Véhicule spawné :", self.vehicle.type_id)

        # Caméra Python
        camera_bp = blueprint_library.find("sensor.camera.rgb")
        camera_bp.set_attribute("image_size_x", "84")
        camera_bp.set_attribute("image_size_y", "84")
        camera_bp.set_attribute("fov", "110")
        camera_transform = carla.Transform(
            carla.Location(x=-7.0, z=3.0),
            carla.Rotation(pitch=-10)
        )
        self.camera_sensor = self.world.spawn_actor(camera_bp, camera_transform, attach_to=self.vehicle)
        self.camera_sensor.listen(lambda data: self._process_image(data))
        print("📷 Caméra Python attachée")

        # Attente première image
        timeout = 5.0
        start = time.time()
        while np.all(self.image == 0):
            if time.time() - start > timeout:
                print("⚠️ Timeout: aucune image reçue de la caméra")
                break
            time.sleep(0.05)

        # Initialiser spectator
        self._init_spectator()
        print("✅ Reset terminé, première image reçue")
        return self.image, {}

    def _process_image(self, image):
        array = np.frombuffer(image.raw_data, dtype=np.uint8)
        array = array.reshape((image.height, image.width, 4))
        self.image = array[:, :, :3]

    def _init_spectator(self):
        """Position initiale du Spectator derrière la voiture"""
        if self.vehicle is not None:
            vehicle_transform = self.vehicle.get_transform()
            target = carla.Transform(
                vehicle_transform.location + carla.Location(x=-7, z=3),
                carla.Rotation(pitch=-10)
            )
            self.spectator_transform = target
            self.world.get_spectator().set_transform(target)

    def _update_spectator(self):
        """Déplacer le spectator de manière fluide derrière la voiture"""
        if self.vehicle is None or self.spectator_transform is None:
            return

        vehicle_transform = self.vehicle.get_transform()
        target_loc = vehicle_transform.location + carla.Location(x=-7, z=3)
        target_rot = carla.Rotation(pitch=-10)

        # Interpolation linéaire
        new_loc = carla.Location(
            x=self.spectator_transform.location.x * (1 - self.smooth_factor) + target_loc.x * self.smooth_factor,
            y=self.spectator_transform.location.y * (1 - self.smooth_factor) + target_loc.y * self.smooth_factor,
            z=self.spectator_transform.location.z * (1 - self.smooth_factor) + target_loc.z * self.smooth_factor
        )
        new_rot = carla.Rotation(
            pitch=self.spectator_transform.rotation.pitch * (1 - self.smooth_factor) + target_rot.pitch * self.smooth_factor
        )
        self.spectator_transform = carla.Transform(new_loc, new_rot)
        self.world.get_spectator().set_transform(self.spectator_transform)

    def step(self, action):
        steer = float(action[0])
        throttle = float(action[1])
        self.vehicle.apply_control(carla.VehicleControl(throttle=throttle, steer=steer))

        # Déplacer le spectator de manière fluide
        self._update_spectator()

        reward = 0.0
        terminated = False
        truncated = False
        info = {}

        time.sleep(0.05)
        return self.image, reward, terminated, truncated, info

    def render(self):
        cv2.imshow("CARLA Camera", self.image)
        cv2.waitKey(1)

    def close(self):
        if self.camera_sensor is not None:
            self.camera_sensor.stop()
            self.camera_sensor.destroy()
            self.camera_sensor = None
        if self.vehicle is not None:
            self.vehicle.destroy()
            self.vehicle = None
        cv2.destroyAllWindows()
        print("🛑 Environnement CARLA fermé")
