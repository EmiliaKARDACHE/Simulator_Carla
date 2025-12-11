import carla
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import math
import cv2
import time
from collections import deque
from typing import Tuple, Dict, Any


class CarlaRLEnv(gym.Env):
    """
    Environnement CARLA optimisé pour RL.
    - Mode synchrone (fixed_delta_seconds)
    - Callback caméra léger (stocke raw_data)
    - Conversion image dans _get_observation (centralisé)
    - Frame stacking
    - Observation: Dict { image: HxWxC, state: vecteur }
    - Reward shaping basé sur progression vers l'objectif
    - Action smoothing (exponential moving average)
    - Compteur de succès / tentatives
    """

    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}

    def __init__(
        self,
        host: str = "localhost",
        port: int = 2000,
        town: str = "Town05",
        image_w: int = 84,
        image_h: int = 84,
        frame_stack: int = 3,
        fixed_delta_seconds: float = 0.05,
        render: bool = True,
    ):
        super().__init__()

        # --- Config ---
        self.host = host
        self.port = port
        self.town = town
        self.image_w = image_w
        self.image_h = image_h
        self.frame_stack = frame_stack
        self.fixed_delta_seconds = fixed_delta_seconds
        self.render = render

        # CARLA client / world
        self.client = carla.Client(self.host, self.port)
        self.client.set_timeout(10.0)
        self.client.load_world(self.town)
        self.world = self.client.get_world()
        self.blueprint_library = self.world.get_blueprint_library()

        # Put world in synchronous mode for RL determinism
        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = self.fixed_delta_seconds
        self.world.apply_settings(settings)
        self._synced = True

        # Actor placeholders
        self.actor_list = []  # only actors we spawned
        self.vehicle = None
        self.collision_sensor = None
        self.lane_sensor = None
        self.camera = None

        # Camera raw storage (small & fast in callback)
        self._last_image_bytes = None
        self._last_image_timestamp = 0.0

        # Frame stack buffer (channels-last)
        self._image_stack = deque(maxlen=self.frame_stack)
        zero_img = np.zeros((self.image_h, self.image_w, 3), dtype=np.uint8)
        for _ in range(self.frame_stack):
            self._image_stack.append(zero_img.copy())

        # State buffers / histories
        self.collision_hist = []
        self.lane_hist = []
        self.prev_dist_to_goal = None
        self.prev_action = np.array([0.0, 0.0], dtype=np.float32)  # throttle, steer

        # Compteur succès / tentatives
        self.success_count = 0
        self.episode_count = 0

        # Action & observation spaces
        self.action_space = spaces.Box(
            low=np.array([0.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )

        stacked_channels = 3 * self.frame_stack
        self.observation_space = spaces.Dict(
            {
                "image": spaces.Box(
                    low=0, high=255, shape=(self.image_h, self.image_w, stacked_channels), dtype=np.uint8
                ),
                "state": spaces.Box(low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32),
            }
        )

        # Arrival target distance (meters)
        self.arrival_distance = 50.0

        # Spawn fixed location (example)
        mp = self.world.get_map()
        waypoint = mp.get_waypoint(carla.Location(x=230, y=180, z=0))
        self.start_transform = waypoint.transform
        self.fixed_spawn = self.start_transform.location
        self.fixed_rotation = self.start_transform.rotation

        # Compute end waypoint
        start_wp = mp.get_waypoint(self.fixed_spawn)
        end_wp = start_wp.next(self.arrival_distance)[0]
        self.end_location = end_wp.transform.location

        # debug draw once
        self._draw_path_on_road(start_wp, end_wp)

    # -----------------------
    # Public API
    # -----------------------
    def reset(self, *, seed=None, options=None) -> Tuple[Dict[str, Any], Dict]:
        super().reset(seed=seed)

        self.episode_count += 1

        self._clean_actors()

        self.collision_hist = []
        self.lane_hist = []
        self._last_image_bytes = None
        self._image_stack.clear()
        zero_img = np.zeros((self.image_h, self.image_w, 3), dtype=np.uint8)
        for _ in range(self.frame_stack):
            self._image_stack.append(zero_img.copy())
        self.prev_action = np.array([0.0, 0.0], dtype=np.float32)

        # Spawn vehicle
        base_transform = carla.Transform(self.fixed_spawn, self.fixed_rotation)
        self.vehicle = None
        bp = self.blueprint_library.filter("model3")[0]
        for i in range(6):
            t = carla.Transform(base_transform.location + carla.Location(z=i * 0.2), base_transform.rotation)
            self.vehicle = self.world.try_spawn_actor(bp, t)
            if self.vehicle:
                break
        if not self.vehicle:
            raise RuntimeError("Impossible de spawn la voiture.")
        self.actor_list.append(self.vehicle)

        # Sensors
        col_bp = self.blueprint_library.find("sensor.other.collision")
        self.collision_sensor = self.world.spawn_actor(col_bp, carla.Transform(), attach_to=self.vehicle)
        self.collision_sensor.listen(lambda event: self.collision_hist.append(event))
        self.actor_list.append(self.collision_sensor)

        lane_bp = self.blueprint_library.find("sensor.other.lane_invasion")
        self.lane_sensor = self.world.spawn_actor(lane_bp, carla.Transform(), attach_to=self.vehicle)
        self.lane_sensor.listen(lambda event: self.lane_hist.append(event))
        self.actor_list.append(self.lane_sensor)

        cam_bp = self.blueprint_library.find("sensor.camera.rgb")
        cam_bp.set_attribute("image_size_x", str(self.image_w * 8))
        cam_bp.set_attribute("image_size_y", str(self.image_h * 6))
        cam_bp.set_attribute("fov", "110")
        cam_transform = carla.Transform(carla.Location(x=1.5, y=0.0, z=1.6), carla.Rotation(pitch=-10))
        self.camera = self.world.spawn_actor(cam_bp, cam_transform, attach_to=self.vehicle)
        self.camera.listen(self._on_cam_image)
        self.actor_list.append(self.camera)

        # compute target
        mp = self.world.get_map()
        start_wp = mp.get_waypoint(self.fixed_spawn)
        end_wp = start_wp.next(self.arrival_distance)[0]
        self.end_location = end_wp.transform.location

        # tick world for sensors
        self.world.tick()
        for _ in range(3):
            self.world.tick()

        self.prev_dist_to_goal = self._distance(self.vehicle.get_location(), self.end_location)

        obs = self._get_observation()
        if self.render:
            self._update_spectator()
        return obs, {}

    def step(self, action: np.ndarray) -> Tuple[Dict[str, Any], float, bool, bool, Dict]:
        alpha = 0.3
        action = np.clip(action, self.action_space.low, self.action_space.high)
        smoothed = alpha * action + (1 - alpha) * self.prev_action
        self.prev_action = smoothed.copy()

        throttle = float(smoothed[0])
        steer = float(smoothed[1])
        self.vehicle.apply_control(carla.VehicleControl(throttle=throttle, steer=steer, brake=0.0))

        self.world.tick()
        if self.render:
            self._update_spectator()

        obs = self._get_observation()

        loc = self.vehicle.get_location()
        current_dist = self._distance(loc, self.end_location)
        progress = (self.prev_dist_to_goal - current_dist) if self.prev_dist_to_goal is not None else 0.0
        self.prev_dist_to_goal = current_dist

        reward = 0.0
        reward += float(progress) * 10.0
        reward -= 0.05
        velocity = self.vehicle.get_velocity()
        speed = math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)
        reward -= abs(steer) * 0.01 * (1.0 + speed)

        if len(self.lane_hist) > 0:
            reward -= 2.0
            self.lane_hist = []

        terminated = False
        if len(self.collision_hist) > 0:
            reward -= 50.0
            terminated = True

        if current_dist < 2.0:
            reward += 200.0
            terminated = True
            self.success_count += 1

        if loc.z < -1.0:
            reward -= 20.0
            terminated = True

        return obs, float(reward), bool(terminated), False, {}

    # -----------------------
    # Camera callback
    # -----------------------
    def _on_cam_image(self, image: carla.Image):
        self._last_image_bytes = bytes(image.raw_data)
        self._last_image_timestamp = image.timestamp

    # -----------------------
    # Observation build
    # -----------------------
    def _get_observation(self) -> Dict[str, Any]:
        if self._last_image_bytes is not None:
            try:
                cam_w = int(self.camera.attributes.get("image_size_x", self.image_w))
                cam_h = int(self.camera.attributes.get("image_size_y", self.image_h))
                arr = np.frombuffer(self._last_image_bytes, dtype=np.uint8)
                expected = cam_w * cam_h * 4
                if arr.size == expected:
                    arr = arr.reshape((cam_h, cam_w, 4))
                    img = arr[:, :, :3][:, :, ::-1]
                else:
                    possible_w = arr.size // (cam_h * 4)
                    if possible_w > 0:
                        arr = arr.reshape((cam_h, possible_w, 4))
                        img = arr[:, :, :3][:, :, ::-1]
                    else:
                        img = np.zeros((self.image_h, self.image_w, 3), dtype=np.uint8)
                img = cv2.resize(img, (self.image_w, self.image_h))
            except Exception:
                img = np.zeros((self.image_h, self.image_w, 3), dtype=np.uint8)
        else:
            img = np.zeros((self.image_h, self.image_w, 3), dtype=np.uint8)

        self._image_stack.append(img)
        stacked = np.concatenate(list(self._image_stack), axis=2).astype(np.uint8)

        velocity = self.vehicle.get_velocity()
        speed = math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)

        vt = self.vehicle.get_transform()
        fwd = vt.get_forward_vector()
        to_goal = carla.Location(
            x=self.end_location.x - vt.location.x,
            y=self.end_location.y - vt.location.y,
            z=self.end_location.z - vt.location.z,
        )

        fwd_vec = np.array([fwd.x, fwd.y], dtype=np.float32)
        goal_vec = np.array([to_goal.x, to_goal.y], dtype=np.float32)
        norm_f = np.linalg.norm(fwd_vec) + 1e-8
        norm_g = np.linalg.norm(goal_vec) + 1e-8
        dot = float(np.dot(fwd_vec, goal_vec) / (norm_f * norm_g))
        dot = np.clip(dot, -1.0, 1.0)
        heading_error = math.acos(dot)

        dist_to_goal = float(self._distance(vt.location, self.end_location))
        progress_est = float(self.prev_dist_to_goal - dist_to_goal) if self.prev_dist_to_goal else 0.0

        state = np.array([speed, heading_error, dist_to_goal, progress_est], dtype=np.float32)

        return {"image": stacked, "state": state}

    # -----------------------
    # Utils
    # -----------------------
    def _distance(self, a: carla.Location, b: carla.Location) -> float:
        return math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2 + (a.z - b.z) ** 2)

    def _draw_marker(self, location: carla.Location, color: carla.Color):
        self.world.debug.draw_box(
            box=carla.BoundingBox(location, carla.Vector3D(0.5, 0.5, 0.5)),
            rotation=carla.Rotation(),
            color=color,
            thickness=0.2,
            life_time=0,
        )

    def _draw_path_on_road(self, start_wp, end_wp):
        waypoints = [start_wp]
        current_wp = start_wp
        while current_wp.transform.location.distance(end_wp.transform.location) > 2.0:
            next_wps = current_wp.next(2.0)
            if len(next_wps) == 0:
                break
            current_wp = next_wps[0]
            waypoints.append(current_wp)
        for i in range(len(waypoints) - 1):
            self.world.debug.draw_line(
                begin=waypoints[i].transform.location,
                end=waypoints[i + 1].transform.location,
                thickness=0.1,
                color=carla.Color(0, 255, 0),
                life_time=0,
            )

    def _update_spectator(self):
        if self.vehicle is None:
            return
        spectator = self.world.get_spectator()
        vt = self.vehicle.get_transform()
        fwd = vt.get_forward_vector()
        pos = vt.location - fwd * 7.0
        pos.z += 3.0
        spectator.set_transform(carla.Transform(pos, carla.Rotation(pitch=-10, yaw=vt.rotation.yaw)))

    def _clean_actors(self):
        for actor in self.actor_list:
            try:
                actor.destroy()
            except Exception:
                pass
        self.actor_list = []
        self.camera = None
        self.collision_sensor = None
        self.lane_sensor = None
        self.vehicle = None
        try:
            self.world.tick()
        except Exception:
            pass

    def close(self):
        self._clean_actors()
        try:
            settings = self.world.get_settings()
            settings.synchronous_mode = False
            settings.fixed_delta_seconds = None
            self.world.apply_settings(settings)
        except Exception:
            pass
