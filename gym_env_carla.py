# gym_env_carla.py
import carla
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import math
import cv2
from collections import deque
from typing import Tuple, Dict, Any


class CarlaRLEnv(gym.Env):
    """
    Carla RL environment with:
    - synchronous mode
    - RGB camera + frame stacking
    - reward for progress + reward for speed
    - idle detection / penalty
    - optional OpenCV preview (show_cam)
    - render flag to update CARLA spectator
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
        show_cam: bool = True,  # show local OpenCV preview of camera frames
        idle_speed_threshold: float = 0.5,  # m/s
        idle_max_steps: int = 50,  # if idle for > this many steps -> penalty & done
    ):
        super().__init__()

        # config
        self.host = host
        self.port = port
        self.town = town
        self.image_w = image_w
        self.image_h = image_h
        self.frame_stack = frame_stack
        self.fixed_delta_seconds = fixed_delta_seconds
        self.render = render
        self.show_cam = show_cam
        self.idle_speed_threshold = idle_speed_threshold
        self.idle_max_steps = idle_max_steps

        # carla client/world
        self.client = carla.Client(self.host, self.port)
        self.client.set_timeout(10.0)
        self.client.load_world(self.town)
        self.world = self.client.get_world()
        self.blueprint_library = self.world.get_blueprint_library()

        # synchronous mode
        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = self.fixed_delta_seconds
        self.world.apply_settings(settings)

        # actors placeholders
        self.actor_list = []
        self.vehicle = None
        self.collision_sensor = None
        self.lane_sensor = None
        self.camera = None

        # frame stack
        self._image_stack = deque(maxlen=self.frame_stack)
        zero_img = np.zeros((self.image_h, self.image_w, 3), dtype=np.uint8)
        for _ in range(self.frame_stack):
            self._image_stack.append(zero_img.copy())

        # camera buffer
        self._last_image_bytes = None
        self._last_image_timestamp = 0.0
        self._warned_black = False  # to warn once if camera black

        # histories & counters
        self.collision_hist = []
        self.lane_hist = []
        self.prev_dist_to_goal = None
        self.prev_action = np.array([0.0, 0.0], dtype=np.float32)
        self.success_count = 0
        self.attempt_count = 0

        # idle detection
        self.idle_steps = 0

        # action & observation space
        self.action_space = spaces.Box(
            low=np.array([0.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )
        stacked_channels = 3 * self.frame_stack
        # state: [speed, heading_error, dist_to_goal, progress_est, dist_from_center]
        self.observation_space = spaces.Dict(
            {
                "image": spaces.Box(
                    low=0,
                    high=255,
                    shape=(self.image_h, self.image_w, stacked_channels),
                    dtype=np.uint8,
                ),
                "state": spaces.Box(low=-np.inf, high=np.inf, shape=(5,), dtype=np.float32),
            }
        )

        # distances / spawn
        self.arrival_distance = 70.0
        mp = self.world.get_map()
        waypoint = mp.get_waypoint(carla.Location(x=230, y=180, z=0))
        self.start_transform = waypoint.transform
        self.fixed_spawn = self.start_transform.location
        self.fixed_rotation = self.start_transform.rotation

        # compute end location
        start_wp = mp.get_waypoint(self.fixed_spawn)
        end_wp = start_wp.next(self.arrival_distance)[0]
        self.end_location = end_wp.transform.location

        # debug draw (only if render; avoid drawing too often)
        if self.render:
            self._draw_path_on_road(start_wp, end_wp)

    # -----------------------
    # reset
    # -----------------------
    def reset(self, *, seed=None, options=None) -> Tuple[Dict[str, Any], Dict]:
        super().reset(seed=seed)
        self.attempt_count += 1

        # cleanup previous actors we spawned
        self._clean_actors()

        # reset histories / buffers
        self.collision_hist = []
        self.lane_hist = []
        self._image_stack.clear()
        zero_img = np.zeros((self.image_h, self.image_w, 3), dtype=np.uint8)
        for _ in range(self.frame_stack):
            self._image_stack.append(zero_img.copy())
        self._last_image_bytes = None
        self._warned_black = False
        self.prev_action = np.array([0.0, 0.0], dtype=np.float32)
        self.prev_dist_to_goal = None
        self.idle_steps = 0

        # spawn vehicle
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

        # sensors
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

        # recompute end
        mp = self.world.get_map()
        start_wp = mp.get_waypoint(self.fixed_spawn)
        end_wp = start_wp.next(self.arrival_distance)[0]
        self.end_location = end_wp.transform.location

        # let the world tick a few times so sensors produce
        for _ in range(3):
            self.world.tick()

        self.prev_dist_to_goal = self._distance(self.vehicle.get_location(), self.end_location)

        # update spectator if rendering enabled
        if self.render:
            self._update_spectator()

        obs = self._get_observation()
        return obs, {}

    # -----------------------
    # step
    # -----------------------
    def step(self, action: np.ndarray) -> Tuple[Dict[str, Any], float, bool, bool, Dict]:
        # smoothing
        alpha = 0.3
        action = np.clip(action, self.action_space.low, self.action_space.high)
        smoothed = alpha * action + (1 - alpha) * self.prev_action
        self.prev_action = smoothed.copy()
        throttle, steer = float(smoothed[0]), float(smoothed[1])

        # apply control
        self.vehicle.apply_control(carla.VehicleControl(throttle=throttle, steer=steer, brake=0.0))
        self.world.tick()

        # update spectator if requested
        if self.render:
            self._update_spectator()

        # build observation
        obs = self._get_observation()

        # compute reward
        loc = self.vehicle.get_location()
        current_dist = self._distance(loc, self.end_location)
        progress = (self.prev_dist_to_goal - current_dist) if self.prev_dist_to_goal is not None else 0.0
        self.prev_dist_to_goal = current_dist

        # speed (m/s)
        vel = self.vehicle.get_velocity()
        speed = math.sqrt(vel.x ** 2 + vel.y ** 2 + vel.z ** 2)

        # reward composition:
        #  - progress along path (main signal)
        #  - small bonus for speed to encourage movement
        #  - penalties: time, steering magnitude, steer changes, lane/collision, distance-to-center
        reward = 0.0
        reward += float(progress) * 10.0            # progress reward
        reward += 0.05 * float(speed)              # reward for moving forward (solution 1)
        reward -= 0.02                             # small time penalty
        reward -= abs(steer) * 0.01 * (1.0 + speed)  # discourage wild steering

        # penalize steer changes (smoothness)
        steer_change = abs(smoothed[1] - getattr(self, "prev_steer", 0.0))
        reward -= steer_change * 0.02
        self.prev_steer = float(smoothed[1])

        terminated = False
        truncated = False

        # lane invasion -> terminate episode and penalty (avoid respawn during episode)
        if len(self.lane_hist) > 0:
            reward -= 8.0
            self.lane_hist = []
            terminated = True

        # collision -> terminate stronger
        if len(self.collision_hist) > 0:
            reward -= 50.0
            terminated = True

        # idle detection (solution 1)
        if speed < self.idle_speed_threshold:
            self.idle_steps += 1
        else:
            self.idle_steps = 0

        if self.idle_steps > self.idle_max_steps:
            # if agent sits idle too long: penalize and end episode
            reward -= 8.0
            terminated = True

        # goal reached
        if current_dist < 2.0:
            reward += 100.0
            terminated = True
            self.success_count += 1

        # penalty for distance from lane center (computed from waypoint)
        try:
            wp = self.world.get_map().get_waypoint(loc)
            lane_center = wp.transform.location
            dist_from_center = self._distance(loc, lane_center)
            # scale penalty (tunable)
            reward -= dist_from_center * 2.0
        except Exception:
            # if waypoint lookup fails, skip penalty
            dist_from_center = 0.0

        return obs, float(reward), bool(terminated), bool(truncated), {}

    # -----------------------
    # camera callback
    # -----------------------
    def _on_cam_image(self, image: carla.Image):
        # minimal copy of raw_data for processing in _get_observation
        self._last_image_bytes = bytes(image.raw_data)
        self._last_image_timestamp = image.timestamp

    # -----------------------
    # observation builder
    # -----------------------
    def _get_observation(self) -> Dict[str, Any]:
        # build RGB image from raw bytes
        if self._last_image_bytes is not None:
            arr = np.frombuffer(self._last_image_bytes, dtype=np.uint8)
            cam_w = int(self.camera.attributes.get("image_size_x", self.image_w))
            cam_h = int(self.camera.attributes.get("image_size_y", self.image_h))
            expected = cam_w * cam_h * 4
            if arr.size == expected:
                arr = arr.reshape((cam_h, cam_w, 4))
                img = arr[:, :, :3][:, :, ::-1]  # BGRA -> RGB
            else:
                img = np.zeros((self.image_h, self.image_w, 3), dtype=np.uint8)
            img = cv2.resize(img, (self.image_w, self.image_h))
        else:
            img = np.zeros((self.image_h, self.image_w, 3), dtype=np.uint8)

        # optional check: is image mostly black? warn once if so
        if self.show_cam:
            mean_val = img.mean()
            if mean_val < 5 and not self._warned_black:
                print("[Warning] camera image very dark/black (mean pixel < 5). Check sensor or lighting.")
                self._warned_black = True
            # show preview (non-blocking)
            try:
                cv2.imshow("carla_cam", img)
                cv2.waitKey(1)
            except Exception:
                # in headless env this can fail; ignore
                pass

        # push into frame stack
        self._image_stack.append(img)
        stacked = np.concatenate(list(self._image_stack), axis=2).astype(np.uint8)

        # build state vector
        vt = self.vehicle.get_transform()
        vel = self.vehicle.get_velocity()
        speed = math.sqrt(vel.x ** 2 + vel.y ** 2 + vel.z ** 2)

        # heading error (2D)
        fwd = vt.get_forward_vector()
        to_goal = np.array([self.end_location.x - vt.location.x, self.end_location.y - vt.location.y], dtype=np.float32)
        fwd_vec = np.array([fwd.x, fwd.y], dtype=np.float32)
        norm_f = np.linalg.norm(fwd_vec) + 1e-8
        norm_g = np.linalg.norm(to_goal) + 1e-8
        dot = float(np.dot(fwd_vec, to_goal) / (norm_f * norm_g + 1e-12))
        dot = np.clip(dot, -1.0, 1.0)
        heading_error = math.acos(dot)

        dist_to_goal = float(self._distance(vt.location, self.end_location))
        progress_est = float(self.prev_dist_to_goal - dist_to_goal) if self.prev_dist_to_goal is not None else 0.0

        # distance to center of lane via waypoint
        try:
            wp = self.world.get_map().get_waypoint(vt.location)
            lane_center = wp.transform.location
            dist_from_center = float(self._distance(vt.location, lane_center))
        except Exception:
            dist_from_center = 0.0

        state = np.array([speed, heading_error, dist_to_goal, progress_est, dist_from_center], dtype=np.float32)

        return {"image": stacked, "state": state}

    # -----------------------
    # utils
    # -----------------------
    def _distance(self, a: carla.Location, b: carla.Location) -> float:
        return math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2 + (a.z - b.z) ** 2)

    def _update_spectator(self):
        try:
            spectator = self.world.get_spectator()
            vt = self.vehicle.get_transform()
            fwd = vt.get_forward_vector()
            pos = vt.location - fwd * 7.0
            pos.z += 3.0
            spectator.set_transform(carla.Transform(pos, carla.Rotation(pitch=-10, yaw=vt.rotation.yaw)))
        except Exception:
            pass

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
        try:
            self._clean_actors()
        finally:
            try:
                settings = self.world.get_settings()
                settings.synchronous_mode = False
                settings.fixed_delta_seconds = None
                self.world.apply_settings(settings)
            except Exception:
                pass
