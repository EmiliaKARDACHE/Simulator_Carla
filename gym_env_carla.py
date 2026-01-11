# gym_env_carla.py
import carla
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import math
import cv2
from collections import deque
from typing import Tuple, Dict, Any
import time
import random


class CarlaRLEnv(gym.Env):
    """
    Carla RL environment with:
    - synchronous mode
    - RGB camera + frame stacking
    - LiDAR BEV grid
    - reward for progress + reward for speed
    - idle detection / penalty
    - traffic + pedestrians + work zones
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
        fixed_delta_seconds: float = 0.001,  # larger dt -> fewer ticks per episode
        render: bool = False,          # was True: default to headless
        show_cam: bool = False,        # was True: no OpenCV window by default
        idle_speed_threshold: float = 0.5,
        idle_max_steps: int = 50,
        max_steps: int = 400,
        scenario: str = "random",          # NEW: "clear" | "wet" | "night" | "random"
        num_npc_vehicles: int = 20,        # NEW
        num_pedestrians: int = 10,         # NEW
        use_lidar: bool = True,           # NEW: allow disabling LiDAR actor
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
        self.max_steps = max_steps
        self.current_step = 0

        self.scenario = scenario
        self.num_npc_vehicles = num_npc_vehicles
        self.num_pedestrians = num_pedestrians
        self.use_lidar = use_lidar       # NEW

        # LiDAR config
        self.lidar_range = 50.0
        self.lidar_bev_size = 64
        self._last_lidar_points = None

        # Highway-code parameters (NEW)
        self.speed_limit_mps = 13.9         # ≈ 50 km/h hard limit
        self.speed_limit_soft_mps = 11.0    # smooth penalty starts here
        self.stop_check_distance = 20.0     # how far we start enforcing stops
        self.stop_speed_threshold = 0.5     # must be almost stopped at stop line

        # carla client/world
        self.client = carla.Client(self.host, self.port)
        self.client.set_timeout(300000.0)
        max_retries = 10
        for attempt in range(1, max_retries + 1):
            try:
                print(f"[CARLA] Connecting / loading world '{self.town}' (attempt {attempt}/{max_retries})...")
                self.client.load_world(self.town)
                self.world = self.client.get_world()
                break
            except RuntimeError as e:
                print(f"[CARLA] Connection failed: {e}")
                if attempt == max_retries:
                    raise RuntimeError(
                        f"Could not connect to CARLA at {self.host}:{self.port} "
                        f"or load world '{self.town}' after {max_retries} attempts. "
                        "Make sure the CARLA simulator is running."
                    )
                time.sleep(2.0)

        self.blueprint_library = self.world.get_blueprint_library()

        # synchronous mode + traffic manager
        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = self.fixed_delta_seconds
        settings.no_rendering_mode = not self.render   # already disables server rendering
        self.world.apply_settings(settings)

        self.traffic_manager = self.client.get_trafficmanager()
        self.traffic_manager.set_synchronous_mode(True)
        self.traffic_manager.set_global_distance_to_leading_vehicle(2.0)

        # Pre-cache traffic infrastructure (NEW)
        self.stop_signs = list(self.world.get_actors().filter("traffic.stop"))

        # actors placeholders
        self.actor_list = []          # per-episode actors (ego + sensors)
        self.vehicle = None
        self.collision_sensor = None
        self.lane_sensor = None
        self.camera = None
        self.lidar = None
        # static / background actors (traffic, pedestrians, controllers, work zones)
        self.background_actors = []   # NEW: persistent across episodes
        self.work_zone_actors = []    # already existed, keep as-is

        # frame stack
        self._image_stack = deque(maxlen=self.frame_stack)
        zero_img = np.zeros((self.image_h, self.image_w, 3), dtype=np.uint8)
        for _ in range(self.frame_stack):
            self._image_stack.append(zero_img.copy())

        # camera / lidar buffers
        self._last_image_bytes = None
        self._last_image_timestamp = 0.0
        self._warned_black = False
        # LiDAR buffer already declared (_last_lidar_points)

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

        self.observation_space = spaces.Dict(
            {
                "image": spaces.Box(
                    low=0,
                    high=255,
                    shape=(self.image_h, self.image_w, stacked_channels),
                    dtype=np.uint8,
                ),
                # LiDAR bird's-eye grid in [0, 1]
                "lidar": spaces.Box(
                    low=0.0,
                    high=1.0,
                    shape=(self.lidar_bev_size, self.lidar_bev_size, 1),
                    dtype=np.float32,
                ),
                # [speed, heading_error, dist_to_goal, progress_est, dist_from_center]
                "state": spaces.Box(low=-np.inf, high=np.inf, shape=(5,), dtype=np.float32),
            }
        )

        # distances / spawn
        self.arrival_distance = 40.0
        mp = self.world.get_map()
        waypoint = mp.get_waypoint(carla.Location(x=230, y=180, z=0))
        self.start_transform = waypoint.transform
        self.fixed_spawn = self.start_transform.location
        self.fixed_rotation = self.start_transform.rotation

        start_wp = mp.get_waypoint(self.fixed_spawn)
        end_wp = start_wp.next(self.arrival_distance)[0]
        self.end_location = end_wp.transform.location

        # spawn static work zones and background traffic ONCE
        self._spawn_static_work_zones(start_wp)
        self._spawn_npc_vehicles()    # NEW: moved from reset
        self._spawn_pedestrians()     # NEW: moved from reset

        if self.render:
            self._draw_path_on_road(start_wp, end_wp)

    # -----------------------
    # reset
    # -----------------------
    def reset(self, *, seed=None, options=None) -> Tuple[Dict[str, Any], Dict]:
        super().reset(seed=seed)
        self.attempt_count += 1
        self.current_step = 0
        self._apply_scenario_weather()

        # cleanup previous episode actors (ego + sensors only)
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
        self._last_lidar_points = None  # NEW

        # spawn ego vehicle
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

        # sensors: collision & lane
        col_bp = self.blueprint_library.find("sensor.other.collision")
        self.collision_sensor = self.world.spawn_actor(col_bp, carla.Transform(), attach_to=self.vehicle)
        self.collision_sensor.listen(lambda event: self.collision_hist.append(event))
        self.actor_list.append(self.collision_sensor)

        lane_bp = self.blueprint_library.find("sensor.other.lane_invasion")
        self.lane_sensor = self.world.spawn_actor(lane_bp, carla.Transform(), attach_to=self.vehicle)
        self.lane_sensor.listen(lambda event: self.lane_hist.append(event))
        self.actor_list.append(self.lane_sensor)

        # camera RGB: 800x600, FOV 90°, downsampled later to 84x84
        cam_bp = self.blueprint_library.find("sensor.camera.rgb")
        cam_bp.set_attribute("image_size_x", "800")
        cam_bp.set_attribute("image_size_y", "600")
        cam_bp.set_attribute("fov", "90")
        cam_transform = carla.Transform(
            carla.Location(x=1.5, y=0.0, z=1.6),
            carla.Rotation(pitch=-10),
        )
        self.camera = self.world.spawn_actor(cam_bp, cam_transform, attach_to=self.vehicle)
        self.camera.listen(self._on_cam_image)
        self.actor_list.append(self.camera)

        # LiDAR: spawn only if enabled
        if self.use_lidar:
            lidar_bp = self.blueprint_library.find("sensor.lidar.ray_cast")
            lidar_bp.set_attribute("channels", "32")
            lidar_bp.set_attribute("range", str(self.lidar_range))
            lidar_bp.set_attribute("rotation_frequency", str(1.0 / self.fixed_delta_seconds))
            lidar_bp.set_attribute("points_per_second", "56000")
            lidar_bp.set_attribute("upper_fov", "10")
            lidar_bp.set_attribute("lower_fov", "-30")
            lidar_transform = carla.Transform(carla.Location(x=0.0, y=0.0, z=2.0))
            self.lidar = self.world.spawn_actor(lidar_bp, lidar_transform, attach_to=self.vehicle)
            self.lidar.listen(self._on_lidar)
            self.actor_list.append(self.lidar)
        else:
            self.lidar = None
            self._last_lidar_points = None

        # recompute end
        mp = self.world.get_map()
        start_wp = mp.get_waypoint(self.fixed_spawn)
        end_wp = start_wp.next(self.arrival_distance)[0]
        self.end_location = end_wp.transform.location

        # NOTE: background traffic/pedestrians are now persistent, so we DO NOT
        # respawn them here anymore
        # (removed: self._spawn_npc_vehicles(); self._spawn_pedestrians())

        # let the world tick a few times so sensors/traffic produce data
        try:
            for _ in range(3):
                self.world.tick()
        except Exception as e:
            # don't hard-crash training if CARLA glitches here
            print(f"[CARLA-ENV] Exception during reset world.tick(): {e}")

        self.prev_dist_to_goal = self._distance(self.vehicle.get_location(), self.end_location)

        if self.render:
            self._update_spectator()

        obs = self._get_observation()
        return obs, {}

    # -----------------------
    # step
    # -----------------------
    def step(self, action: np.ndarray) -> Tuple[Dict[str, Any], float, bool, bool, Dict]:
        # If ego vehicle is missing or dead, recover by resetting the episode
        if self.vehicle is None or not getattr(self.vehicle, "is_alive", True):
            print("[CARLA-ENV] Ego vehicle invalid in step(), resetting episode.")
            obs, _ = self.reset()
            return obs, 0.0, False, True, {"error": "ego_invalid"}

        # smoothing
        alpha = 0.3
        action = np.clip(action, self.action_space.low, self.action_space.high)
        smoothed = alpha * action + (1 - alpha) * self.prev_action
        self.prev_action = smoothed.copy()
        throttle, steer = float(smoothed[0]), float(smoothed[1])

        try:
            # apply control + tick
            self.vehicle.apply_control(
                carla.VehicleControl(throttle=throttle, steer=steer, brake=0.0)
            )
            self.world.tick()
        except Exception as e:
            # Any CARLA RPC failure -> safely truncate and reset
            print(f"[CARLA-ENV] Exception during world.tick()/control: {e}. Resetting episode.")
            obs, _ = self.reset()
            return obs, 0.0, False, True, {"error": str(e)}

        self.current_step += 1

        # update spectator if requested
        if self.render:
            self._update_spectator()

        # build observation
        try:
            obs = self._get_observation()
        except Exception as e:
            print(f"[CARLA-ENV] Exception while building observation: {e}. Resetting episode.")
            obs, _ = self.reset()
            return obs, 0.0, False, True, {"error": f"obs_error: {e}"}

        # compute reward
        try:
            loc = self.vehicle.get_location()
        except Exception as e:
            print(f"[CARLA-ENV] Exception while reading vehicle location: {e}. Resetting episode.")
            obs, _ = self.reset()
            return obs, 0.0, False, True, {"error": f"loc_error: {e}"}

        current_dist = self._distance(loc, self.end_location)
        progress = (self.prev_dist_to_goal - current_dist) if self.prev_dist_to_goal is not None else 0.0
        self.prev_dist_to_goal = current_dist

        vel = self.vehicle.get_velocity()
        speed = math.sqrt(vel.x ** 2 + vel.y ** 2 + vel.z ** 2)

        # Base reward
        reward = 0.0
        reward += float(progress) * 10.0          # go forward
        reward += 0.05 * float(speed)            # encourage moving
        reward -= 0.02                           # time penalty
        reward -= abs(steer) * 0.01 * (1.0 + speed)  # discourage sharp steering

        # Smoothness
        steer_change = abs(smoothed[1] - getattr(self, "prev_steer", 0.0))
        reward -= steer_change * 0.02
        self.prev_steer = float(smoothed[1])

        terminated = False
        truncated = False

        # --- Highway code rules ---

        # 1) Speed limit: soft penalty above ~40 km/h, stronger above 50 km/h
        if speed > self.speed_limit_soft_mps:
            excess_soft = speed - self.speed_limit_soft_mps
            reward -= excess_soft * 0.5
        if speed > self.speed_limit_mps:
            excess_hard = speed - self.speed_limit_mps
            reward -= excess_hard * 1.0  # stronger penalty

        # 2) Stop signs: must almost stop when passing close to a stop sign
        for ss in self.stop_signs:
            try:
                d_ss = self._distance(loc, ss.get_transform().location)
            except Exception:
                continue
            if d_ss < 8.0:  # near a stop sign
                if speed > self.stop_speed_threshold:
                    # ran a stop sign
                    reward -= 40.0
                    terminated = True
                break

        # 3) Traffic lights: must stop at red/yellow before the light
        try:
            tl = self.vehicle.get_traffic_light()
        except Exception:
            tl = None

        if tl is not None:
            try:
                tl_state = tl.get_state()
                tl_loc = tl.get_transform().location
                dist_tl = self._distance(loc, tl_loc)
            except Exception:
                tl_state = None
                dist_tl = float("inf")

            if dist_tl < self.stop_check_distance and tl_state is not None:
                # Approaching a signalised junction
                if tl_state in (carla.TrafficLightState.Red, carla.TrafficLightState.Yellow):
                    # Must be slow when close, and fully stopped near the line
                    if dist_tl < 15.0 and speed > 2.0:
                        reward -= (speed - 2.0) * 0.5  # encourage braking
                    if dist_tl < 8.0 and speed > self.stop_speed_threshold:
                        # passed red/yellow without stopping
                        reward -= 60.0
                        terminated = True

        # --- Existing termination logic ---

        if len(self.lane_hist) > 0:
            reward -= 8.0
            self.lane_hist = []
            terminated = True

        if len(self.collision_hist) > 0:
            # already covers cars, pedestrians, obstacles
            reward -= 50.0
            terminated = True

        if speed < self.idle_speed_threshold:
            self.idle_steps += 1
        else:
            self.idle_steps = 0

        if self.idle_steps > self.idle_max_steps:
            reward -= 8.0
            terminated = True

        if current_dist < 2.0:
            reward += 100.0
            terminated = True
            self.success_count += 1

        if self.current_step >= self.max_steps and not terminated:
            truncated = True

        # Lane keeping: stronger penalty for leaving the center of the lane
        try:
            wp = self.world.get_map().get_waypoint(loc)
            lane_center = wp.transform.location
            dist_from_center = self._distance(loc, lane_center)
            reward -= dist_from_center * 3.0  # was 2.0, encourage staying in lane
        except Exception:
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
    # LiDAR callback (NEW)
    # -----------------------
    def _on_lidar(self, point_cloud: carla.LidarMeasurement) -> None:
        """Store latest LiDAR point cloud."""
        try:
            pts = np.frombuffer(point_cloud.raw_data, dtype=np.float32)
            pts = pts.reshape(-1, 4)[:, :3]  # x, y, z
            self._last_lidar_points = pts
        except Exception:
            self._last_lidar_points = None

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
            if cam_w != self.image_w or cam_h != self.image_h:
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

        # LiDAR BEV grid
        lidar_bev = self._build_lidar_bev()  # NEW

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

        return {"image": stacked, "lidar": lidar_bev, "state": state}

    # -----------------------
    # LiDAR BEV helper
    # -----------------------
    def _build_lidar_bev(self) -> np.ndarray:
        """Project LiDAR points into a simple front-facing BEV occupancy grid."""
        # If LiDAR is disabled, always return empty grid
        if not self.use_lidar:
            return np.zeros((self.lidar_bev_size, self.lidar_bev_size, 1), dtype=np.float32)

        grid = np.zeros((self.lidar_bev_size, self.lidar_bev_size), dtype=np.float32)
        pts = self._last_lidar_points
        if pts is None or len(pts) == 0:
            return grid[..., None]

        # Only points in front of vehicle (x >= 0) within range
        x = pts[:, 0]
        y = pts[:, 1]
        dist = np.sqrt(x * x + y * y)
        mask = (x >= 0.0) & (dist <= self.lidar_range)
        x = x[mask]
        y = y[mask]

        if x.size == 0:
            return grid[..., None]

        # Map x in [0, range] -> row [0, size)
        # Map y in [-range, range] -> col [0, size)
        rn = self.lidar_bev_size
        cn = self.lidar_bev_size
        row = (x / self.lidar_range * (rn - 1)).astype(int)
        col = ((y + self.lidar_range) / (2 * self.lidar_range) * (cn - 1)).astype(int)

        row = np.clip(row, 0, rn - 1)
        col = np.clip(col, 0, cn - 1)

        grid[row, col] = 1.0  # simple occupancy
        return grid[..., None]

    # -----------------------
    # scenario / traffic helpers (NEW)
    # -----------------------
    def _apply_scenario_weather(self) -> None:
        if self.scenario == "clear":
            weather = carla.WeatherParameters.ClearNoon
        elif self.scenario == "wet":
            weather = carla.WeatherParameters.WetCloudyNoon
        elif self.scenario == "night":
            weather = carla.WeatherParameters.SoftRainNight
        else:
            weather = random.choice(
                [
                    carla.WeatherParameters.ClearNoon,
                    carla.WeatherParameters.WetCloudyNoon,
                    carla.WeatherParameters.SoftRainNight,
                ]
            )
        self.world.set_weather(weather)

    def _spawn_npc_vehicles(self) -> None:
        """Spawn ~num_npc_vehicles traffic vehicles controlled by TrafficManager."""
        try:
            spawn_points = list(self.world.get_map().get_spawn_points())
            random.shuffle(spawn_points)
            ego_loc = self.fixed_spawn

            # avoid spawning too close to ego
            spawn_points = [
                sp for sp in spawn_points
                if sp.location.distance(ego_loc) > 10.0
            ]

            bp_candidates = self.blueprint_library.filter("vehicle.*")
            count = min(self.num_npc_vehicles, len(spawn_points))
            for i in range(count):
                bp = random.choice(bp_candidates)
                bp.set_attribute("role_name", "autopilot")
                vehicle = self.world.try_spawn_actor(bp, spawn_points[i])
                if vehicle:
                    vehicle.set_autopilot(True, self.traffic_manager.get_port())
                    self.background_actors.append(vehicle)  # CHANGED: persistent
        except Exception:
            pass

    def _spawn_pedestrians(self) -> None:
        """Spawn ~num_pedestrians walkers with AI controllers."""
        try:
            walker_bps = self.blueprint_library.filter("walker.pedestrian.*")
            controller_bp = self.blueprint_library.find("controller.ai.walker")

            spawn_transforms = []
            for _ in range(self.num_pedestrians):
                loc = self.world.get_random_location_from_navigation()
                if loc is not None:
                    spawn_transforms.append(carla.Transform(loc))

            walkers = []
            for tr in spawn_transforms:
                bp = random.choice(walker_bps)
                if bp.has_attribute("is_invincible"):
                    bp.set_attribute("is_invincible", "false")
                w = self.world.try_spawn_actor(bp, tr)
                if w:
                    walkers.append(w)
                    self.background_actors.append(w)  # CHANGED: persistent

            for w in walkers:
                controller = self.world.spawn_actor(controller_bp, carla.Transform(), attach_to=w)
                self.background_actors.append(controller)  # CHANGED: persistent
                controller.start()
                controller.go_to_location(self.world.get_random_location_from_navigation())
                controller.set_max_speed(random.uniform(1.0, 1.5))
        except Exception:
            pass

    def _spawn_static_work_zones(self, start_wp) -> None:
        """Spawn a couple of static props to represent work zones along the route."""
        try:
            distances = [20.0, 35.0]
            patterns = ["static.prop.*cone*", "static.prop.*barrier*"]
            for d, pattern in zip(distances, patterns):
                next_wps = start_wp.next(d)
                if not next_wps:
                    continue
                wp = next_wps[0]
                loc = wp.transform.location + carla.Location(y=1.5)
                tr = carla.Transform(loc, wp.transform.rotation)
                bps = self.blueprint_library.filter(pattern)
                if not bps:
                    continue
                bp = bps[0]
                actor = self.world.try_spawn_actor(bp, tr)
                if actor:
                    self.work_zone_actors.append(actor)
        except Exception:
            pass

    # -----------------------
    # utils
    # -----------------------
    def _distance(self, a: carla.Location, b: carla.Location) -> float:
        return math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2 + (a.z - b.z) ** 2)

    # -----------------------
    # spectator update (NEW)
    # -----------------------
    def _update_spectator(self):
        # if render is False, never touch spectator (fully headless)
        if not self.render:
            return
        try:
            if self.vehicle is None or not getattr(self.vehicle, "is_alive", True):
                return
            spectator = self.world.get_spectator()
            vt = self.vehicle.get_transform()
            fwd = vt.get_forward_vector()
            pos = vt.location - fwd * 7.0
            pos.z += 3.0
            spectator.set_transform(
                carla.Transform(pos, carla.Rotation(pitch=-10, yaw=vt.rotation.yaw))
            )
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
            if actor is not None:
                try:
                    actor.destroy()
                except RuntimeError:
                    # l'actor a déjà été détruit ou n'existe plus → ok
                    pass
        self.actor_list = []
        self.camera = None
        self.collision_sensor = None
        self.lane_sensor = None
        self.vehicle = None
        self.lidar = None
        self._last_lidar_points = None
        # tick uniquement si synchronisé
        if hasattr(self.world, 'tick'):
            try:
                self.world.tick()
            except Exception:
                pass

    def close(self):
        try:
            # destroy per-episode actors
            self._clean_actors()
            # destroy static work zones
            for actor in self.work_zone_actors:
                try:
                    if actor is not None and getattr(actor, "is_alive", True):
                        actor.destroy()
                except Exception:
                    pass
            self.work_zone_actors = []
            # destroy background traffic + pedestrians + controllers
            for actor in self.background_actors:
                try:
                    if actor is not None and getattr(actor, "is_alive", True):
                        actor.destroy()
                except Exception:
                    pass
            self.background_actors = []
        finally:
            try:
                settings = self.world.get_settings()
                settings.synchronous_mode = False
                settings.fixed_delta_seconds = None
                self.world.apply_settings(settings)
            except Exception:
                pass
            try:
                self.traffic_manager.set_synchronous_mode(False)
            except Exception:
                pass
