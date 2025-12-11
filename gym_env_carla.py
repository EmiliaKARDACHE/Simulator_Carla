import carla
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import math
import cv2
import time


class CarlaRLEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}

    def __init__(self):
        super().__init__()

        # Connexion serveur CARLA
        self.client = carla.Client("localhost", 2000)
        self.client.set_timeout(10.0)

        # Charger map Town05
        self.client.load_world("Town05")
        self.world = self.client.get_world()
        self.blueprint_library = self.world.get_blueprint_library()

        # Véhicule
        self.vehicle_bp = self.blueprint_library.filter('model3')[0]

        # Acteurs
        self.actor_list = []
        self.vehicle = None
        self.collision_sensor = None
        self.lane_sensor = None
        self.camera = None

        # Historiques
        self.collision_hist = []
        self.lane_hist = []
        self.cam_image = np.zeros((84, 84, 3), dtype=np.uint8)

        # Action et observation
        self.action_space = spaces.Box(
            low=np.array([0.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0], dtype=np.float32),
            dtype=np.float32
        )
        # Observation enrichie : 3 (RGB) + 3 (masques)
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(84, 84, 6), dtype=np.uint8
        )

        # Distance d'arrivée
        self.arrival_distance = 50.0

        # Spawn fixe
        map = self.world.get_map()
        waypoint = map.get_waypoint(carla.Location(x=230, y=180, z=0))
        self.start_transform = waypoint.transform
        self.fixed_spawn = self.start_transform.location
        self.fixed_rotation = self.start_transform.rotation

        # Objectif sur la route
        start_wp = map.get_waypoint(self.fixed_spawn)
        end_wp = start_wp.next(self.arrival_distance)[0]
        self.end_location = end_wp.transform.location

        # Tracé visuel
        self._draw_path_on_road(start_wp, end_wp)

    # ----------------------------------------------------------------------
    # RESET
    # ----------------------------------------------------------------------
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        self._clean_actors()
        self.collision_hist = []
        self.lane_hist = []

        self.start_transform = carla.Transform(self.fixed_spawn, self.fixed_rotation)

        # Recalcul du waypoint d'arrivée
        map = self.world.get_map()
        start_wp = map.get_waypoint(self.fixed_spawn)
        end_wp = start_wp.next(self.arrival_distance)[0]
        self.end_location = end_wp.transform.location

        # Tracé visuel
        self._draw_path_on_road(start_wp, end_wp)

        # Spawn véhicule
        base_z = self.start_transform.location.z
        self.vehicle = None
        for i in range(5):
            self.start_transform.location.z = base_z + i * 0.2
            self.vehicle = self.world.try_spawn_actor(self.vehicle_bp, self.start_transform)
            if self.vehicle:
                break
        if not self.vehicle:
            raise RuntimeError("Impossible de spawn la voiture.")
        self.actor_list.append(self.vehicle)

        # Spectateur
        self._update_spectator()

        # Sensors
        col_bp = self.blueprint_library.find("sensor.other.collision")
        self.collision_sensor = self.world.spawn_actor(col_bp, carla.Transform(), attach_to=self.vehicle)
        self.collision_sensor.listen(lambda e: self.collision_hist.append(e))
        self.actor_list.append(self.collision_sensor)

        lane_bp = self.blueprint_library.find("sensor.other.lane_invasion")
        self.lane_sensor = self.world.spawn_actor(lane_bp, carla.Transform(), attach_to=self.vehicle)
        self.lane_sensor.listen(lambda e: self.lane_hist.append(e))
        self.actor_list.append(self.lane_sensor)

        cam_bp = self.blueprint_library.find("sensor.camera.rgb")
        cam_bp.set_attribute("image_size_x", "640")
        cam_bp.set_attribute("image_size_y", "480")
        cam_bp.set_attribute("fov", "110")
        cam_transform = carla.Transform(carla.Location(x=1.5, y=0.0, z=1.6), carla.Rotation(pitch=-10))
        self.camera = self.world.spawn_actor(cam_bp, cam_transform, attach_to=self.vehicle)
        self.camera.listen(self._on_cam_image)
        self.actor_list.append(self.camera)

        self._draw_marker(self.end_location, carla.Color(255, 0, 0))

        # Attente image caméra
        timeout = time.time() + 5.0
        while self.cam_image is None and time.time() < timeout:
            self.world.tick()
            time.sleep(0.05)
        if self.cam_image is None:
            self.cam_image = np.zeros((84, 84, 3), dtype=np.uint8)

        print("Reset done, vehicle spawned at:", self.fixed_spawn)
        return self._get_observation(), {}

    # ----------------------------------------------------------------------
    # STEP
    # ----------------------------------------------------------------------
    def step(self, action):
        throttle = float(action[0])
        steer = float(action[1])

        self.vehicle.apply_control(carla.VehicleControl(throttle=throttle, steer=steer, brake=0.0))
        self._update_spectator()

        reward = 0.0
        terminated = False
        truncated = False
        loc = self.vehicle.get_location()

        dist_from_start = self._distance(loc, self.start_transform.location)
        reward += dist_from_start * 0.02

        if loc.z < 0.5:
            reward -= 0.1
        reward -= abs(steer) * 0.03

        if len(self.lane_hist) > 0:
            reward -= 3.0
            self.lane_hist = []

        if len(self.collision_hist) > 0:
            reward -= 200
            terminated = True

        dist_to_goal = self._distance(loc, self.end_location)
        if dist_to_goal < 2.0:
            reward += 500
            terminated = True

        return self._get_observation(), reward, terminated, truncated, {}

    # ----------------------------------------------------------------------
    # CAMERA CALLBACK
    # ----------------------------------------------------------------------
    def _on_cam_image(self, image):
        array = image.to_array()
        rgb = array[:, :, :3]
        rgb = cv2.resize(rgb, (84, 84))
        self.cam_image = rgb

    # ----------------------------------------------------------------------
    # OBSERVATION ENRICHIE
    # ----------------------------------------------------------------------
    def _get_observation(self):
        rgb = self.cam_image.copy()

        # Lane mask
        lane_mask = np.zeros((84, 84), dtype=np.uint8)
        if len(self.lane_hist) > 0:
            lane_mask[:, :] = 255

        # Trottoir mask
        trottoir_mask = np.zeros((84, 84), dtype=np.uint8)
        if self.vehicle.get_location().z < 0.5:
            trottoir_mask[:, :] = 255

        # Obstacles mask
        obstacle_mask = np.zeros((84, 84), dtype=np.uint8)
        for actor in self.world.get_actors():
            if actor.type_id.startswith("vehicle.") and actor.id != self.vehicle.id:
                loc = actor.get_location()
                x_img = int(84 * (loc.x - self.vehicle.get_location().x + 10) / 20)
                y_img = int(84 * (loc.y - self.vehicle.get_location().y + 10) / 20)
                if 0 <= x_img < 84 and 0 <= y_img < 84:
                    obstacle_mask[y_img, x_img] = 255

        obs = np.dstack([rgb, lane_mask, trottoir_mask, obstacle_mask])
        return obs

    # ----------------------------------------------------------------------
    # UTILS
    # ----------------------------------------------------------------------
    def _distance(self, a, b):
        return math.sqrt((a.x - b.x)**2 + (a.y - b.y)**2 + (a.z - b.z)**2)

    def _draw_marker(self, location, color):
        self.world.debug.draw_box(
            box=carla.BoundingBox(location, carla.Vector3D(0.5, 0.5, 0.5)),
            rotation=carla.Rotation(),
            color=color,
            thickness=0.2,
            life_time=0
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
                life_time=0
            )

    def _update_spectator(self):
        spectator = self.world.get_spectator()
        vt = self.vehicle.get_transform()
        fwd = vt.get_forward_vector()
        pos = vt.location - fwd * 7.0
        pos.z += 3.0
        spectator.set_transform(carla.Transform(pos, carla.Rotation(pitch=-10, yaw=vt.rotation.yaw)))

    def _clean_actors(self):
        all_actors = self.world.get_actors()
        for actor in all_actors:
            if actor.type_id.startswith("vehicle.") or actor.type_id.startswith("sensor."):
                try:
                    actor.destroy()
                except:
                    pass
        self.actor_list = []
        self.camera = None
        self.collision_sensor = None
        self.lane_sensor = None
        self.vehicle = None
        self.world.tick()
