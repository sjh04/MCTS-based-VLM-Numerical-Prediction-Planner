import sys
import os
sys.path.append('/home/ubuntu/dockerCarla/PythonAPI/carla/dist/carla-0.9.13-py3.7-linux-x86_64.egg')

import carla
import numpy as np
import cv2
from PIL import Image
import io
import queue
import time
import math
from ..main import parse_arguments

class CameraManager:
    """Camera manager class to handle camera sensors and image processing"""
    def __init__(self):
        self.images = {}
        self._queues = {}
        self.sensors = {}

    def add_sensor(self, name, sensor):
        """Add a sensor to the manager"""
        self.sensors[name] = sensor
        self._queues[name] = queue.Queue()
        sensor.listen(lambda data: self._queues[name].put(data))

    def get_image(self, name, timeout=5.0):
        """Get latest image from the named sensor"""
        try:
            return self._queues[name].get(timeout=timeout)
        except queue.Empty:
            return None

    def destroy(self):
        """Destroy all sensors"""
        for sensor in self.sensors.values():
            if sensor is not None:
                sensor.stop()
                sensor.destroy()
        self.sensors.clear()
        self._queues.clear()

def setup_cameras(args, vehicle, world):
    """
    Set up the vehicle's camera system
    Includes: front view, rear view and side view cameras
    """
    # Use camera parameters from args
    camera_width = args.camera_width
    camera_height = args.camera_height
    fov = args.camera_fov
    
    # Create camera manager
    camera_manager = CameraManager()
    
    # Get the blueprint library
    blueprint_library = world.get_blueprint_library()
    
    # Creating a Camera Blueprint
    camera_bp = blueprint_library.find(args.camera_type)
    camera_bp.set_attribute('image_size_x', str(camera_width))
    camera_bp.set_attribute('image_size_y', str(camera_height))
    camera_bp.set_attribute('fov', str(fov))
    
    # Front view camera
    front_camera = carla.Transform(carla.Location(x=2.0, z=1.4))
    front_sensor = world.spawn_actor(camera_bp, front_camera, attach_to=vehicle)
    camera_manager.add_sensor('front', front_sensor)
    
    # Front left corner camera 
    front_left_camera = carla.Transform(
        carla.Location(x=1.5, y=-1.0, z=1.4),
        carla.Rotation(yaw=-45)
    )
    front_left_sensor = world.spawn_actor(camera_bp, front_left_camera, attach_to=vehicle)
    camera_manager.add_sensor('front_left', front_left_sensor)
    
    # Front right corner camera
    front_right_camera = carla.Transform(
        carla.Location(x=1.5, y=1.0, z=1.4),
        carla.Rotation(yaw=45)
    )
    front_right_sensor = world.spawn_actor(camera_bp, front_right_camera, attach_to=vehicle)
    camera_manager.add_sensor('front_right', front_right_sensor)
    
    # Back view camera
    back_camera = carla.Transform(
        carla.Location(x=-2.0, z=1.4),
        carla.Rotation(yaw=180)
    )
    back_sensor = world.spawn_actor(camera_bp, back_camera, attach_to=vehicle)
    camera_manager.add_sensor('back', back_sensor)
    
    # Back left corner camera
    back_left_camera = carla.Transform(
        carla.Location(x=-1.5, y=-1.0, z=1.4),
        carla.Rotation(yaw=-135)
    )
    back_left_sensor = world.spawn_actor(camera_bp, back_left_camera, attach_to=vehicle)
    camera_manager.add_sensor('back_left', back_left_sensor)
    
    # Back right corner camera
    back_right_camera = carla.Transform(
        carla.Location(x=-1.5, y=1.0, z=1.4),
        carla.Rotation(yaw=135)
    )
    back_right_sensor = world.spawn_actor(camera_bp, back_right_camera, attach_to=vehicle)
    camera_manager.add_sensor('back_right', back_right_sensor)
    
    return camera_manager

def destroy_cameras(camera_manager):
    """
    Destroy all cameras
    """
    camera_manager.destroy()

def process_camera_data(image):
    """
    Process camera data, convert to the format suitable for Qwen2.5-vl-3b model
    """
    # Convert raw data to numpy array
    array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
    array = np.reshape(array, (image.height, image.width, 4))
    
    # Convert BGRA to RGB
    array = array[:, :, :3]
    array = array[:, :, ::-1]  # BGR to RGB
    
    # Convert to PIL Image
    pil_image = Image.fromarray(array)
    
    # Adjust image size to the standard size required by the model
    pil_image = pil_image.resize((448, 448))
    
    return pil_image

def get_all_camera_images(camera_manager):
    """
    Get all camera images
    Return a dictionary containing all processed images from the cameras
    """
    camera_images = {}
    
    for name in camera_manager.sensors.keys():
        # Get camera data
        image = camera_manager.get_image(name)
        if image is not None:
            # Process data
            processed_image = process_camera_data(image)
            camera_images[name] = processed_image
    
    return camera_images

def create_camera_description(camera_images):
    """
    Create a description of the camera images, for Qwen2.5-vl-3b model input
    """
    description = "Vehicle surroundings images:\n"
    description += "- Front camera: Shows the road and traffic ahead\n"
    description += "- Front left corner camera: Shows the road and traffic on the left\n"
    description += "- Front right corner camera: Shows the road and traffic on the right\n"
    description += "- Back camera: Shows the road and traffic behind\n"
    description += "- Back left corner camera: Shows the road and traffic on the left\n"
    description += "- Back right corner camera: Shows the road and traffic on the right"
    
    return description

def get_vehicle_state(vehicle):
    """
    Get the vehicle state with relevant driving metrics
    """
    # Get velocity (in m/s)
    velocity = vehicle.get_velocity()
    speed = 3.6 * math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)  # km/h
    
    # Get current control state
    control = vehicle.get_control()
    
    # Get vehicle location and transform
    transform = vehicle.get_transform()
    location = transform.location
    rotation = transform.rotation
    
    # Get acceleration
    acceleration = vehicle.get_acceleration()
    
    # Get angular velocity
    angular_velocity = vehicle.get_angular_velocity()
    
    # Create comprehensive state dictionary
    state = {
        "speed": speed,  # km/h
        "steering": control.steer,  # -1 to 1
        "throttle": control.throttle,  # 0 to 1
        "brake": control.brake,  # 0 to 1
        "handbrake": control.hand_brake,  # boolean
        "gear": control.gear,
        "location": {
            "x": location.x,
            "y": location.y,
            "z": location.z
        },
        "rotation": {
            "pitch": rotation.pitch,
            "yaw": rotation.yaw,
            "roll": rotation.roll
        },
        "acceleration": {
            "x": acceleration.x,
            "y": acceleration.y,
            "z": acceleration.z
        },
        "angular_velocity": {
            "x": angular_velocity.x,
            "y": angular_velocity.y,
            "z": angular_velocity.z
        }
    }
    
    return state

def get_navigation_instructions(vehicle, destination, map_data):
    """
    Get the navigation instructions to reach a destination
    
    Args:
        vehicle: The CARLA vehicle object
        destination: Target location coordinates (carla.Location)
        map_data: CARLA map data
    
    Returns:
        Dictionary containing navigation information
    """
    # Get current vehicle location
    current_location = vehicle.get_location()
    
    # Calculate direct distance to destination
    direct_distance = current_location.distance(destination)
    
    # Get waypoint for current location
    waypoint = map_data.get_waypoint(current_location)
    
    # Calculate direction vector to destination
    direction_vector = destination - current_location
    
    # Get vehicle's forward vector
    forward_vector = vehicle.get_transform().get_forward_vector()
    
    # Calculate angle between forward vector and direction to destination
    dot_product = forward_vector.x * direction_vector.x + forward_vector.y * direction_vector.y
    magnitude = math.sqrt(direction_vector.x**2 + direction_vector.y**2) * math.sqrt(forward_vector.x**2 + forward_vector.y**2)
    angle = math.acos(max(min(dot_product / magnitude, 1.0), -1.0)) * 180 / math.pi
    
    # Determine if destination is to the left or right
    cross_product = forward_vector.x * direction_vector.y - forward_vector.y * direction_vector.x
    turn_direction = "left" if cross_product > 0 else "right"
    
    # Generate navigation instructions
    instructions = {
        "distance_to_destination": direct_distance,
        "current_road_id": waypoint.road_id,
        "current_lane_id": waypoint.lane_id,
        "angle_to_destination": angle,
        "turn_direction": turn_direction,
        "is_junction": waypoint.is_junction
    }
    
    return instructions

