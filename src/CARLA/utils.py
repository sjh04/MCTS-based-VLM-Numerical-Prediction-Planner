import carla
import numpy as np
import cv2
from PIL import Image
import io

def setup_cameras(vehicle):
    """
    Set up the vehicle's camera system
    Includes: front view, rear view and side view cameras
    """
    # Camera parameter settings
    camera_width = 800
    camera_height = 600
    fov = 90
    
    # Creating a Camera Blueprint
    camera_bp = carla.Blueprint('sensor.camera.rgb')
    camera_bp.set_attribute('image_size_x', str(camera_width))
    camera_bp.set_attribute('image_size_y', str(camera_height))
    camera_bp.set_attribute('fov', str(fov))
    
    # Front view camera
    front_camera = carla.Transform(carla.Location(x=2.0, z=1.4))
    front_sensor = carla.SpawnActor(camera_bp, front_camera, attach_to=vehicle)
    
    # Front left corner camera 
    front_left_camera = carla.Transform(
        carla.Location(x=1.5, y=-1.0, z=1.4),
        carla.Rotation(yaw=-45)
    )
    front_left_sensor = carla.SpawnActor(camera_bp, front_left_camera, attach_to=vehicle)
    
    # Front right corner camera
    front_right_camera = carla.Transform(
        carla.Location(x=1.5, y=1.0, z=1.4),
        carla.Rotation(yaw=45)
    )
    front_right_sensor = carla.SpawnActor(camera_bp, front_right_camera, attach_to=vehicle)
    
    # Back view camera
    back_camera = carla.Transform(
        carla.Location(x=-2.0, z=1.4),
        carla.Rotation(yaw=180)
    )
    back_sensor = carla.SpawnActor(camera_bp, back_camera, attach_to=vehicle)
    
    # Back left corner camera
    back_left_camera = carla.Transform(
        carla.Location(x=-1.5, y=-1.0, z=1.4),
        carla.Rotation(yaw=-135)
    )
    back_left_sensor = carla.SpawnActor(camera_bp, back_left_camera, attach_to=vehicle)
    
    # Back right corner camera
    back_right_camera = carla.Transform(
        carla.Location(x=-1.5, y=1.0, z=1.4),
        carla.Rotation(yaw=135)
    )
    back_right_sensor = carla.SpawnActor(camera_bp, back_right_camera, attach_to=vehicle)
    
    return {
        'front': front_sensor,
        'front_left': front_left_sensor,
        'front_right': front_right_sensor,
        'back': back_sensor,
        'back_left': back_left_sensor,
        'back_right': back_right_sensor
    }

def destroy_cameras(cameras):
    """
    Destroy all cameras
    """
    for camera in cameras.values():
        camera.destroy()

def process_camera_data(camera_data):
    """
    Process camera data, convert to the format suitable for Qwen2.5-vl-3b model
    """
    # Convert raw data to numpy array
    array = np.frombuffer(camera_data.raw_data, dtype=np.dtype("uint8"))
    array = np.reshape(array, (camera_data.height, camera_data.width, 4))
    
    # Convert to RGB format (remove alpha channel)
    array = array[:, :, :3]
    
    # Convert to PIL Image
    image = Image.fromarray(array)
    
    # Adjust image size to the standard size required by the model (if needed)
    # Qwen2.5-vl-3b usually uses 448x448 input size
    image = image.resize((448, 448))
    
    return image

def get_all_camera_images(cameras):
    """
    Get all camera images
    Return a dictionary containing all processed images from the cameras
    """
    camera_images = {}
    
    for name, camera in cameras.items():
        # Get camera data
        camera_data = camera.get()
        # Process data
        processed_image = process_camera_data(camera_data)
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

