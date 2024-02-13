import os
from pyspark.sql import SparkSession
from hdfs import InsecureClient
import cv2
import tempfile
import uuid

def save_image_to_hdfs(image, target_dir, operation_type, filename_prefix='result'):
    """
    Save the processed image to HDFS.

    Parameters:
    - image: The processed image (NumPy array).
    - target_dir: The target directory in HDFS.
    - operation_type: The type of operation applied to the image (e.g., rotation, cropping, enhancement).
    - filename_prefix: Prefix for the saved filename (default is 'result').

    Returns:
    - saved_filename: The filename under which the image is saved.
    """
    # HDFS Client
    hdfs_client = InsecureClient('http://localhost:9870', user='hdfs')

    # Check and create target directory if it doesn't exist
    if not hdfs_client.content(target_dir, strict=False):
        print(f"Creating directory: {target_dir}")
        hdfs_client.makedirs(target_dir, permission=777)

    # Define the filename based on the operation type
    saved_filename = f'{filename_prefix}_{operation_type}_{uuid.uuid4()}.png'

    # Save the image locally
    local_path = os.path.join(tempfile.gettempdir(), saved_filename)
    cv2.imwrite(local_path, image)

    # Write the image to HDFS
    with hdfs_client.write(f'{target_dir}/{saved_filename}', overwrite=True) as hdfs_file:
        with open(local_path, 'rb') as local_file:
            hdfs_file.write(local_file.read())

    return saved_filename