from kafka import KafkaConsumer
from hdfs import InsecureClient
import uuid
# Kafka Consumer
consumer = KafkaConsumer('video_topic', bootstrap_servers='localhost:9092')

# HDFS Client
hdfs_client = InsecureClient('http://localhost:9870', user='hdfs')

# HDFS target directory
target_dir = '/student_videos'

# Check and create target directory if it doesn't exist
if not hdfs_client.content(target_dir, strict=False):
    print(f"Creating directory: {target_dir}")
    hdfs_client.makedirs(target_dir)
    
    # Set permission to 777
    hdfs_client.chmod(target_dir, 777)

def save_to_hdfs(content, filename):
    print(f"Saving {filename} to HDFS")
    with hdfs_client.write(f'{target_dir}/{filename}', overwrite=True) as f:
        f.write(content)

for message in consumer:
    video_content = message.value

    # Define how to set unique filenames
    video_filename = f'video-{uuid.uuid4()}.mp4'
    save_to_hdfs(video_content, video_filename)
