from kafka import KafkaConsumer
from hdfs import InsecureClient
import uuid
# Kafka Consumer
consumer = KafkaConsumer('video_topic', bootstrap_servers='localhost:9092')
# HDFS Client
hdfs_client = InsecureClient('http://localhost:9870', user='hdfs')
# Répertoire cible dans HDFS
target_dir = '/student_videos'
# Vérifie et crée le répertoire cible s'il n'existe pas
if not hdfs_client.content(target_dir, strict=False):
    print(f"Creating directory: {target_dir}")
    hdfs_client.makedirs(target_dir) 
    # Définit les permissions à 777
    hdfs_client.chmod(target_dir, 777)
# Fonction pour enregistrer le contenu dans HDFS
def save_to_hdfs(content, filename):
    print(f"Saving {filename} to HDFS")
    with hdfs_client.write(f'{target_dir}/{filename}', overwrite=True) as f:
        f.write(content)
# Consommer les messages de Kafka et enregistrer les vidéos dans HDFS
for message in consumer:
    video_content = message.value
    # Enregistrer le contenu dans HDFS
    video_filename = f'video-{uuid.uuid4()}.mp4'
    save_to_hdfs(video_content, video_filename)
