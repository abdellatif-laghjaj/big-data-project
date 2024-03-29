from kafka import KafkaProducer
import requests
# Liste des liens vidéo
videos = [
    "https://www.pexels.com/download/video/5198159",
    "https://www.pexels.com/download/video/5198168/",
    "https://www.pexels.com/download/video/8198503/",
    "https://www.pexels.com/download/video/5427882/",
    "https://www.pexels.com/download/video/8419341/",
    "https://www.pexels.com/download/video/8198511/",
]
# Initialisation du producteur Kafka
producer = KafkaProducer(bootstrap_servers='localhost:9092', max_request_size=115343360)
# Fonction pour télécharger et envoyer une vidéo à Kafka
def download_and_send(url):
    try:
        print(f"Downloading video from {url}")
        response = requests.get(url)
        if response.status_code == 200:
            print(f"Sending video to Kafka topic")  
            # Envoie le contenu de la vidéo à Kafka      
            producer.send('video_topic', response.content)
            producer.send('video_topic', 'test'.encode('utf-8'))
            # Forcer l'envoi de tous les messages
            producer.flush()
        else:
            print(f"Failed to download video from {url}, status code: {response.status_code}")
    except Exception as e:
        print(f"Error downloading/sending video: {e}")
# Télécharger et envoyer les vidéos
for video in videos:
    download_and_send(video)