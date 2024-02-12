from kafka import KafkaProducer
import requests

# Videos links list
videos = [
    #"https://v1.cdnpk.net/videvo_files/video/premium/getty_108/large_watermarked/istock-993951822_preview.mp4",
    "https://v1.cdnpk.net/videvo_files/video/premium/getty_132/large_watermarked/istock-962828842_preview.mp4",
    #"https://dt36.dlsnap05.xyz/download?file=NjRhYzVhZjZkZjc3Zjk0MzA0YzdkYWQ5M2NlNjg4Y2UxYzE5ZGUzYjZkMTFjYWJhYjA3NzU1Nzc4ZmM3YjhkMV8xMDgwcC5tcDTimK95dDVzLmlvLVNjaG9vbCBSdWxlcyAmIFBvc2l0aXZlIEJlaGF2aW9yIHwgR29vZCBhbmQgQmFkIEV4YW1wbGVz4pivMTA4MHA",
]
# Kafka Producer
producer = KafkaProducer(bootstrap_servers='localhost:9092', max_request_size=115343360)

def download_and_send(url):
    try:
        print(f"Downloading video from {url}")
        response = requests.get(url)

        if response.status_code == 200:
            print(f"Sending video to Kafka topic")        
            producer.send('video_topic', response.content)
            producer.send('video_topic', 'test'.encode('utf-8'))

            # Force sending of all messages
            producer.flush()
        else:
            print(f"Failed to download video from {url}, status code: {response.status_code}")
    
    except Exception as e:
        print(f"Error downloading/sending video: {e}")

# Download and send videos
for video in videos:
    download_and_send(video)