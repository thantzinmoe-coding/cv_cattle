import requests
from pathlib import Path


def main():
    source = Path("dataset/raw/CattleLameness/Data/Lame/L (1).mp4")
    if not source.exists():
        raise FileNotFoundError(f"API test video not found: {source}")
    with source.open("rb") as video:
        response = requests.post(
            "http://127.0.0.1:8000/predict/image",
            files={"file": (source.name, video, "video/mp4")},
            timeout=600,
        )
    print(response.status_code)
    print(response.json())


if __name__ == "__main__":
    main()
