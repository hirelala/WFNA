import requests
import json
import nltk
import spacy
from pathlib import Path
import sys

nltk.download('punkt')
nltk.download('stopwords')
spacy.load('en_core_web_sm')

novels_path = Path('novels')


def download_novels():
    print('Downloading novels...')
    with open("novels.json", "r") as f:
        novels = json.load(f)

    for novel in novels:
        novel_dir = novels_path.joinpath(novel['name'])
        if not novel_dir.exists():
            novel_dir.mkdir(parents=True, exist_ok=True)

        novel_path = novel_dir.joinpath(f"{novel['name']}.txt")
        if novel_path.exists():
            continue

        response = requests.get(novel['url'])
        with open(novel_path, "wb") as f:
            f.write(response.content)
            print(f"Downloaded {novel['name']}")
    print('All done!')


def clean():
    print('Cleaning novels...')
    with open("novels.json", "r") as f:
        novels = json.load(f)

    for novel in novels:
        novel_dir = novels_path.joinpath(novel['name'])
        novel_path = novel_dir.joinpath(f"{novel['name']}.txt")
        cleaned_path = novel_dir.joinpath(f"{novel['name']}_cleaned.txt")
        if cleaned_path.exists():
            continue
        lines = extract_lines(novel_path, start=novel['actual_lines'][0], end=novel['actual_lines'][1])
        with open(cleaned_path, "w") as f:
            f.writelines(lines)
            print(f"Cleaned {novel['name']}")
    print('All done!')


def extract_lines(file_path, start=1, end=sys.maxsize):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    start = max(1, start)
    end = min(len(lines), end)
    return lines[start - 1:end]


def analysis():
    # sentence count
    # word count
    # punctuation count
    # unique word count
    # unique word frequency
    # token count
    # unique token count
    # NER count
    pass


if __name__ == '__main__':
    download_novels()
    clean()
