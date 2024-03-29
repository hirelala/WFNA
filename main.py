import requests
import json
import spacy
from pathlib import Path
import sys
from collections import Counter

spacy.cli.download('en_core_web_sm')
nlp = spacy.load('en_core_web_sm')

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
        cleaned_path = novel_dir.joinpath(f"{novel['name']}@Cleaned.txt")
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


def analysis_novels():
    print('Analyzing novels...')
    with open("novels.json", "r") as f:
        novels = json.load(f)

    for novel in novels:
        novel_dir = novels_path.joinpath(novel['name'])
        novel_path = novel_dir.joinpath(f"{novel['name']}_cleaned.txt")
        with open(novel_path, "r") as f:
            content = f.read()
            data = analysis(content)
            with open(novel_dir.joinpath(f"{novel['name']}@Analysis2.json"), "w") as f:
                json.dump(data, f, ensure_ascii=False, indent=4)
            print(f"Analyzed {novel['name']}")
    print('All done!')


def analysis(content: str):
    nlp.max_length = len(content) + 100
    doc = nlp(content)

    sentence_count = len(list(doc.sents))

    word_count = 0
    token_count = len(doc)
    punctuation_count = 0
    tokens = list()
    lemmas = list()
    for token in doc:
        tokens.append(token.text)
        if token.is_alpha:
            word_count += 1

        if token.is_punct:
            punctuation_count += 1
        else:
            lemmas.append(token.lemma_)

    token_freq = Counter(tokens)
    tokens = list(set(tokens))
    tokens.sort()

    lemma_freq = Counter(lemmas)
    lemmas = list(set(lemmas))
    lemmas.sort()

    ner_count = len(doc.ents)
    ner = dict()
    for ent in doc.ents:
        if ent.label_ not in ner:
            ner[ent.label_] = list()
        ner[ent.label_].append(ent.text)

    for key, value in ner.items():
        ner[key] = Counter(value)
        # ner[key].sort(key=lambda x: x[1], reverse=True)

    return {
        'sentence_count': sentence_count,
        'word_count': word_count,
        'punctuation_count': punctuation_count,

        'token_count': token_count,
        'tokens': tokens,
        'token_freq': token_freq,

        'ner_count': ner_count,
        'ner': ner,

        'lemmas': lemmas,
        'lemma_count': len(lemmas),
        'lemma_freq': lemma_freq
    }


if __name__ == '__main__':
    download_novels()
    clean()
    analysis_novels()
