import argparse
from bisect import bisect_right
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta, timezone
import json
from math import ceil
from pathlib import Path
import random
import re


def clean_text(text):
    return ' '.join(text.split()) # text.split()은 공백·탭·줄바꿈을 기준으로 문자열을 나눕니다.


def read_article(content_file):
    with content_file.open(encoding='utf-8-sig') as content_records:
        content = json.load(content_records)
    fields = {item['field']: item['value'] for item in content['fields']}
    if not all(field in fields for field in ('title', 'body', 'kw-category')):
        return None

    categories = fields['kw-category']
    if isinstance(categories, str):
        category, subcategory = categories, ''
    else:
        if not categories:
            return None
        category = categories[0]
        subcategory = categories[1] if len(categories) > 1 else ''
    body = fields['body']
    return content['id'], {
        'title': clean_text(fields['title']),
        'body': clean_text(body if isinstance(body, str) else ' '.join(body[:5])),
        'category': clean_text(category),
        'subcategory': clean_text(subcategory),
    }


def process_news(content_directory):
    content_files = sorted(path for path in Path(content_directory).iterdir() if path.is_file())
    news = {}
    # map은 파일 순서를 유지하고, 작업을 나눠 대기 중인 파일 수를 제한한다.
    batch_size = 1024
    with ThreadPoolExecutor(max_workers=8) as readers:
        for start in range(0, len(content_files), batch_size):
            for result in readers.map(read_article, content_files[start:start + batch_size]):
                if result is not None:
                    article_id, article = result
                    news[article_id] = article

    for index, article in enumerate(news.values(), start=1):
        article['news_id'] = f'N{index}'
    return news # news_id, title, body, category, subcategory

def daily_event_files(event_directory):
    files_by_date = {}
    for path in sorted(Path(event_directory).iterdir(), key=lambda path: path.name):
        match = re.fullmatch(r'(\d{4})(\d{2})(\d{2})(?:\.jsonl?)?', path.name)
        if match is None or not path.is_file():
            continue
        event_date = date(*map(int, match.groups()))
        previous = files_by_date.get(event_date)
        if previous is None or not path.suffix:
            files_by_date[event_date] = path
        elif previous.suffix:
            raise ValueError(f'Ambiguous event files for {event_date}: {previous.name}, {path.name}')

    dates = sorted(files_by_date)
    if len(dates) != 7 or dates[-1] - dates[0] != timedelta(days=6):
        raise ValueError(f'{event_directory}: expected 7 consecutive daily event files, found {dates}')
    return [files_by_date[event_date] for event_date in dates]


def process_events(event_directory, news):
    """일별 파일 7개에서 제목이 일치하는 클릭을 모아 시간순으로 정렬한다."""
    title_to_news_id = {}
    for article in news.values():
        title_to_news_id.setdefault(article['title'], article['news_id']) # 뉴스의 title을 key로, id를 value로 저장함.(title이 동일한 뉴스가 여러 개이면, 첫번째 뉴스의 id를 사용함)

    uid2index = {}
    user_clicks = {}
    for day, event_file in enumerate(daily_event_files(event_directory), start=1):
        matched = 0
        with event_file.open(encoding='utf-8-sig') as event_records:
            for line in event_records:
                event = json.loads(line)
                if 'title' not in event: # 제목이 없는 event는 건너뜀(뉴스 기사 페이지가 아니라는 의미)
                    continue

                news_id = title_to_news_id.get(clean_text(event['title']))
                if news_id is None:
                    continue

                click_time = int(event['time'])
                user_id = event['userId'] # user id: 긴 문자열을 정수에 매핑함
                if user_id not in uid2index:
                    uid2index[user_id] = len(uid2index)
                user_index = uid2index[user_id]
                user_clicks.setdefault(user_index, []).append((click_time, day, news_id))
                matched += 1
        print(f'Day {day} ({event_file.name}): {matched:,} matched clicks', flush=True)

    for click_info in user_clicks.values():
        click_info.sort(key=lambda x: x[0]) # click_time으로 정렬(=시간순 정렬)
    return user_clicks

def write_mind_files(news, user_clicks, output_directory, negative_sample_num=20, seed=0):
    """뉴스 8개 컬럼과 행동 5개 컬럼을 train·dev·test에 저장한다."""
    if negative_sample_num < 1:
        raise ValueError('negative_sample_num must be positive')
    rng = random.Random(seed)
    all_news_ids = [article['news_id'] for article in news.values()] # article[news_id]: N1
    news_positions = {news_id: index for index, news_id in enumerate(all_news_ids)}
    train_rows, evaluation_rows = [], []

    for user_id, click_info in user_clicks.items():
        click_news = [news_id for _, day, news_id in click_info if day < 6] # 1~5일차: click history
        train_clicks = [(time, news_id) for time, day, news_id in click_info if day == 6]
        test_clicks = [(time, news_id) for time, day, news_id in click_info if day == 7]
        if not train_clicks and not test_clicks:
            continue

        excluded_positions = sorted({news_positions[news_id] for _, _, news_id in click_info})
        available_count = len(all_news_ids) - len(excluded_positions)
        if available_count < negative_sample_num:
            raise ValueError(f'User {user_id}: {available_count} negatives available, {negative_sample_num} requested')
        # 클릭 기사를 건너뛰며 표본 순위를 실제 뉴스 목록의 위치로 바꾼다.
        excluded_ranks = [position - index for index, position in enumerate(excluded_positions)]

        for positives, history_news, rows in [
            (train_clicks, click_news, train_rows),
            (test_clicks, click_news + [news_id for _, news_id in train_clicks], evaluation_rows),
        ]:
            history = ' '.join(history_news)
            for click_time, positive_id in positives:
                negative_ranks = rng.sample(range(available_count), negative_sample_num)
                negatives = [all_news_ids[rank + bisect_right(excluded_ranks, rank)] for rank in negative_ranks]
                candidates = ' '.join([f'{positive_id}-1'] + [f'{news_id}-0' for news_id in negatives])
                impression_time = datetime.fromtimestamp(click_time, timezone.utc).strftime('%m/%d/%Y %I:%M:%S %p')
                rows.append((user_id, impression_time, history, candidates))

    # 7일 차의 전체 행에서 dev 30%, test 70%를 나눈다.
    random.Random(77).shuffle(evaluation_rows)
    dev_count = ceil(len(evaluation_rows) * 0.3)
    split_rows = {
        'train': train_rows,
        'dev': evaluation_rows[:dev_count],
        'test': evaluation_rows[dev_count:],
    }

    output_directory = Path(output_directory)
    for split, rows in split_rows.items():
        split_directory = output_directory / split
        split_directory.mkdir(parents=True, exist_ok=True)
        with (split_directory / 'news.tsv').open('w', encoding='utf-8', newline='\n') as news_file:
            for article in news.values():
                fields = [
                    article['news_id'], article['category'], article['subcategory'],
                    article['title'], article['body'], '', '[]', '[]',
                ]
                news_file.write('\t'.join(fields) + '\n')

        with (split_directory / 'behaviors.tsv').open('w', encoding='utf-8', newline='\n') as behavior_file:
            for impression_id, (user_id, impression_time, history, candidates) in enumerate(rows, start=1):
                fields = [str(impression_id), str(user_id), impression_time, history, candidates]
                behavior_file.write('\t'.join(fields) + '\n')

    return {split: len(rows) for split, rows in split_rows.items()}


def main():
    project_root = Path(__file__).resolve().parent
    raw_directory = project_root / 'data/adressa'
    parser = argparse.ArgumentParser(description='Convert Adressa one-week JSONL to MIND TSV')
    parser.add_argument('--content_dir', type=Path, default=raw_directory / 'contentdata/home/lemeiz/content_refine')
    parser.add_argument('--event_dir', type=Path, default=raw_directory / 'one_week')
    parser.add_argument('--output_dir', type=Path, default=project_root / 'data/mind')
    parser.add_argument('--negative_sample_num', type=int, default=20)
    parser.add_argument('--seed', type=int, default=2023)
    args = parser.parse_args()

    news = process_news(args.content_dir)
    print(f'Loaded {len(news):,} news articles', flush=True)
    user_clicks = process_events(args.event_dir, news)
    counts = write_mind_files(news, user_clicks, args.output_dir, args.negative_sample_num, args.seed)
    print(f"Saved behaviors: {counts}")


if __name__ == '__main__':
    main()