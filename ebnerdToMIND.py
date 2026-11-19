import argparse
from pathlib import Path
from shutil import copyfile
from tempfile import TemporaryDirectory
import polars as pl


def process_news(articles_parquet_path, news_tsv_path):
    """기사 ID·카테고리·제목·본문을 news.tsv의 8개 컬럼으로 저장한다."""
    columns = ['article_id', 'category', 'subcategory', 'title', 'body']
    articles = pl.read_parquet(articles_parquet_path, columns=columns)
    news_tsv_path = Path(news_tsv_path)
    news_tsv_path.parent.mkdir(parents=True, exist_ok=True)

    with news_tsv_path.open('w', encoding='utf-8', newline='\n') as news_file:
        for row in articles.iter_rows(named=True):
            category = str(row['category']) if row['category'] is not None else '<UNKNOWN>'
            subcategories = row['subcategory'] or []
            subcategory = str(subcategories[0]) if subcategories else '<UNKNOWN>'
            title = (row['title'] or '').replace('\t', ' ').replace('\r', ' ').replace('\n', ' ')
            body = (row['body'] or '').replace('\t', ' ').replace('\r', ' ').replace('\n', ' ')

            # MIND: ID, category, subcategory, title, abstract, URL, title entities, abstract entities
            # EB-NeRD의 본문을 abstract 칸에 넣고, 사용하지 않는 entity는 빈 JSON 배열로 둔다.
            fields = [str(row['article_id']), category, subcategory, title, body, '', '[]', '[]']
            news_file.write('\t'.join(fields) + '\n')

    return articles.height


def process_behaviors(behaviors_parquet_path, history_parquet_path):
    history_columns = ['user_id', 'article_id_fixed', 'impression_time_fixed']
    history = pl.read_parquet(history_parquet_path, columns=history_columns)

    history_texts = [] # article_ids
    last_click_times = []
    for row in history.iter_rows(named=True):
        article_ids = row['article_id_fixed']
        click_times = row['impression_time_fixed']
        if len(article_ids) != len(click_times):
            raise ValueError('History article IDs and timestamps must have equal lengths')

        clicks = sorted(zip(click_times, article_ids), key=lambda click: click[0])
        history_texts.append(' '.join(str(article_id) for _, article_id in clicks))
        last_click_times.append(clicks[-1][0] if clicks else None)

    history = history.select('user_id').with_columns(
        pl.Series('history', history_texts, dtype=pl.String),
        pl.Series('last_click_time', last_click_times, dtype=pl.Datetime('us')),
    )

    behavior_columns = [
        'impression_id', 'user_id', 'impression_time', 'article_ids_inview', 'article_ids_clicked',
    ]
    behaviors = pl.read_parquet(behaviors_parquet_path, columns=behavior_columns)
    behaviors = behaviors.join(history, on='user_id', how='left', validate='m:1', maintain_order='left')

    if behaviors['impression_time'].null_count() > 0:
        raise ValueError('Impressions must have timestamps')
    if behaviors['history'].null_count() > 0:
        raise ValueError('A user in behaviors has no matching history')
    if behaviors.filter(pl.col('last_click_time') >= pl.col('impression_time')).height > 0:
        raise ValueError('Click history must precede its impression')

    return behaviors


def write_behaviors(behaviors, behaviors_tsv_path):
    behaviors_tsv_path = Path(behaviors_tsv_path)
    behaviors_tsv_path.parent.mkdir(parents=True, exist_ok=True)
    with behaviors_tsv_path.open('w', encoding='utf-8', newline='\n') as behaviors_file:
        for row in behaviors.iter_rows(named=True):
            candidate_ids = row['article_ids_inview']
            clicked_ids = set(row['article_ids_clicked'] or [])
            if not candidate_ids or not clicked_ids:
                raise ValueError('Each impression must have candidates and clicked articles')
            if not clicked_ids.issubset(candidate_ids):
                raise ValueError('Clicked articles must be included in the candidates')
            if clicked_ids == set(candidate_ids):
                raise ValueError('Each impression must have a non-clicked candidate')

            # 실제 노출된 후보를 원래 순서대로 보존한다. 클릭=1, 미클릭=0.
            candidates = ' '.join(
                f'{article_id}-{1 if article_id in clicked_ids else 0}'
                for article_id in candidate_ids
            )
            fields = [
                str(row['impression_id']), str(row['user_id']), str(row['impression_time']),
                row['history'], candidates,
            ]
            behaviors_file.write('\t'.join(fields) + '\n')

    return behaviors.height


def split_training(behaviors):
    dev_size = (behaviors.height + 9) // 10
    if dev_size == 0:
        raise ValueError('The official training split is empty')
    cutoff = behaviors['impression_time'].sort()[-dev_size]
    train = behaviors.filter(pl.col('impression_time') < cutoff)
    dev = behaviors.filter(pl.col('impression_time') >= cutoff)
    if train.is_empty() or dev.is_empty():
        raise ValueError('Need non-empty train and dev with distinct time ranges')
    return train, dev


def raw_file(root, standard_name, local_name):
    for name in (standard_name, local_name):
        path = root / name
        if path.is_file():
            return path
    raise FileNotFoundError(f'Expected {standard_name} or {local_name} in {root}')


def main():
    project_root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description='Convert EB-NeRD Parquet to MIND TSV')
    parser.add_argument('--raw-dir', type=Path, default=project_root / 'data/ebnerd/raw/ebnerd_small')
    parser.add_argument('--output-dir', type=Path, default=project_root / 'data/ebnerd')
    args = parser.parse_args()
    raw_root = args.raw_dir
    output_root = args.output_dir.resolve()
    articles = raw_file(raw_root, 'articles.parquet', 'eb-nerd_articles.parquet')
    train_behaviors = raw_file(raw_root / 'train', 'behaviors.parquet', 'eb-berd_behaviors.parquet')
    train_history = raw_file(raw_root / 'train', 'history.parquet', 'eb-nerd_history.parquet')
    validation = raw_root / 'validation'

    output_root.mkdir(parents=True, exist_ok=True)
    counts = {}
    with TemporaryDirectory(prefix='.convert-', dir=output_root) as temporary:
        staged = Path(temporary)
        train_news = staged / 'train/news.tsv'
        news_count = process_news(articles, train_news)

        train, dev = split_training(process_behaviors(train_behaviors, train_history))
        counts['train'] = write_behaviors(train, staged / 'train/behaviors.tsv')
        counts['dev'] = write_behaviors(dev, staged / 'dev/behaviors.tsv')
        del train, dev

        test = process_behaviors(validation / 'behaviors.parquet', validation / 'history.parquet')
        if test.is_empty():
            raise ValueError('The official validation split is empty')
        counts['test'] = write_behaviors(test, staged / 'test/behaviors.tsv')
        for split in ('dev', 'test'):
            copyfile(train_news, staged / split / 'news.tsv')

        for split, impressions in counts.items():
            split_root = output_root / split
            split_root.mkdir(exist_ok=True)
            for name in ('news.tsv', 'behaviors.tsv'):
                (staged / split / name).replace(split_root / name)
            print(f'{split}: {news_count:,} news, {impressions:,} impressions -> {split_root}')


if __name__ == '__main__':
    main()