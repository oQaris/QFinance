import datetime
import os

DATE_FORMAT = '%Y-%m-%d %H:%M:%S'


def check_file_path(file_path):
    # Проверяем существование директорий
    dir_path = os.path.dirname(file_path)
    if not os.path.exists(dir_path):
        raise FileNotFoundError(f"Директория {dir_path} не существует")
    # Проверяем существование файла
    if os.path.exists(file_path):
        print(f"Предупреждение: файл {file_path} уже существует.")


def string_to_utc_datetime(str_date):
    return datetime.datetime.strptime(str_date, DATE_FORMAT).replace(tzinfo=datetime.timezone.utc)


def datetime_to_utc_string(date):
    return date.strftime(DATE_FORMAT)
