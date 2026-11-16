"""Create the Russian report from completed measurements."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main():
    """Render the measured dataset, benchmark and learning tables."""
    results = ROOT / 'results'
    metrics = json.loads((results / 'metrics.json').read_text(encoding='utf-8'))
    dataset = metrics['dataset']
    history = metrics['training']['history']
    class_rows = ['| Класс | Число изображений |', '|---|---:|']
    class_rows += [f'| {name} | {count} |' for name, count in zip(dataset['classes'], dataset['counts'])]
    bench_rows = ['| Сторона | Загрузка 100, мс | Аугментация 100, мс | Пик Python, КиБ | Два RGB-буфера, КиБ |',
                  '|---:|---:|---:|---:|---:|']
    bench_rows += [f"| {r['size']} | {r['load_ms']:.2f} | {r['transform_ms']:.2f} | "
                   f"{r['python_peak_bytes'] / 1024:.2f} | {r['rgb_bytes'] / 1024:.2f} |"
                   for r in metrics['benchmark']]
    train_rows = ['| Эпоха | Train loss | Train acc | Val loss | Val acc |',
                  '|---:|---:|---:|---:|---:|']
    train_rows += [f"| {r['epoch']} | {r['train']['loss']:.4f} | {r['train']['accuracy']:.2%} | "
                   f"{r['val']['loss']:.4f} | {r['val']['accuracy']:.2%} |" for r in history]
    sizes = dataset['widths_heights']
    first, last = metrics['benchmark'][0], metrics['benchmark'][-1]
    text = [
        '# Урок 5. Аугментации и работа с изображениями',
        '## Задание и цель работы\n\nИзучить аугментации, сравнить затраты на обработку изображений '
        'и дообучить модель для классификации персонажей.',
        '## Использованные методы\n\nИспользована папка train из [набора преподавателя]'
        '(https://drive.google.com/drive/folders/1BzUQeJgICv2ZDXA9HEWcfiAXBRi4Ib2a). '
        'В корне опубликованного набора были train и test. Валидация сформирована '
        'стратифицированным отбором примерно 20% из train с seed 42. '
        'Точные файловые дубликаты удалены до разбиения. Официальный test не использовался.',
        '## Реализация\n\nЗагрузка через CustomImageDataset. Сравнены пять стандартных '
        'преобразований torchvision и их композиция. Кастомные операции включают размытие, '
        'горизонтальный сдвиг по строкам и изменение яркости с контрастом. '
        'Для визуального сравнения использованы шум, CutOut и AutoContrast из extra_augs.py. '
        'Три конфигурации сохраняют обработанные обучающие изображения локально в results/generated.',
        '## Эксперименты\n\nНа одних и тех же 100 обучающих изображениях измерены загрузка '
        'и тяжёлая аугментация при четырёх размерах. tracemalloc учитывает Python-аллокации. '
        'Объём двух RGB-буферов рассчитан отдельно по размеру изображения. Пиковая память процесса '
        'и нативные промежуточные буферы Pillow не измерялись.\n\n'
        f"MobileNetV3 Small с весами ImageNet дообучалась 3 эпохи. Train — {history[0]['train']['n']}, "
        f"val — {history[0]['val']['n']} изображений. AdamW, шаг 0,0003, батч 16, вход 128×128. "
        'Случайные аугментации использованы только при обучении.',
        '## Результаты, графики и таблицы',
        '\n'.join(class_rows),
        f"Минимальные ширина и высота — {sizes['min']}; максимальные — {sizes['max']}; "
        f"средние — {sizes['mean'][0]:.1f} и {sizes['mean'][1]:.1f} пикселя.",
        '[Распределение](results/dataset.png) · [Стандартные операции](results/standard.png) · '
        '[Кастомные операции](results/custom.png) · [Конфигурации](results/pipelines.png)',
        '\n'.join(bench_rows),
        '[График времени и памяти](results/benchmark.png)',
        '\n'.join(train_rows),
        '[Кривые обучения](results/training.png) · [JSON](results/metrics.json) · [Веса](results/mobilenet.pt)',
        '## Выводы\n\n'
        f"При переходе с 64 до 512 пикселей время аугментации выросло "
        f"с {first['transform_ms']:.2f} до {last['transform_ms']:.2f} мс на 100 изображений. "
        f"Валидационная точность изменилась с {history[0]['val']['accuracy']:.2%} "
        f"до {history[-1]['val']['accuracy']:.2%}. Результат относится к одному небольшому разбиению.",
    ]
    (ROOT / 'REPORT.md').write_text('\n\n'.join(text) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
