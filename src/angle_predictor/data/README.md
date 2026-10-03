# Загрузка и преобразование датасета углов

`AngleMemmapDataset` читает `images.dat`, `labels.dat` и `meta.json` из каталога,
созданного генератором `synthetic_lines`. Изображения остаются на диске до
обращения к конкретной записи. Dataset возвращает `uint8` RGB `[3,H,W]` и
`float32` метку `[sin(2θ), cos(2θ)]`; θ — угол **неориентированной** линии в
координатах изображения, в диапазоне `[0, π)`.

## Разделение данных

Индексы перемешиваются один раз с заданным `seed`, затем делятся на 80% train,
10% validation и 10% test. Для 150 000 изображений это 120 000 / 15 000 /
15 000. У всех трёх объектов Dataset должны быть одинаковые `seed` и доли.
Validation и test не аугментируются. Зафиксируйте seed и доли в конфиге
эксперимента; для сопоставимых запусков используйте тот же test split.

```python
import torch
from torch.utils.data import DataLoader
from angle_predictor.data.angle_dataset import AngleMemmapDataset
from angle_predictor.data.angle_transforms import AngleBatchPreprocessor


def main() -> None:
    root = "data/datasets/synthetic_lines_150k"
    train = AngleMemmapDataset(root, "train", seed=42)
    val = AngleMemmapDataset(root, "val", seed=42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_pin_memory = device.type == "cuda"
    train_loader = DataLoader(
        train,
        batch_size=32,
        shuffle=True,
        num_workers=4,
        pin_memory=use_pin_memory,
        persistent_workers=True,
        generator=torch.Generator().manual_seed(42),
    )
    val_loader = DataLoader(
        val,
        batch_size=32,
        shuffle=False,
        num_workers=4,
        pin_memory=use_pin_memory,
        persistent_workers=True,
    )
    preprocess = AngleBatchPreprocessor().to(device)
    for images, labels in train_loader:
        images = images.to(device, non_blocking=use_pin_memory)
        labels = labels.to(device, non_blocking=use_pin_memory)
        polar_images, labels = preprocess(images, labels, augment=True)

    for images, labels in val_loader:
        images = images.to(device, non_blocking=use_pin_memory)
        labels = labels.to(device, non_blocking=use_pin_memory)
        polar_images, labels = preprocess(images, labels, augment=False)


if __name__ == "__main__":
    main()
```

На Windows `num_workers > 0` запускает процессы через `spawn`, поэтому код
создания DataLoader и обучения должен находиться под указанным `__main__` guard.
Число workers и размер батча подбирайте измерением throughput на своём диске
и GPU; `num_workers=0` может оказаться быстрее при дешёвом чтении из memmap.
На RTX 5070 с этим датасетом 4 workers дали лучший результат среди измеренных
вариантов 0, 2 и 4. Это не сравнение с шестью workers: текущие experiment YAML
и 50 loss-sweep запусков использовали 6, а benchmark этого значения не
проверял. Полные замеры и сравнение с TorchVision, Kornia и Albumentations
приведены в [`docs/augmentation-speed.md`](../../../docs/augmentation-speed.md).

При инференсе передавайте такой же `uint8` RGB батч `[B,3,H,W]` и вызывайте
`preprocess(images, augment=False)`; возвращаемая вторая величина будет `None`,
если метки не переданы. Обучение и инференс должны использовать одинаковые
`input_size`, `output_size`, среднее и стандартное отклонение. Сохраните эти
параметры вместе с checkpoint модели.

## Порядок преобразований

1. `uint8` RGB → `float32` в `[0,1]`.
2. Только train: независимые горизонтальное и вертикальное отражения с
   вероятностью 0.5 каждое. Для любого одного отражения метка становится
   `[-sin(2θ), cos(2θ)]`; для обоих сразу она не меняется.
3. Только train: мягкие изменения яркости, контраста и насыщенности в диапазоне
   множителей `[0.9,1.1]` с вероятностью 0.7. Hue пока не меняем: белая тонкая
   линия остаётся белой, а цвет фона уже существенно меняется генератором.
4. Только train: аддитивный гауссов шум с вероятностью 0.3 и σ в
   `[0.004,0.015]`; мультипликативный (speckle) шум с вероятностью 0.15 и
   коэффициентом `[0.005,0.020]`. Значения ограничиваются диапазоном `[0,1]`.
   Шумы намеренно малы относительно яркости линии толщиной 1.25 пикселя.
5. Обязательное signed-polar преобразование **одним** `grid_sample` на батч.
   Его можно выполнять на GPU после передачи батча. Колонки — углы `[0,π)`,
   строки — радиус `[-R,R]`, где `R=min((W-1)/2,(H-1)/2)`.
6. Фиксированная нормализация `(x−0.5)/0.5`, то есть примерно `[-1,1]`.
   Она не требует оценки статистики на 29.5 GB данных (27.5 GiB) и в точности
   повторяется на инференсе. Если позже оценивать RGB mean/std, считайте их
   **только по train split после поляризации без аугментаций**, а значения
   сохраняйте в checkpoint; статистику validation/test использовать нельзя.

Полярная сетка вычислена для `align_corners=True`. Все её точки лежат внутри
вписанного круга и дополнительно ограничены интервалом `[-1,1]`, поэтому при
текущей геометрии `padding_mode` не влияет на результат. Выбранный
`padding_mode="border"` остаётся разумной защитой, если позже разрешить точки
за границей: тогда `zeros` добавит искусственный тёмный край, а `reflection`
отразит изображение. Угловая граница имеет особую топологию:
`P(r, θ+π)=P(-r, θ)`. Функция `pad_signed_polar_angle` реализует такое
замыкание для свёртки, которая работает непосредственно с полной
двумерной полярной картой. Сейчас модель её не вызывает: ConvNeXt использует
собственный padding, а угловая `Conv1d` в regression head применяется уже
после агрегации по радиусу и использует обычный circular padding. Изменение
обработки seam требует отдельного изменения архитектуры. Радиальная ось не
является периодической.

Подходящие абляции после базовой модели: убрать каждый вид шума, снизить или
повысить jitter, сравнить разрешение по углу и попробовать статистику train
split. Метрика ошибки угла должна учитывать период π: например,
`abs(((θ_pred−θ_true+π/2) % π)−π/2)`.
