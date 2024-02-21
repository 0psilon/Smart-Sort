import albumentations as A

RESCALE_SIZE = 224
PATH_TO_MODEL = "weights/SmartSort.onnx"

TRANSFORM = A.Compose([
    A.LongestMaxSize(RESCALE_SIZE),
    A.PadIfNeeded(RESCALE_SIZE, RESCALE_SIZE, border_mode=2),
    A.Normalize()
])

CLASSES = [
    'Стекло',
    'Остальное',
    'Картон',
    'Пластик',
    'Аккумуляторы',
    'Стекло',
    'Металл',
    'Стекло',
    'Бумага',
    'Органика'
]

MESSAGE_TEXTS = [
    'Идет распознавание...',
    'Одну минутку...',
    'Секундочку...',
    'Минуточку...',
    'Предсказание совсем близко...',
    'Буквально один момент...',
    'Почти-почти...',
    'Алгоритмы работают...'
]

PLASTICS = [
    'pet',
    'hdpe',
    'pvc',
    'ldpe',
    'pp',
    'ps',
    'o'
]
