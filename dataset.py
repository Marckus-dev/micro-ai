import os
import torch
from tokenizer import CustomBPETokenizer


class ChatDataset:

    def __init__(self, data_path="data.txt", vocab_path="vocab.json", seq_len=128):
        self.seq_len = seq_len
        self.tokenizer = CustomBPETokenizer()
        self.tokenizer.load(vocab_path)

        # Кэшируем переведённые в цифры данные, чтобы не ждать каждый раз при запуске обучения
        cache_path = "dataset_cache.pt"
        if os.path.exists(cache_path):
            print("Загружаем готовый кэш токенов из dataset_cache.pt...")
            self.data = torch.load(cache_path)
        else:
            print("Переводим весь data.txt в цифры (токены)...")
            with open(data_path, "r", encoding="utf-8") as f:
                text = f.read()

            token_ids = self.tokenizer.encode(text)
            self.data = torch.tensor(token_ids, dtype=torch.long)
            torch.save(self.data, cache_path)
            print(f"Готово! Всего токенов в датасете: {len(self.data)}")

    def get_batch(self, batch_size=32):
        """Вырезает случайные куски диалогов для одного шага обучения.

        x — входной текст, y — тот же текст, сдвинутый на 1 токен вперёд (то, что
        модель должна угадать).
        """
        max_start = len(self.data) - self.seq_len - 1
        # Выбираем случайные стартовые позиции в тексте
        ix = torch.randint(0, max_start, (batch_size,))
        x = torch.stack([self.data[i : i + self.seq_len] for i in ix])
        y = torch.stack([self.data[i + 1 : i + self.seq_len + 1] for i in ix])
        return x, y


if __name__ == "__main__":
    ds = ChatDataset()
    x, y = ds.get_batch(batch_size=4)
    print(f"Размер батча X (вход): {x.shape}")
    print(f"Размер батча Y (правильный ответ): {y.shape}")
    print(
        "\nПример первого кусочка, который увидит нейросеть при обучении:"
    )
    print(ds.tokenizer.decode(x.tolist()))
