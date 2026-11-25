import json
import re
from collections import Counter


class CustomBPETokenizer:

    def __init__(self):
        # Специальные токены, которые нельзя разрезать на буквы
        self.special_tokens = ["<|pad|>", "<|user|>", "<|bot|>", "<|end|>"]
        self.vocab = {}  # id -> строка
        self.inverse_vocab = {}  # строка -> id
        self.merges = {}  # (id1, id2) -> new_id

    def train(self, text, vocab_size=2500):
        print("Начинаем обучение своего BPE-токенизатора...")
        # 1. Добавляем спецтокены в начало словаря (ID 0, 1, 2, 3)
        for idx, token in enumerate(self.special_tokens):
            self.vocab[idx] = token
            self.inverse_vocab[token] = idx

        # 2. Вырезаем спецтокены из текста, чтобы собрать все уникальные обычные символы
        clean_for_chars = text
        for st in self.special_tokens:
            clean_for_chars = clean_for_chars.replace(st, "")

        unique_chars = sorted(list(set(clean_for_chars)))
        for ch in unique_chars:
            idx = len(self.vocab)
            self.vocab[idx] = ch
            self.inverse_vocab[ch] = idx

        print(f"Базовый алфавит (буквы, знаки, эмодзи): {len(self.vocab)} шт.")

        # 3. Разбиваем текст на куски, сохраняя спецтокены целыми
        pattern = (
            "(" + "|".join(re.escape(st) for st in self.special_tokens) + ")"
        )
        chunks = re.split(pattern, text)

        # Переводим весь текст в список списков чисел (ID символов)
        token_sequences = []
        for chunk in chunks:
            if not chunk:
                continue
            if chunk in self.inverse_vocab:
                token_sequences.append([self.inverse_vocab[chunk]])
            else:
                # Разбиваем текст на слова с сохранением пробелов впереди, чтобы модель знала границы слов
                words = re.findall(r"\s*\S+|\s+", chunk)
                for w in words:
                    token_sequences.append([self.inverse_vocab[c] for c in w])

        # 4. Главный цикл BPE: склеиваем самые частые пары соседей
        num_merges = vocab_size - len(self.vocab)
        for step in range(num_merges):
            pair_counts = Counter()
            for seq in token_sequences:
                for i in range(len(seq) - 1):
                    pair_counts[(seq[i], seq[i + 1])] += 1

            if not pair_counts:
                break

            # Находим самую популярную пару в ваших переписках
            best_pair = pair_counts.most_common(1)[0][0]
            new_id = len(self.vocab)
            new_token_str = self.vocab[best_pair[0]] + self.vocab[best_pair[1]]

            self.vocab[new_id] = new_token_str
            self.inverse_vocab[new_token_str] = new_id
            self.merges[best_pair] = new_id

            # Заменяем найденную пару на новый склеенный токен во всех словах
            new_sequences = []
            for seq in token_sequences:
                if len(seq) < 2:
                    new_sequences.append(seq)
                    continue
                new_seq = []
                i = 0
                while i < len(seq):
                    if (
                        i < len(seq) - 1
                        and seq[i] == best_pair[0]
                        and seq[i + 1] == best_pair[1]
                    ):
                        new_seq.append(new_id)
                        i += 2
                    else:
                        new_seq.append(seq[i])
                        i += 1
                new_sequences.append(new_seq)
            token_sequences = new_sequences

            if (step + 1) % 250 == 0:
                print(
                    f"Склеек сделано: {step + 1}/{num_merges} | Последний токен: '{new_token_str}'"
                )

        print(f"Обучение токенизатора завершено! Размер словаря: {len(self.vocab)}")

    def encode(self, text):
        """Превращает текст в список цифр (токенов) для нейросети."""
        pattern = (
            "(" + "|".join(re.escape(st) for st in self.special_tokens) + ")"
        )
        chunks = re.split(pattern, text)
        result_ids = []

        for chunk in chunks:
            if not chunk:
                continue
            if chunk in self.inverse_vocab:
                result_ids.append(self.inverse_vocab[chunk])
            else:
                words = re.findall(r"\s*\S+|\s+", chunk)
                for w in words:
                    # Неизвестные редкие символы заменяем на пробел, чтобы не было ошибок
                    ids = [
                        self.inverse_vocab.get(c, self.inverse_vocab.get(" ", 0))
                        for c in w
                    ]
                    while len(ids) >= 2:
                        # Ищем пару с самым ранним (приоритетным) правилом склейки
                        best_pair = None
                        best_merge_id = float("inf")
                        for i in range(len(ids) - 1):
                            pair = (ids[i], ids[i + 1])
                            if (
                                pair in self.merges
                                and self.merges[pair] < best_merge_id
                            ):
                                best_merge_id = self.merges[pair]
                                best_pair = pair
                        if best_pair is None:
                            break
                        new_ids = []
                        i = 0
                        while i < len(ids):
                            if (
                                i < len(ids) - 1
                                and ids[i] == best_pair[0]
                                and ids[i + 1] == best_pair[1]
                            ):
                                new_ids.append(best_merge_id)
                                i += 2
                            else:
                                new_ids.append(ids[i])
                                i += 1
                        ids = new_ids
                    result_ids.extend(ids)
        return result_ids

    def decode(self, ids):
        """Превращает список цифр обратно в читаемый текст."""
        return "".join(self.vocab.get(i, "") for i in ids)

    def save(self, filepath="vocab.json"):
        data = {
            "vocab": {str(k): v for k, v in self.vocab.items()},
            "merges": {f"{k[0]},{k[1]}": v for k, v in self.merges.items()},
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"Словарь сохранён в {filepath}")

    def load(self, filepath="vocab.json"):
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.vocab = {int(k): v for k, v in data["vocab"].items()}
        self.inverse_vocab = {v: int(k) for k, v in data["vocab"].items()}
        self.merges = {}
        for k, v in data["merges"].items():
            a, b = map(int, k.split(","))
            self.merges[(a, b)] = v


if __name__ == "__main__":
    with open("data.txt", "r", encoding="utf-8") as f:
        text = f.read()

    tokenizer = CustomBPETokenizer()
    # Обучаем словарь на 2500 токенов (займёт около 15-30 секунд на твоём Ryzen 5)
    tokenizer.train(text, vocab_size=2500)
    tokenizer.save("vocab.json")

    # Тестируем работу на живой фразе
    test_phrase = "<|user|> я тебе эдгара на десятый ранг апну <|bot|> Фактишь 😂 <|end|>"
    encoded = tokenizer.encode(test_phrase)
    pieces = [tokenizer.vocab[i] for i in encoded]
    decoded = tokenizer.decode(encoded)

    print("\n--- ТЕСТ ТОКЕНИЗАТОРА ---")
    print("Исходный текст:", test_phrase)
    print("Как видит нейросеть (куски):", pieces)
    print("Цифры (ID токенов):", encoded)
    print("Обратно в текст:", decoded)
