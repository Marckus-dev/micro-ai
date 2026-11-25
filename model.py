import math
import torch
import torch.nn as nn
from torch.nn import functional as F


class CausalSelfAttention(nn.Module):
    """Механизм внимания (Self-Attention): учит слова смотреть друг на друга и понимать контекст."""

    def __init__(self, n_embd, n_head, block_size, dropout=0.1):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_head = n_head
        self.n_embd = n_embd

        # Линейные слои для создания Запросов (Query), Ключей (Key) и Значений (Value)
        self.c_attn = nn.Linear(n_embd, 3 * n_embd)
        self.c_proj = nn.Linear(n_embd, n_embd)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

        # Маска запрещает нейросети "подглядывать" в будущие слова при обучении
        self.register_buffer(
            "bias",
            torch.tril(torch.ones(block_size, block_size)).view(
                1, 1, block_size, block_size
            ),
        )

    def forward(self, x):
        B, T, C = (
            x.size()
        )  # B - размер батча, T - длина фразы, C - размер вектора (192)

        # Разделяем на Q, K, V и раскидываем по "головам" внимания
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        # Считаем, насколько сильно каждое слово связано с предыдущими словами
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)

        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.c_proj(y))


class FeedForward(nn.Module):
    """Обычная полносвязная нейросеть внутри каждого блока для обработки смысла."""

    def __init__(self, n_embd, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),  # Плавная функция активации (как в настоящем GPT)
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class TransformerBlock(nn.Module):
    """Один полноценный слой (блок) GPT: Внимание + Нейросеть + Нормализация."""

    def __init__(self, n_embd, n_head, block_size, dropout=0.1):
        super().__init__()
        self.ln_1 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head, block_size, dropout)
        self.ln_2 = nn.LayerNorm(n_embd)
        self.mlp = FeedForward(n_embd, dropout)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class MiniGPT(nn.Module):
    """Сама языковая модель с нуля."""

    def __init__(
        self,
        vocab_size=2500,
        n_embd=192,
        n_head=6,
        n_layer=4,
        block_size=128,
        dropout=0.1,
    ):
        super().__init__()
        self.block_size = block_size

        # Таблица смыслов слов и таблица позиций слов в предложении
        self.token_embedding = nn.Embedding(vocab_size, n_embd)
        self.position_embedding = nn.Embedding(block_size, n_embd)
        self.drop = nn.Dropout(dropout)

        # Стек из 4 слоёв Transformer
        self.blocks = nn.ModuleList([
            TransformerBlock(n_embd, n_head, block_size, dropout)
            for _ in range(n_layer)
        ])
        self.ln_f = nn.LayerNorm(n_embd)

        # Выходной слой: предсказывает номер следующего токена из словаря
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)

    def forward(self, idx, targets=None):
        B, T = idx.size()
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)

        # Складываем смысл слова и его позицию в предложении
        tok_emb = self.token_embedding(idx)
        pos_emb = self.position_embedding(pos)
        x = self.drop(tok_emb + pos_emb)

        # Прогоняем через все 4 слоя Transformer
        for block in self.blocks:
            x = block(x)

        x = self.ln_f(x)
        logits = self.lm_head(x)

        # Если переданы правильные ответы (во время обучения) — считаем ошибку (Loss)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1)
            )

        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens=60, temperature=0.8, stop_token=3):
        """Генерирует ответ по одному токену за раз."""
        for _ in range(max_new_tokens):
            # Обрезаем контекст до максимальной длины памяти (128 токенов)
            idx_cond = (
                idx
                if idx.size(1) <= self.block_size
                else idx[:, -self.block_size :]
            )
            logits, _ = self(idx_cond)
            # Берём предсказание только для самого последнего слова и делим на "температуру" (креативность)
            logits = logits[:, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)
            # Выбираем следующее слово с учётом вероятностей
            next_token = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, next_token), dim=1)

            # Если модель выдала токен <|end|> (ID 3) — заканчиваем ответ
            if next_token.item() == stop_token:
                break
        return idx


if __name__ == "__main__":
    model = MiniGPT()
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Мозг MiniGPT успешно собран! Всего параметров: {total_params:,}")
