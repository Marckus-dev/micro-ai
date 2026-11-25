import os
import time
import torch
from dataset import ChatDataset
from model import MiniGPT


def chat_loop(model, tokenizer, device):
    """Чат с памятью разговора."""
    model.eval()
    print("\n" + "=" * 50)
    print("🤖 ЧАТ ЗАПУЩЕН (Умный режим с памятью)!")
    print(
        "Пиши сообщения. Команда 'сброс' очистит память, 'выход' — закрыть."
    )
    print("=" * 50 + "\n")

    history = ""

    while True:
        user_input = input("Ты: ").strip()
        if not user_input:
            continue
        if user_input.lower() in ["выход", "exit", "quit"]:
            break
        if user_input.lower() == "сброс":
            history = ""
            print("[Память очищена]\n")
            continue

        history += f"<|user|> {user_input} <|bot|>"
        input_ids = tokenizer.encode(history)

        if len(input_ids) > 100:
            input_ids = input_ids[-100:]

        x = torch.tensor([input_ids], dtype=torch.long, device=device)

        output_ids = model.generate(
            x, max_new_tokens=50, temperature=0.55, stop_token=3
        )

        # squeeze(0) безопасно превращает двумерный тензор в плоский список чисел
        flat_ids = output_ids.squeeze(0).tolist()
        generated_only = flat_ids[len(input_ids) :]
        reply = tokenizer.decode(generated_only).replace("<|end|>", "").strip()

        history += f" {reply} <|end|> "
        print(f"Бот: {reply}\n")


def main():
    torch.set_num_threads(os.cpu_count() or 4)
    device = torch.device("cpu")

    dataset = ChatDataset(
        data_path="data.txt", vocab_path="vocab.json", seq_len=128
    )
    vocab_size = len(dataset.tokenizer.vocab)

    model = MiniGPT(
        vocab_size=vocab_size,
        n_embd=192,
        n_head=6,
        n_layer=4,
        block_size=128,
        dropout=0.1,
    ).to(device)

    weights_path = "my_gpt_weights.pt"
    if os.path.exists(weights_path):
        print(f"Найдены сохранённые веса ({weights_path})! Загружаем...")
        model.load_state_dict(torch.load(weights_path, map_location=device))
        choice = (
            input(
                "Хочешь сразу поболтать (1) или доучить её ещё (2)? Введи 1 или 2: "
            )
            .strip()
        )
        if choice == "1":
            chat_loop(model, dataset.tokenizer, device)
            return

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4)

    total_steps = 2000
    batch_size = 32

    print(
        f"\n🚀 Начинаем обучение на {total_steps} шагов (Ctrl+C — остановить и перейти в чат)..."
    )
    model.train()
    start_time = time.time()

    try:
        for step in range(1, total_steps + 1):
            xb, yb = dataset.get_batch(batch_size=batch_size)
            xb, yb = xb.to(device), yb.to(device)

            logits, loss = model(xb, yb)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            if step % 50 == 0:
                elapsed = time.time() - start_time
                print(
                    f"Шаг {step}/{total_steps} | Ошибка (Loss): {loss.item():.4f} | Прошло времени: {elapsed:.0f} сек"
                )

            if step % 250 == 0:
                torch.save(model.state_dict(), weights_path)
                model.eval()
                test_prompt = "<|user|> привет как дела <|bot|>"
                test_ids = torch.tensor(
                    [dataset.tokenizer.encode(test_prompt)],
                    dtype=torch.long,
                    device=device,
                )
                out = model.generate(
                    test_ids, max_new_tokens=30, temperature=0.55, stop_token=3
                )
                flat_out = out.squeeze(0).tolist()
                sample_reply = (
                    dataset.tokenizer.decode(flat_out)
                    .replace("<|end|>", "")
                    .strip()
                )
                print(f"   [Проверка речи на шаге {step}]: {sample_reply}")
                model.train()

    except KeyboardInterrupt:
        print("\nОбучение остановлено вручную! Сохраняем прогресс...")

    torch.save(model.state_dict(), weights_path)
    print(f"Веса модели сохранены в {weights_path}!")
    chat_loop(model, dataset.tokenizer, device)


if __name__ == "__main__":
    main()
