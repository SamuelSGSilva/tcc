import re, time, sys

OUTPUT_FILE = r"C:\Users\samuc\AppData\Local\Temp\claude\C--Users-samuc-Downloads-tcc\06f66159-490f-417d-b845-b485c637266e\tasks\bfslw7vlx.output"
seen_epochs = set()

print("Monitorando treinamento...", flush=True)
while True:
    try:
        with open(OUTPUT_FILE, "rb") as f:
            content = f.read().decode("utf-8", errors="ignore")
    except:
        time.sleep(10)
        continue

    epochs = re.findall(r"(Epoch \[(\d+)/\d+\][^\n]+)", content)
    for line, num in epochs:
        if num not in seen_epochs:
            seen_epochs.add(num)
            print(f"\n>>> EPOCA {num} CONCLUIDA <<<", flush=True)
            print(line.strip(), flush=True)

    if "Treinamento conclu" in content or "concluido" in content.lower():
        print("\n=== TREINAMENTO FINALIZADO ===", flush=True)
        break

    time.sleep(30)
