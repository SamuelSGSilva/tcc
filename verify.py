"""
verify.py
=========
Script de auditoria: usa o pipeline HÍBRIDO completo (ResNet50 + classificador de
limiares físicos) para re-verificar TODAS as imagens já classificadas no dataset
rotulado, comparando a predição do pipeline com o rótulo atual (nome da pasta) e
gerando um relatório detalhado com métricas por classe (precision, recall, F1).

O pipeline híbrido reproduz exatamente o comportamento de produção:
  1. ResNet50 produz uma predição com confiança (com TTA).
  2. Se a confiança for >= CONFIDENCE_THRESHOLD (0.55), aceita a predição da rede.
  3. Se a confiança for < CONFIDENCE_THRESHOLD, usa o classificador de limiares
     físicos (largura_mediana_mm do results.csv) como fallback.
  4. SAFETY OVERRIDE: se o classificador de limiares detectar CRITICA e a rede
     discordar, a classe CRITICA prevalece incondicionalmente.

Para imagens sem entrada em results.csv (sem largura_mediana_mm disponível),
o script recai para predict() puro e registra a inferência como "sem_csv" —
essas imagens são contabilizadas separadamente nas métricas.

Uso:
    python verify.py --data_dir ./dataset_rotulado
    python verify.py --data_dir ./dataset_rotulado --output_dir ./verificacao
    python verify.py --data_dir ./dataset_rotulado --export_discordantes
    python verify.py --data_dir ./dataset_rotulado --results_csv ./dataset_rotulado/results.csv

Estrutura esperada de --data_dir:
    data_dir/
        CRITICA/         *.jpg / *.png (rótulo atual = CRITICA)
        SEMI_CRITICA/
        NAO_CRITICA/
        DESCARTADA/

Saídas geradas em --output_dir:
    relatorio_verificacao.csv     — resultado imagem por imagem
    relatorio_verificacao.txt     — resumo de métricas (concordância, precision,
                                    recall, F1 por classe e macro)
    discordantes/                 — (opcional) cópias das imagens em desacordo
"""

import argparse
import sys
import shutil
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))

from src.deep_classifier import DeepCrackClassifier, CLASSES
from src.classification import CriticalityClassifier

# Caminho padrão dos pesos
DEFAULT_WEIGHTS = "src/model_weights/resnet50_crack.pth"

# Caminho padrão do CSV com larguras medidas pelo pipeline
DEFAULT_RESULTS_CSV = "dataset_rotulado/results.csv"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Re-verifica imagens rotuladas usando o ResNet50 fine-tuned"
    )
    parser.add_argument(
        "--data_dir",
        required=True,
        type=str,
        help="Pasta com subdiretórios por classe (CRITICA/, SEMI_CRITICA/, NAO_CRITICA/, DESCARTADA/)",
    )
    parser.add_argument(
        "--model_weights",
        default=DEFAULT_WEIGHTS,
        type=str,
        help=f"Caminho para os pesos do ResNet50 (default: {DEFAULT_WEIGHTS})",
    )
    parser.add_argument(
        "--output_dir",
        default="./verificacao",
        type=str,
        help="Pasta de saída para relatórios (default: ./verificacao)",
    )
    parser.add_argument(
        "--results_csv",
        default=DEFAULT_RESULTS_CSV,
        type=str,
        help=(
            f"CSV com colunas 'arquivo' e 'largura_mediana_mm' gerado pelo pipeline "
            f"(default: {DEFAULT_RESULTS_CSV}). Usado para reconstituir o threshold_class "
            f"e acionar o predict_with_fallback do pipeline híbrido."
        ),
    )
    parser.add_argument(
        "--export_discordantes",
        action="store_true",
        help="Copiar imagens em desacordo para output_dir/discordantes/ por subpasta",
    )
    parser.add_argument(
        "--min_confidence",
        default=0.0,
        type=float,
        help="Confiança mínima para considerar o modelo assertivo (default: 0.0 = todos)",
    )
    parser.add_argument(
        "--device",
        default=None,
        type=str,
        choices=["cuda", "cpu"],
        help="Dispositivo de inferência. Padrão: detecta automaticamente.",
    )
    parser.add_argument(
        "--limit",
        default=0,
        type=int,
        help="Limita a quantidade de imagens para teste rápido (0 = sem limite)",
    )
    return parser.parse_args()


def collect_images(data_dir: Path) -> list:
    """
    Percorre data_dir e retorna lista de (image_path, rotulo_atual).
    Considera apenas subpastas cujos nomes estão em CLASSES.
    """
    entries = []
    exts = {".jpg", ".jpeg", ".png", ".bmp"}

    for class_dir in sorted(data_dir.iterdir()):
        if not class_dir.is_dir():
            continue
        rotulo = class_dir.name
        if rotulo not in CLASSES:
            print(f"  [AVISO] Subpasta ignorada (não é uma classe válida): {class_dir.name}")
            continue
        for img_path in sorted(class_dir.glob("*")):
            if img_path.suffix.lower() in exts:
                entries.append((img_path, rotulo))

    return entries


def print_and_write(f, text: str):
    print(text)
    f.write(text + "\n")


def main():
    args = parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    weights_path = Path(args.model_weights)

    # Validações
    if not data_dir.exists():
        print(f"[ERRO] Diretório não encontrado: {data_dir}")
        sys.exit(1)

    if not weights_path.exists():
        print(f"[ERRO] Pesos do ResNet50 não encontrados: {weights_path}")
        print("        Execute 'python train.py' primeiro para treinar o modelo.")
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Pastas para discordantes
    if args.export_discordantes:
        for cls in CLASSES:
            (output_dir / "discordantes" / cls).mkdir(parents=True, exist_ok=True)

    # Carrega o CSV de larguras (results.csv) para reconstituir o threshold_class
    results_csv_path = Path(args.results_csv)
    width_index: dict = {}   # arquivo -> largura_mediana_mm (float)
    if results_csv_path.exists():
        df_widths = pd.read_csv(results_csv_path)
        if "arquivo" in df_widths.columns and "largura_mediana_mm" in df_widths.columns:
            width_index = dict(zip(df_widths["arquivo"], df_widths["largura_mediana_mm"]))
            print(f"[CSV] {len(width_index)} entradas de largura carregadas de: {results_csv_path}")
        else:
            print(f"[AVISO] {results_csv_path} não possui colunas 'arquivo'/'largura_mediana_mm'. "
                  f"Inferência será feita sem pipeline híbrido.")
    else:
        print(f"[AVISO] results.csv não encontrado em: {results_csv_path}")
        print(f"         O script usará predict() puro (sem safety override nem fallback de limiares).")

    # Instancia o classificador de limiares físicos (para reconstituir threshold_class).
    # scale_factor=1.0 porque passamos largura_mediana_mm diretamente para classify(),
    # que espera width_px: width_mm = width_px * scale_factor → com scale=1.0, os
    # valores em mm são tratados como se fossem pixels de mesma magnitude.
    threshold_clf = CriticalityClassifier(scale_factor=1.0)

    # Carrega o modelo
    print(f"\n[ResNet50] Carregando pesos de: {weights_path}")
    clf = DeepCrackClassifier(weights_path=str(weights_path), device=args.device)
    print(f"[ResNet50] Device: {clf.device}")

    # Coleta imagens
    print(f"\nColetando imagens de: {data_dir}")
    entries = collect_images(data_dir)
    
    if args.limit > 0:
        entries = entries[:args.limit]
        print(f"[TESTE] Lista limitada a {args.limit} imagens.")
        
    total = len(entries)

    if total == 0:
        print("[ERRO] Nenhuma imagem encontrada.")
        sys.exit(1)

    print(f"Total de imagens encontradas: {total}\n")

    # ------------------------------------------------------------------ #
    # Inferência — pipeline híbrido (predict_with_fallback)
    # ------------------------------------------------------------------ #
    results = []
    n_sem_csv = 0

    for img_path, rotulo_atual in tqdm(entries, desc="Verificando imagens", unit="img"):
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            results.append({
                "arquivo": img_path.name,
                "caminho": str(img_path),
                "rotulo_atual": rotulo_atual,
                "predicao_pipeline": "ERRO_LEITURA",
                "confianca": 0.0,
                "usou_deep": False,
                "metodo": "erro",
                "largura_mm": None,
                "concordancia": False,
            })
            continue

        largura_mm = width_index.get(img_path.name)

        if largura_mm is not None:
            # Pipeline híbrido completo: threshold_class reconstruído a partir
            # da largura_mediana_mm armazenada no results.csv, sem necessidade
            # de re-executar a segmentação UNet.
            # scale_factor=1.0 porque largura_mm já está convertida para mm.
            threshold_class = threshold_clf.classify(largura_mm)
            predicao, confianca, usou_deep = clf.predict_with_fallback(
                img_bgr,
                binary_mask=None,      # máscara não necessária para o safety override
                threshold_class=threshold_class,
                use_tta=True,
            )
            metodo = "deep" if usou_deep else "threshold"
        else:
            # Sem dados de largura: recai para predict() puro (apenas ResNet50)
            predicao, confianca = clf.predict(img_bgr, binary_mask=None, use_tta=True)
            usou_deep = True
            metodo = "sem_csv"
            n_sem_csv += 1

        concordancia = (predicao == rotulo_atual)

        # Exporta discordantes
        if args.export_discordantes and not concordancia:
            dest = output_dir / "discordantes" / rotulo_atual / img_path.name
            shutil.copy2(str(img_path), str(dest))

        results.append({
            "arquivo": img_path.name,
            "caminho": str(img_path),
            "rotulo_atual": rotulo_atual,
            "predicao_pipeline": predicao,
            "confianca": round(confianca, 4),
            "usou_deep": usou_deep,
            "metodo": metodo,
            "largura_mm": round(float(largura_mm), 4) if largura_mm is not None else None,
            "concordancia": concordancia,
        })

    if n_sem_csv > 0:
        print(f"\n[AVISO] {n_sem_csv} imagem(ns) sem entrada em results.csv — "
              f"avaliadas apenas com ResNet50 (metodo='sem_csv').")

    # ------------------------------------------------------------------ #
    # Relatório CSV
    # ------------------------------------------------------------------ #
    df = pd.DataFrame(results)
    csv_path = output_dir / "relatorio_verificacao.csv"
    df.to_csv(csv_path, index=False)

    # ------------------------------------------------------------------ #
    # Métricas e relatório .txt
    # ------------------------------------------------------------------ #
    txt_path = output_dir / "relatorio_verificacao.txt"

    # Filtra pelo min_confidence se necessário
    if args.min_confidence > 0.0:
        df_eval = df[df["confianca"] >= args.min_confidence]
        n_filtrados = total - len(df_eval)
    else:
        df_eval = df
        n_filtrados = 0

    # Remove erros de leitura das métricas
    df_eval = df_eval[df_eval["predicao_pipeline"] != "ERRO_LEITURA"]

    total_eval = len(df_eval)
    concordantes = int(df_eval["concordancia"].sum())
    discordantes_n = total_eval - concordantes
    taxa_concordancia = (concordantes / total_eval * 100) if total_eval > 0 else 0.0

    # Contabiliza uso do pipeline por método
    n_por_metodo = df_eval["metodo"].value_counts().to_dict()

    # ------------------------------------------------------------------
    # Calcula precision, recall e F1 por classe (somente classes presentes)
    # ------------------------------------------------------------------
    def _prf(df_in: pd.DataFrame, cls: str) -> tuple:
        """Retorna (precision, recall, f1, suporte) para uma classe."""
        tp = int(((df_in["rotulo_atual"] == cls) & (df_in["predicao_pipeline"] == cls)).sum())
        fp = int(((df_in["rotulo_atual"] != cls) & (df_in["predicao_pipeline"] == cls)).sum())
        fn = int(((df_in["rotulo_atual"] == cls) & (df_in["predicao_pipeline"] != cls)).sum())
        support = tp + fn
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1        = (2 * precision * recall / (precision + recall)
                     if (precision + recall) > 0 else 0.0)
        return precision, recall, f1, support

    class_metrics = {}
    for cls in CLASSES:
        if (df_eval["rotulo_atual"] == cls).sum() == 0:
            continue
        class_metrics[cls] = _prf(df_eval, cls)

    # F1 macro (média sobre classes presentes)
    f1_macro = (sum(v[2] for v in class_metrics.values()) / len(class_metrics)
                if class_metrics else 0.0)
    # F1 weighted (ponderado pelo suporte)
    total_support = sum(v[3] for v in class_metrics.values())
    f1_weighted = (sum(v[2] * v[3] for v in class_metrics.values()) / total_support
                   if total_support > 0 else 0.0)

    # ------------------------------------------------------------------
    # Matthews Correlation Coefficient (MCC) multiclasse
    # MCC é a métrica mais informativa para datasets desbalanceados:
    # retorna +1 para predição perfeita, 0 para predição aleatória e -1
    # para predição inversa, levando em conta todas as células da matriz.
    # ------------------------------------------------------------------
    def _mcc_multiclass(df_in: pd.DataFrame, classes: list) -> float:
        """Calcula o MCC multiclasse a partir da matriz de confusão."""
        n = len(classes)
        cls_to_idx = {c: i for i, c in enumerate(classes)}

        # Constrói a matriz de confusão
        C = np.zeros((n, n), dtype=np.int64)
        for _, row in df_in.iterrows():
            true_idx = cls_to_idx.get(row["rotulo_atual"])
            pred_idx = cls_to_idx.get(row["predicao_pipeline"])
            if true_idx is not None and pred_idx is not None:
                C[true_idx, pred_idx] += 1

        # Fórmula de Gorodkin (2004) para MCC multiclasse:
        # MCC = (tr(C) * sum(C) - sum_k(row_k * col_k)) /
        #       sqrt((sum(C)^2 - sum_k(col_k^2)) * (sum(C)^2 - sum_k(row_k^2)))
        t = np.sum(C)
        if t == 0:
            return 0.0
        sum_sq_cols = np.sum(np.sum(C, axis=0) ** 2)
        sum_sq_rows = np.sum(np.sum(C, axis=1) ** 2)
        numerator   = np.trace(C) * t - sum(
            np.sum(C, axis=0)[k] * np.sum(C, axis=1)[k] for k in range(n)
        )
        denom_left  = t ** 2 - sum_sq_cols
        denom_right = t ** 2 - sum_sq_rows
        denominator = np.sqrt(float(denom_left) * float(denom_right))
        if denominator == 0:
            return 0.0
        return float(numerator / denominator)

    mcc = _mcc_multiclass(df_eval, CLASSES)

    # ------------------------------------------------------------------
    # Matriz de confusão (classes x classes presentes no conjunto)
    # ------------------------------------------------------------------
    present_classes = [c for c in CLASSES if (df_eval["rotulo_atual"] == c).sum() > 0]
    cls_to_idx_present = {c: i for i, c in enumerate(present_classes)}
    conf_matrix = np.zeros((len(present_classes), len(present_classes)), dtype=np.int64)
    for _, row in df_eval.iterrows():
        ti = cls_to_idx_present.get(row["rotulo_atual"])
        pi = cls_to_idx_present.get(row["predicao_pipeline"])
        if ti is not None and pi is not None:
            conf_matrix[ti, pi] += 1

    with open(txt_path, "w", encoding="utf-8") as f:
        sep = "=" * 70

        print_and_write(f, sep)
        print_and_write(f, "  RELATÓRIO DE RE-VERIFICAÇÃO — PIPELINE HÍBRIDO")
        print_and_write(f, "  (ResNet50 + Classificador de Limiares Físicos)")
        print_and_write(f, sep)
        print_and_write(f, f"  Dataset Dir   : {data_dir.resolve()}")
        print_and_write(f, f"  Pesos Modelo  : {weights_path.resolve()}")
        print_and_write(f, f"  Results CSV   : {results_csv_path.resolve()}")
        print_and_write(f, f"  Device        : {clf.device}")
        print_and_write(f, f"  Conf. Threshold: {clf.confidence_threshold:.2f}")
        print_and_write(f, sep)
        print_and_write(f, f"  Total imagens          : {total}")
        if n_filtrados > 0:
            print_and_write(f, f"  Filtradas (conf<{args.min_confidence:.2f})  : {n_filtrados}")
        print_and_write(f, f"  Avaliadas              : {total_eval}")
        print_and_write(f, f"  Concordantes           : {concordantes}")
        print_and_write(f, f"  Discordantes           : {discordantes_n}")
        print_and_write(f, f"  Taxa de Concordância   : {taxa_concordancia:.1f}%")
        print_and_write(f, sep)

        # Distribuição por método de decisão
        print_and_write(f, "\n  Distribuição por Método de Decisão:")
        for metodo_nome, cnt in sorted(n_por_metodo.items()):
            pct_m = cnt / total_eval * 100 if total_eval > 0 else 0.0
            descricao = {
                "deep":      "ResNet50 (confiança >= threshold)",
                "threshold": "Limiares físicos (fallback ou safety override)",
                "sem_csv":   "ResNet50 puro (sem largura_mm disponível)",
                "erro":      "Erro de leitura",
            }.get(metodo_nome, metodo_nome)
            print_and_write(f, f"    {metodo_nome:<12} : {cnt:>6}  ({pct_m:.1f}%)  — {descricao}")
        print_and_write(f, sep)

        # Métricas por classe
        print_and_write(f, "\n  Métricas por Classe (Precision / Recall / F1 / Suporte):")
        print_and_write(
            f,
            f"  {'Classe':<16} {'Precision':>10} {'Recall':>8} {'F1':>8} "
            f"{'Suporte':>9} | Pipeline previu como:"
        )
        print_and_write(f, "  " + "-" * 82)

        for cls in CLASSES:
            if cls not in class_metrics:
                continue
            prec, rec, f1_cls, support = class_metrics[cls]
            subset = df_eval[df_eval["rotulo_atual"] == cls]

            pred_dist = subset["predicao_pipeline"].value_counts().to_dict()
            pred_str = ", ".join(f"{k}:{v}" for k, v in sorted(pred_dist.items()))

            print_and_write(
                f,
                f"  {cls:<16} {prec:>9.1%}  {rec:>7.1%}  {f1_cls:>7.1%}  "
                f"{support:>8}  | {pred_str}"
            )

        print_and_write(f, "  " + "-" * 82)
        print_and_write(
            f,
            f"  {'F1 macro':<16} {'':>10} {'':>8} {f1_macro:>7.1%}  "
            f"{'':>8}"
        )
        print_and_write(
            f,
            f"  {'F1 weighted':<16} {'':>10} {'':>8} {f1_weighted:>7.1%}  "
            f"{'':>8}"
        )
        print_and_write(
            f,
            f"  {'MCC (multiclasse)':<16} {'':>10} {'':>8} {mcc:>+7.4f}  "
            f"{'':>8}  (intervalo [-1, +1]; 0 = aleatório)"
        )
        print_and_write(f, sep)

        # Matriz de confusão
        col_w = 14
        print_and_write(f, "\n  Matriz de Confusão (linhas = rótulo real, colunas = predição):")
        header = "  " + " " * 16 + "".join(f"{c:>{col_w}}" for c in present_classes)
        print_and_write(f, header)
        print_and_write(f, "  " + "-" * (16 + col_w * len(present_classes)))
        for i, cls_real in enumerate(present_classes):
            row_str = f"  {cls_real:<16}" + "".join(
                f"{conf_matrix[i, j]:>{col_w}}" for j in range(len(present_classes))
            )
            print_and_write(f, row_str)
        print_and_write(f, sep)

        # Métricas separadas para imagens sem_csv (avaliadas apenas com ResNet50 puro)
        df_sem_csv = df_eval[df_eval["metodo"] == "sem_csv"]
        if len(df_sem_csv) > 0:
            conc_sem_csv = int(df_sem_csv["concordancia"].sum())
            taxa_sem_csv = conc_sem_csv / len(df_sem_csv) * 100
            print_and_write(f, f"\n  Métricas (apenas imagens sem_csv — ResNet50 puro, n={len(df_sem_csv)}):")
            print_and_write(f, f"  Taxa de concordância : {taxa_sem_csv:.1f}%  ({conc_sem_csv}/{len(df_sem_csv)})")
            print_and_write(
                f,
                "  AVISO: estas imagens não tiveram safety override nem fallback por limiares."
                " Interpretá-las separadamente evita distorção das métricas globais do pipeline híbrido."
            )
            print_and_write(f, sep)

        # Lista das discordantes
        df_disc = df_eval[~df_eval["concordancia"]]
        if len(df_disc) > 0:
            print_and_write(f, f"\n  Imagens Discordantes ({len(df_disc)} total):")
            print_and_write(
                f,
                f"  {'Arquivo':<35} {'Rótulo Atual':<16} {'Pipeline Previu':<16} "
                f"{'Conf.':>6} {'Método':<12}"
            )
            print_and_write(f, "  " + "-" * 90)
            for _, row in df_disc.sort_values("confianca", ascending=False).iterrows():
                print_and_write(
                    f,
                    f"  {row['arquivo']:<35} {row['rotulo_atual']:<16} "
                    f"{row['predicao_pipeline']:<16} {row['confianca']:>5.1%}  "
                    f"{row['metodo']:<12}"
                )
        else:
            print_and_write(f, "\n  Todas as imagens estão em concordância com o pipeline!")

        print_and_write(f, sep)
        print_and_write(f, f"\n  Relatório CSV salvo em: {csv_path.resolve()}")
        if args.export_discordantes:
            print_and_write(f, f"  Imagens discordantes copiadas em: {(output_dir / 'discordantes').resolve()}")

    print(f"\n[OK] Relatório completo salvo em: {txt_path.resolve()}")


if __name__ == "__main__":
    main()
