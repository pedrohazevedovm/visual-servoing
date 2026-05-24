import sys
import csv
from pathlib import Path
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from skimage.exposure import match_histograms
from datetime import datetime

# Adiciona o módulo boruvka-superpixel ao path
sys.path.insert(0, "boruvka-superpixel/pybuild")
import boruvka_superpixel

from lightglue import LightGlue, SuperPoint
from lightglue.utils import load_image, rbd
from lightglue import viz2d

# --- Configurações ---
torch.set_grad_enabled(False)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Rodando em: {device}")

# Instancia os modelos
extractor = SuperPoint(max_num_keypoints=2048).eval().to(device)
matcher = LightGlue(
    feature="superpoint",
    flash=True,               # Ativa aceleração de hardware para atenção se disponível
    depth_confidence=0.9,     # Limiar para parar mais cedo se estiver confiante
    width_confidence=0.95     # Remove pontos que dificilmente darão match logo cedo
).eval().to(device)

# --- Funções Modulares do Pipeline ---

def match_histogram_tensor(source_tensor, template_tensor):
    """Fase: Histogram Matching (HM)"""
    src_np = source_tensor.permute(1, 2, 0).cpu().numpy()
    tmpl_np = template_tensor.permute(1, 2, 0).cpu().numpy()
    matched_np = match_histograms(src_np, tmpl_np, channel_axis=2)
    matched_np = np.clip(matched_np, 0.0, 1.0)
    return torch.from_numpy(matched_np).permute(2, 0, 1).float()

def apply_bilateral_filter(image_tensor):
    """Fase: Bilateral Filter (BF)"""
    img_np = (image_tensor.permute(1, 2, 0).cpu().numpy() * 255.0).astype(np.uint8)
    img_np = np.ascontiguousarray(img_np)
    img_filtered = cv2.bilateralFilter(img_np, d=9, sigmaColor=75, sigmaSpace=75)
    return torch.from_numpy(img_filtered).permute(2, 0, 1).float() / 255.0

def apply_canny_edge(image_tensor):
    """Fase: Canny Edge Detection (CE)"""
    img_np = (image_tensor.permute(1, 2, 0).cpu().numpy() * 255.0).astype(np.uint8)
    img_np = np.ascontiguousarray(img_np)
    edge_map = cv2.Canny(img_np, 50, 150)
    return edge_map

def run_boruvka(image_tensor, edge_map, n_supix):
    """Fase: Boruvka Superpixel (SH)"""
    img_np = (image_tensor.permute(1, 2, 0).cpu().numpy() * 255.0).astype(np.uint8)
    img_np = np.ascontiguousarray(img_np)
    
    if edge_map is None:
        edge_map = np.zeros(img_np.shape[:2], dtype=np.uint8)
        
    bosupix = boruvka_superpixel.BoruvkaSuperpixel()
    bosupix.build_2d(img_np, edge_map)
    out = bosupix.average(n_supix, 3, img_np)
    
    return torch.from_numpy(out).permute(2, 0, 1).float() / 255.0

def run_lightglue_pipeline(img0_tensor, img1_tensor, title):
    """
    Módulo de Extração e Matching do LightGlue adaptativo.
    Retorna a figura do plot, a contagem de matches e a camada de parada precoce.
    """
    img0 = img0_tensor.to(device)
    img1 = img1_tensor.to(device)

    # Extração de Features
    feats0 = extractor.extract(img0)
    feats1 = extractor.extract(img1)

    # Matching Adaptativo: Injeta o limiar de filtragem/poda nas primeiras camadas
    matches0 = matcher({
        "image0": feats0, 
        "image1": feats1,
        "filter_threshold": 0.1  # Controla a agressividade da poda de keypoints inválidos
    })
    
    # CORREÇÃO GEOMÉTRICA: Captura a camada de parada tratando corretamente se é int (CPU) ou Tensor (GPU)
    if "stop" in matches0:
        stop_val = matches0["stop"]
        stop_layer = stop_val.item() if hasattr(stop_val, "item") else int(stop_val)
    else:
        stop_layer = -1

    # Remove batch dimension
    feats0, feats1, matches0 = [rbd(x) for x in [feats0, feats1, matches0]]

    # Filtra keypoints válidos após a poda adaptativa
    kpts0, kpts1, matches = feats0["keypoints"], feats1["keypoints"], matches0["matches"]
    m_kpts0, m_kpts1 = kpts0[matches[..., 0]], kpts1[matches[..., 1]]

    count = len(matches)
    
    # Força o fechamento de resíduos anteriores e plota no canvas correto
    plt.close('all') 
    axes = viz2d.plot_images([img0_tensor, img1_tensor])
    viz2d.plot_matches(m_kpts0, m_kpts1, color="lime", lw=0.2)
    
    # Adiciona a informação científica da camada de parada diretamente na imagem salva
    viz2d.add_text(0, f'{title}\nMatches: {count} | Stop Layer: {stop_layer}', fs=12)
    
    # Captura a figura que o viz2d acabou de desenhar no buffer
    fig = plt.gcf()
    
    return fig, count, stop_layer

# --- EXECUÇÃO EXPERIMENTAL ---

path_ref = Path("src/assets/vaso_1.jpeg")
path_cur = Path("src/assets/vaso_2.jpeg")

try:
    # 1. Carrega as imagens originais puras
    ref_orig = load_image(path_ref)
    cur_orig = load_image(path_cur)

    # 2. Configura a árvore de diretórios do experimento atual
    run_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    base_dir = Path(f"runs/run_{run_id}")
    base_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n=> Iniciando Experimento. Diretorio central: {base_dir}")

    # Valores de n baseados na granularidade recomendada pelo artigo
    n_values = [50, 75, 100, 150, 200, 300, 400]

    # Definição dos 4 Fluxos do Pipeline
    fluxos = [
        {"id": "1_Boruvka_Puro", "desc": "Boruvka Puro + LightGlue"},
        {"id": "2_HM_Boruvka", "desc": "Histogram Matching + Boruvka + LightGlue"},
        {"id": "3_HM_BF_Boruvka", "desc": "Histogram Matching + Bilateral Filter + Boruvka + LightGlue"},
        {"id": "4_HM_BF_CE_Boruvka", "desc": "Histogram Matching + Bilateral Filter + Canny + Boruvka + LightGlue"}
    ]

    for fluxo in fluxos:
        fluxo_id = fluxo["id"]
        fluxo_desc = fluxo["desc"]
        print(f"\n=== Executando Fluxo: {fluxo_desc} ===")
        
        # Cria a subpasta específica deste fluxo
        fluxo_dir = base_dir / fluxo_id
        fluxo_dir.mkdir(parents=True, exist_ok=True)
        
        # Inicializa a lista para salvar os dados do arquivo CSV
        csv_data = []
        
        for n in n_values:
            title = f"N={n}"
            
            # --- CONSTRUÇÃO DINÂMICA DAS FASES DO PIPELINE ---
            
            # Fluxo 1: Boruvka Puro (Sem HM, Sem BF, Sem Canny)
            if fluxo_id == "1_Boruvka_Puro":
                ref_proc = run_boruvka(ref_orig, edge_map=None, n_supix=n)
                cur_proc = run_boruvka(cur_orig, edge_map=None, n_supix=n)
                
            # Fluxo 2: HM + Boruvka
            elif fluxo_id == "2_HM_Boruvka":
                cur_hm = match_histogram_tensor(cur_orig, ref_orig)
                ref_proc = run_boruvka(ref_orig, edge_map=None, n_supix=n)
                cur_proc = run_boruvka(cur_hm, edge_map=None, n_supix=n)
                
            # Fluxo 3: HM + BF + Boruvka
            elif fluxo_id == "3_HM_BF_Boruvka":
                cur_hm = match_histogram_tensor(cur_orig, ref_orig)
                ref_bf = apply_bilateral_filter(ref_orig)
                cur_bf = apply_bilateral_filter(cur_hm)
                ref_proc = run_boruvka(ref_bf, edge_map=None, n_supix=n)
                cur_proc = run_boruvka(cur_bf, edge_map=None, n_supix=n)
                
            # Fluxo 4: HM + BF + Canny + Boruvka
            elif fluxo_id == "4_HM_BF_CE_Boruvka":
                cur_hm = match_histogram_tensor(cur_orig, ref_orig)
                ref_bf = apply_bilateral_filter(ref_orig)
                cur_bf = apply_bilateral_filter(cur_hm)
                
                ref_edge = apply_canny_edge(ref_bf)
                cur_edge = apply_canny_edge(cur_bf)
                
                ref_proc = run_boruvka(ref_bf, edge_map=ref_edge, n_supix=n)
                cur_proc = run_boruvka(cur_bf, edge_map=cur_edge, n_supix=n)

            # Executa o LightGlue adaptativo (capturando o stop_layer)
            fig, num_matches, stop_layer = run_lightglue_pipeline(ref_proc, cur_proc, f"{fluxo_id} - {title}")
            print(f" -> {title}: {num_matches} matches encontrados (Parou na camada: {stop_layer}).")
            
            # Salva o gráfico gerado na subpasta do fluxo
            img_filename = f"matches_n_{n}.png"
            filepath = fluxo_dir / img_filename
            fig.savefig(filepath, bbox_inches="tight", dpi=150)
            plt.close(fig)
            
            # Guarda as métricas expandidas para o CSV
            csv_data.append({
                "n_superpixels": n, 
                "matches_count": num_matches,
                "stop_layer": stop_layer  # <-- Nova coluna de métrica científica!
            })

        # --- GERAÇÃO DO RELATÓRIO CSV DO FLUXO (Atualizar cabeçalho) ---
        csv_filepath = fluxo_dir / "metrics.csv"
        with open(csv_filepath, mode="w", newline="", encoding="utf-8") as csv_file:
            # Adicionado fieldname 'stop_layer'
            writer = csv.DictWriter(csv_file, fieldnames=["n_superpixels", "matches_count", "stop_layer"])
            writer.writeheader()
            writer.writerows(csv_data)

    print("\n=> Experimento Concluído com sucesso! Verifique a pasta 'runs/' para analisar as imagens reais geradas.")

except FileNotFoundError as e:
    print(f"Erro de arquivo: Verifique os caminhos dos assets. {e}")
except Exception as e:
    print(f"Erro crítico no pipeline: {e}")