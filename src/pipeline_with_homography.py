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

extractor = SuperPoint(max_num_keypoints=2048).eval().to(device)
matcher = LightGlue(feature="superpoint", flash=True).eval().to(device)

# --- Funções Modulares Existentes ---

def match_histogram_tensor(source_tensor, template_tensor):
    src_np = source_tensor.permute(1, 2, 0).cpu().numpy()
    tmpl_np = template_tensor.permute(1, 2, 0).cpu().numpy()
    matched_np = match_histograms(src_np, tmpl_np, channel_axis=2)
    matched_np = np.clip(matched_np, 0.0, 1.0)
    return torch.from_numpy(matched_np).permute(2, 0, 1).float()

def apply_bilateral_filter(image_tensor):
    img_np = (image_tensor.permute(1, 2, 0).cpu().numpy() * 255.0).astype(np.uint8)
    img_np = np.ascontiguousarray(img_np)
    img_filtered = cv2.bilateralFilter(img_np, d=9, sigmaColor=75, sigmaSpace=75)
    return torch.from_numpy(img_filtered).permute(2, 0, 1).float() / 255.0

def apply_canny_edge(image_tensor):
    img_np = (image_tensor.permute(1, 2, 0).cpu().numpy() * 255.0).astype(np.uint8)
    img_np = np.ascontiguousarray(img_np)
    return cv2.Canny(img_np, 50, 150)

def run_boruvka(image_tensor, edge_map, n_supix):
    img_np = (image_tensor.permute(1, 2, 0).cpu().numpy() * 255.0).astype(np.uint8)
    img_np = np.ascontiguousarray(img_np)
    if edge_map is None:
        edge_map = np.zeros(img_np.shape[:2], dtype=np.uint8)
    bosupix = boruvka_superpixel.BoruvkaSuperpixel()
    bosupix.build_2d(img_np, edge_map)
    out = bosupix.average(n_supix, 3, img_np)
    return torch.from_numpy(out).permute(2, 0, 1).float() / 255.0

def crop_center_tensor(image_tensor, pct_w=0.3, pct_h=0.3):
    C, H, W = image_tensor.shape
    crop_w, crop_h = int(W * pct_w), int(H * pct_h)
    x_start, y_start = (W - crop_w) // 2, (H - crop_h) // 2
    cropped_tensor = image_tensor[:, y_start:y_start+crop_h, x_start:x_start+crop_w]
    return cropped_tensor, (x_start, y_start)

# --- NOVA FUNÇÃO: ESTIMATIVA DE HOMOGRAFIA ---

def estimate_homography(pts0, pts1):
    """
    Estima a matriz de Homografia robusta usando RANSAC.
    pts0: pontos da imagem de referência (N, 2) em numpy
    pts1: pontos da imagem atual (N, 2) em numpy
    Retorna: Matriz H (3, 3), máscara de inliers e a contagem de inliers válidos.
    """
    if len(pts0) < 4:
        print("Aviso: Menos de 4 pontos para homografia.")
        return None, None, 0
    
    # RANSAC com limiar de 3 pixels para considerar um ponto como correto (inlier)
    H, mask = cv2.findHomography(pts1, pts0, cv2.RANSAC, 3.0)
    
    inliers_count = int(np.sum(mask)) if mask is not None else 0
    return H, mask, inliers_count

# --- PIPELINE ATUALIZADO ---

def run_pipeline_with_homography(img0_orig, img1_orig, title, use_roi=True, pct_w=0.3, pct_h=0.3):
    """
    Roda o pipeline completo do LightGlue com estimativa de homografia por RANSAC.
    
    use_roi (bool): Se True, restringe a extração de features à região central definida.
                    Se False, processa as imagens em tamanho original completo.
    """
    # 1. Definição da Região de Busca (Com ROI ou Imagem Completa)
    if use_roi:
        # Aplica o Crop Central (ROI) e armazena os offsets de translação
        img0_proc, offset0 = crop_center_tensor(img0_orig, pct_w, pct_h)
        img1_proc, offset1 = crop_center_tensor(img1_orig, pct_w, pct_h)
    else:
        # Usa as imagens originais inteiras (offsets zerados)
        img0_proc, offset0 = img0_orig, (0, 0)
        img1_proc, offset1 = img1_orig, (0, 0)
    
    # 2. Extração de Features (na ROI ou na Imagem Cheia)
    feats0 = extractor.extract(img0_proc.to(device))
    feats1 = extractor.extract(img1_proc.to(device))

    # 3. Matching Adaptativo LightGlue
    matches0 = matcher({"image0": feats0, "image1": feats1, "filter_threshold": 0.1})
    
    # Tratamento seguro se 'stop' for int (CPU) ou Tensor (GPU)
    if "stop" in matches0:
        stop_val = matches0["stop"]
        stop_layer = stop_val.item() if hasattr(stop_val, "item") else int(stop_val)
    else:
        stop_layer = -1

    # Remove batch dimension
    feats0, feats1, matches0 = [rbd(x) for x in [feats0, feats1, matches0]]

    kpts0, kpts1 = feats0["keypoints"], feats1["keypoints"]
    matches = matches0["matches"]
    
    m_kpts0_crop = kpts0[matches[..., 0]]
    m_kpts1_crop = kpts1[matches[..., 1]]

    # 4. Translação de Coordenadas de volta para o Espaço Original (Se use_roi=False, offsets são 0)
    offset0_tensor = torch.tensor([offset0[0], offset0[1]], device=m_kpts0_crop.device)
    offset1_tensor = torch.tensor([offset1[0], offset1[1]], device=m_kpts1_crop.device)
    
    m_kpts0_orig = (m_kpts0_crop + offset0_tensor).cpu().numpy()
    m_kpts1_orig = (m_kpts1_crop + offset1_tensor).cpu().numpy()

    # 5. Cálculo da Homografia com RANSAC sobre os pontos convertidos
    H, mask, inliers = estimate_homography(m_kpts0_orig, m_kpts1_orig)
    
    # 6. Renderização Visual dos Resultados
    plt.close('all') 
    axes = viz2d.plot_images([img0_orig, img1_orig])
    
    # Plota apenas os matches considerados válidos (Inliers) pelo RANSAC
    if mask is not None and len(mask) > 0:
        inliers_mask = mask.ravel() == 1
        viz2d.plot_matches(
            torch.tensor(m_kpts0_orig[inliers_mask]), 
            torch.tensor(m_kpts1_orig[inliers_mask]), 
            color="lime", lw=0.3
        )
    
    W0, H0 = img0_orig.shape[2], img0_orig.shape[1]

    # Desenha os delimitadores geométricos na tela se a ROI estiver ativa
    if use_roi:
        # Desenha o Retângulo da ROI (Vermelho) na imagem de Referência (Esquerda)
        rect_roi = plt.Rectangle((offset0[0], offset0[1]), W0*pct_w, H0*pct_h, 
                                 edgecolor="red", facecolor="none", linestyle="--", linewidth=1.5)
        plt.gca().add_patch(rect_roi)
    
    # Visualização da Homografia: Projeta a região correspondente na Imagem Atual (Direita)
    if H is not None:
        if use_roi:
            # Se usou ROI, projeta o retângulo central deformado
            cantos_base = np.array([
                [offset0[0], offset0[1]],
                [offset0[0] + W0*pct_w, offset0[1]],
                [offset0[0] + W0*pct_w, offset0[1] + H0*pct_h],
                [offset0[0], offset0[1] + H0*pct_h]
            ], dtype=np.float32).reshape(-1, 1, 2)
        else:
            # Se não usou ROI, projeta as próprias bordas externas da imagem inteira (0 a W0, 0 a H0)
            cantos_base = np.array([
                [0, 0],
                [W0, 0],
                [W0, H0],
                [0, H0]
            ], dtype=np.float32).reshape(-1, 1, 2)
        
        cantos_projetados = cv2.perspectiveTransform(cantos_base, np.linalg.inv(H))
        cantos_plot = cantos_projetados.squeeze() + np.array([W0, 0])
        
        # Desenha o polígono azul mostrando onde a região correspondente foi mapeada
        polygon = plt.Polygon(cantos_plot, edgecolor="cyan", facecolor="none", linewidth=2.5, linestyle="-")
        plt.gca().add_patch(polygon)

    roi_status = "ROI Ativa (30%)" if use_roi else "Imagem Completa"
    viz2d.add_text(0, f'{title} [{roi_status}]\nMatches Totais: {len(matches)} | Inliers RANSAC: {inliers}\nStop Layer: {stop_layer}', fs=10)
    fig = plt.gcf()
    
    return fig, len(matches), inliers, stop_layer


if __name__ == "__main__":
    # 1. Carrega as imagens originais
    path_ref = Path("src/assets/vaso_1.jpeg")
    path_cur = Path("src/assets/vaso_2.jpeg")
    
    if not path_ref.exists() or not path_cur.exists():
        print(f"Erro: Certifique-se de que as imagens existem em 'src/assets/'.")
        sys.exit(1)

    ref_tensor = load_image(path_ref)
    cur_tensor = load_image(path_cur)

    # 2. Aplica as etapas preliminares nas imagens originais
    # Opcional: Adicionar apply_bilateral_filter se quiser rodar o Fluxo 4 completo
    ref_edge = apply_canny_edge(ref_tensor)
    cur_edge = apply_canny_edge(cur_tensor)

    # 3. Executa a redução por Superpixels do Borůvka (SH)
    print("Processando segmentação por Boruvka...")
    ref_processada = run_boruvka(ref_tensor, edge_map=ref_edge, n_supix=100)
    cur_processada = run_boruvka(cur_tensor, edge_map=cur_edge, n_supix=100)

    # 4. Alimenta o pipeline com ROI e estimativa de Homografia robusta
    print("Calculando casamento com LightGlue e estimando Homografia...")
    fig, total, inliers, stop = run_pipeline_with_homography(
        img0_orig=ref_processada, 
        img1_orig=cur_processada, 
        title="Teste Homografia",
        use_roi=False
    )

    # 5. EXIBIÇÃO E SALVAMENTO DA IMAGEM GERADA
    # Define o título da janela gráfica
    fig.canvas.manager.set_window_title("Resultado: Homografia e Rastreamento de ROI")
    
    # Salva o arquivo em disco para verificação fora do terminal
    output_filename = "resultado_homografia_roi.png"
    fig.savefig(output_filename, bbox_inches="tight", dpi=150)
    print(f"\n=> Sucesso! Gráfico salvo em alta resolução como: '{output_filename}'")
    
    # Abre a janela interativa do Matplotlib na sua tela
    print("Abrindo visualização gráfica... Feche a janela para encerrar o script.")
    plt.show()