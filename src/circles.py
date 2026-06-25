import sys
from pathlib import Path
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from skimage.exposure import match_histograms

# Garanta que o path do Boruvka esteja correto
sys.path.insert(0, "boruvka-superpixel/pybuild")
import boruvka_superpixel

from lightglue import LightGlue, SuperPoint
from lightglue.utils import load_image, rbd
from lightglue import viz2d

# --- Configurações Locais ---
torch.set_grad_enabled(False)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

extractor = SuperPoint(max_num_keypoints=2048).eval().to(device)
matcher = LightGlue(feature="superpoint", flash=True).eval().to(device)

# --- Funções do Pipeline ---
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

def estimate_homography(pts0, pts1):
    if len(pts0) < 4:
        return None, None, 0
    H, mask = cv2.findHomography(pts0, pts1, cv2.RANSAC, 3.0)
    inliers_count = int(np.sum(mask)) if mask is not None else 0
    return H, mask, inliers_count

def gerar_pontos_circulo_virtual(cx, cy, raio, num_pontos=8):
    pontos = []
    for i in range(num_pontos):
        angulo = 2 * np.pi * i / num_pontos
        x = cx + raio * np.cos(angulo)
        y = cy + raio * np.sin(angulo)
        pontos.append([x, y])
    return np.array(pontos, dtype=np.float32).reshape(-1, 1, 2)


if __name__ == "__main__":
    path_ref = Path("src/assets/vaso_1.jpeg")
    path_cur = Path("src/assets/vaso_2.jpeg")
    
    if not path_ref.exists() or not path_cur.exists():
        print("Erro: Imagens de teste não encontradas.")
        sys.exit(1)
        
    ref_tensor = load_image(path_ref)
    cur_tensor = load_image(path_cur)
    w_img, h_img = ref_tensor.shape[2], ref_tensor.shape[1]
    
    # Executa pré-processamento completo
    cur_hm = match_histogram_tensor(cur_tensor, ref_tensor)
    ref_bf = apply_bilateral_filter(ref_tensor)
    cur_bf = apply_bilateral_filter(cur_hm)
    ref_edge = apply_canny_edge(ref_tensor)
    cur_edge = apply_canny_edge(cur_hm)
    ref_proc = run_boruvka(ref_bf, edge_map=ref_edge, n_supix=100)
    cur_proc = run_boruvka(cur_bf, edge_map=cur_edge, n_supix=100)
    
    # LightGlue Matching
    feats0 = extractor.extract(ref_proc.to(device))
    feats1 = extractor.extract(cur_proc.to(device))
    matches0 = matcher({"image0": feats0, "image1": feats1, "filter_threshold": 0.1})
    
    feats0, feats1, matches0 = [rbd(x) for x in [feats0, feats1, matches0]]
    kpts0, kpts1 = feats0["keypoints"].cpu().numpy(), feats1["keypoints"].cpu().numpy()
    matches = matches0["matches"].cpu().numpy()
    
    m_kpts0 = kpts0[matches[..., 0]]
    m_kpts1 = kpts1[matches[..., 1]]
    
    H, mask, inliers_totais = estimate_homography(m_kpts0, m_kpts1)
    if H is None:
        sys.exit(1)

    # Configuração geométrica do círculo central na referência
    cx, cy = w_img / 2.0, h_img / 2.0
    raio_virtual = 350.0
    pontos_ref = gerar_pontos_circulo_virtual(cx, cy, raio_virtual, 12)
    pontos_estimados_elipse = cv2.perspectiveTransform(pontos_ref, H)
    
    pts_ref_flat = pontos_ref.squeeze()
    pts_est_flat = pontos_estimados_elipse.squeeze()

    # --- NOVO: FILTRAGEM DE PONTOS DENTRO DO POLÍGONO ---
    indices_dentro_poligono = []
    inliers_ransac_dentro_poligono = 0
    
    mask_flat = mask.ravel() if mask is not None else np.zeros(len(matches))

    for idx, pt in enumerate(m_kpts0):
        # Testa se o ponto da imagem de referência está contido no polígono alvo (s*)
        # Retorna positivo se estiver dentro, negativo se estiver fora
        dist = cv2.pointPolygonTest(pts_ref_flat, (float(pt[0]), float(pt[1])), False)
        if dist >= 0:
            indices_dentro_poligono.append(idx)
            # Verifica se esse ponto interno também foi validado como inlier pelo RANSAC
            if mask_flat[idx] == 1:
                inliers_ransac_dentro_poligono += 1

    matches_dentro_poligono = len(indices_dentro_poligono)

    # --- CÁLCULO DOS ERROS GEOMÉTRICOS ---
    erro_x = np.mean(pts_ref_flat[:, 0] - pts_est_flat[:, 0])
    erro_y = np.mean(pts_ref_flat[:, 1] - pts_est_flat[:, 1])
    angulos_ref = np.arctan2(pts_ref_flat[:, 1] - cy, pts_ref_flat[:, 0] - cx)
    angulos_est = np.arctan2(pts_est_flat[:, 1] - (cy - erro_y), pts_est_flat[:, 0] - (cx - erro_x))
    erro_angular_graus = np.degrees(np.mean(np.unwrap(angulos_ref - angulos_est)))

    # --- PLOT DETALHADO ---
    plt.figure(figsize=(14, 7))
    axes = viz2d.plot_images([ref_tensor, cur_tensor])
    
    # 1. Plota todos os matches normais em verde oliva (fino)
    viz2d.plot_matches(torch.tensor(m_kpts0), torch.tensor(m_kpts1), color="yellowgreen", lw=0.1)
    
    # 2. Destaca com linhas mais grossas em azul ciano apenas os matches de dentro do círculo
    if matches_dentro_poligono > 0:
        pts0_internos = m_kpts0[indices_dentro_poligono]
        pts1_internos = m_kpts1[indices_dentro_poligono]
        viz2d.plot_matches(torch.tensor(pts0_internos), torch.tensor(pts1_internos), color="cyan", lw=0.5)

    # Desenha o círculo alvo (verde) na imagem de referência
    poly_ref = plt.Polygon(pts_ref_flat, edgecolor='lime', facecolor='none', linewidth=2, label="Alvo")
    plt.gca().add_patch(poly_ref)
    
    # Desenha a elipse projetada (vermelha) na imagem atual
    pts_est_shifted = pts_est_flat.copy()
    pts_est_shifted[:, 0] += w_img
    poly_est = plt.Polygon(pts_est_shifted, edgecolor='red', facecolor='none', linewidth=2, linestyle='--')
    plt.gca().add_patch(poly_est)
    
    # Legenda explicativa rica em metadados acadêmicos
    texto_relatorio = (
        f"MÉTRICAS DO SISTEMA (ZONA CENTRAL)\n"
        f"-----------------------------------------\n"
        f"Matches Totais do LightGlue: {len(matches)}\n"
        f"Matches Internos ao Círculo: {matches_dentro_poligono}\n"
        f"Inliers do RANSAC na Região Útil: {inliers_ransac_dentro_poligono}\n"
        f"Inliers Totais da Imagem Completa: {inliers_totais}\n"
        f"-----------------------------------------\n"
        f"Erro Linear X: {erro_x:.2f} px | Erro Linear Y: {erro_y:.2f} px\n"
        f"Erro Angular Z (Torção): {erro_angular_graus:.3f}°"
    )
    viz2d.add_text(0, texto_relatorio, fs=9, color="white")
    
    print("\nExibindo gráfico com discriminação de matches por região...")
    plt.show()