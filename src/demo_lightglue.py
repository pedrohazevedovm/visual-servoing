from pathlib import Path

from lightglue import LightGlue, SuperPoint, DISK
from lightglue.utils import load_image, rbd
from lightglue import viz2d
import torch


torch.set_grad_enabled(False)
images = Path("assets")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

extractor = SuperPoint(max_num_keypoints=2048).eval().to(device)
matcher = LightGlue(feature="superpoint").eval().to(device)

ref_img = load_image(Path("src/assets/ref_img.jpeg"))
current_img = load_image(Path("src/assets/current_img.jpeg"))

ref_feats = extractor.extract(ref_img.to(device))
current_feats = extractor.extract(current_img.to(device))

matches = matcher({"image0": ref_feats, "image1": current_feats})
ref_feats, current_feats, matches = [rbd(x) for x in [ref_feats, current_feats, matches]] # Remove batch dimension

kpts_ref, kpts_current, matches = ref_feats["keypoints"], current_feats["keypoints"], matches["matches"]
m_kpts_ref, m_kpts_current = kpts_ref[matches[..., 0]], kpts_current[matches[..., 1]]

axes = viz2d.plot_images([ref_img, current_img])
viz2d.plot_matches(m_kpts_ref, m_kpts_current, color="lime", lw=0.2)
# viz2d.add_text(0, f'Stop after {matches["stop"]} layers', fs=20)

from lightglue import match_pair
feats0, feats1, matches01 = match_pair(extractor, matcher, ref_img, current_img)