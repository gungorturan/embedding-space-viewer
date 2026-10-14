"""RTD-AE: topolojiyi koruyan autoencoder projeksiyonu.

Trofimov, Cherniavskii, Tulchinskii, Balabin, Burnaev, Barannikov,
"Learning Topology-Preserving Data Representations", ICLR 2023 (arXiv:2302.00136).

Bu dosya makaledeki tanımdan bağımsız olarak yazılmıştır (resmi repo lisans içermiyor).
Representation Topology Divergence, iki mesafe matrisinden kurulan "ikizlenmiş" grafiğin
H1 persistence barkodudur (R-Cross-Barcode):

        M = | 0    Aᵀ        |
            | A    min(A, B) |

RTD(A, B) = Σ (ölüm − doğum) uzunlukları. Türev için her barkod aralığının doğum ve ölüm
kenarları persistent homology'den alınır; ölüm değeri, sağ alt blokta min(A, B) yerine
latent mesafelerden okunur, böylece gradyan latent temsile akar (makaledeki altgradyan şeması).
"""

from __future__ import annotations

import numpy as np
import torch
from gph import ripser_parallel
from torch import nn

# Makalenin 3B deneylerindeki ayarlar; Flickr+CLIP üzerinde en iyi değerler ~80. epoch civarında
WARMUP_EPOCHS = 20
RTD_EPOCHS = 60
BATCH = 80
CARD = 50          # her adımda kullanılan en uzun H1 aralığı sayısı
LR = 5e-4
HIDDEN = 512
LAMBDA = 1.0


def _h1_edges(M: np.ndarray, card: int) -> np.ndarray:
    """En uzun `card` H1 aralığının [doğum v0, v1, ölüm v2, v3] kenar indeksleri."""
    rc = ripser_parallel(M, maxdim=1, metric="precomputed", collapse_edges=False,
                         return_generators=True, n_threads=-1)
    gens = rc["gens"][1]
    if not len(gens) or not len(gens[0]):
        return np.zeros((0, 4), np.int64)
    g = np.asarray(gens[0], np.int64).reshape(-1, 4)
    lengths = M[g[:, 2], g[:, 3]] - M[g[:, 0], g[:, 1]]
    return g[np.argsort(-lengths, kind="stable")[:card]]


def _cross_barcode_length(A: torch.Tensor, B: torch.Tensor, Dz: torch.Tensor, card: int) -> torch.Tensor:
    zero = torch.zeros_like(A)
    M = torch.cat([torch.cat([zero, A.T], 1), torch.cat([A, torch.minimum(A, B)], 1)], 0)
    route = torch.cat([torch.cat([zero, A.T], 1), torch.cat([A, Dz], 1)], 0)
    g = _h1_edges(M.detach().cpu().numpy().astype(np.float64), card)
    if not len(g):
        return M.new_zeros(())
    g = torch.from_numpy(g)
    return (route[g[:, 2], g[:, 3]] - M[g[:, 0], g[:, 1]]).abs().sum()


def rtd_loss(Dx: torch.Tensor, Dz: torch.Tensor, card: int = CARD) -> torch.Tensor:
    """Simetrik RTD: ½ (RTD(X, Z) + RTD(Z, X)). Dx sabit, Dz gradyan taşır."""
    return 0.5 * (_cross_barcode_length(Dx, Dz, Dz, card) + _cross_barcode_length(Dz, Dx, Dz, card))


def _mlp(i: int, h: int, o: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(i, h), nn.ReLU(), nn.Linear(h, h), nn.ReLU(),
                         nn.Linear(h, h), nn.ReLU(), nn.Linear(h, o))


def train(X: np.ndarray, seed: int = 0, progress=lambda msg: None) -> nn.Sequential:
    """X (N × d) için 3B encoder eğitir: önce yalnızca yeniden oluşturma, sonra + λ·RTD."""
    gen = torch.Generator().manual_seed(seed)
    torch.manual_seed(seed)
    Xt = torch.from_numpy(np.ascontiguousarray(X, np.float32))
    d = Xt.shape[1]
    enc, dec = _mlp(d, HIDDEN, 3), _mlp(3, HIDDEN, d)
    # Latent mesafeler için öğrenilebilir ölçek: RTD ölçeğe duyarlıdır
    scale = nn.Parameter(torch.ones(1))
    opt = torch.optim.AdamW([*enc.parameters(), *dec.parameters(), scale], lr=LR)
    mse = nn.MSELoss()
    total = WARMUP_EPOCHS + RTD_EPOCHS
    for ep in range(total):
        use_rtd = ep >= WARMUP_EPOCHS
        progress(f"RTD-AE eğitiliyor… epoch {ep + 1}/{total}")
        perm = torch.randperm(len(Xt), generator=gen)
        for s in range(0, len(Xt) - BATCH + 1, BATCH):
            x = Xt[perm[s:s + BATCH]]
            z = enc(x)
            loss = mse(dec(z), x)
            if use_rtd:
                Dx = torch.cdist(x, x) / np.sqrt(d)
                Dz = scale * torch.cdist(z, z) / np.sqrt(3)
                loss = loss + LAMBDA * rtd_loss(Dx, Dz)
            opt.zero_grad()
            loss.backward()
            opt.step()
    return enc.eval()


def mlp_layers(enc: nn.Sequential) -> list[tuple[np.ndarray, np.ndarray]]:
    """Linear katmanların (W, b) çiftleri; tarayıcıda aynı ileri geçiş için."""
    return [(m.weight.detach().numpy(), m.bias.detach().numpy()) for m in enc if isinstance(m, nn.Linear)]
