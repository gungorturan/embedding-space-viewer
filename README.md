# Embedding Space Viewer

CLIP / SigLIP gibi görüntü-metin modellerinin ortak embedding uzayını 3B ve interaktif olarak inceleme aracı.

## Çalıştırma

```bash
./run.sh            # http://127.0.0.1:8765
```

İlk açılışta Flickr30k-1K + CLIP ViT-B/32 otomatik yüklenir. Modeller ve dataset'ler Hugging Face önbelleğine iner, embedding'ler `cache/emb/` altına kaydedilir. Aynı dataset/model/n kombinasyonu ikinci kez hesaplanmaz.

## Statik sürüm (GitHub Pages)

`docs/` klasörü backend gerektirmeyen bir sürümdür. Embedding'ler ve projeksiyonlar (3 yöntem × 3 centering) önceden hesaplanmıştır. Metin sorguları ziyaretçinin tarayıcısında [transformers.js](https://github.com/huggingface/transformers.js) ile CLIP text encoder'ı (fp16, ~127 MB, ilk sorguda bir kez iner) çalıştırılarak yapılır.

```bash
uv run python -m scripts.export_static   # docs/ klasörünü yeniden üretir
```

Kombinasyonlar `scripts/export_static.py` içindeki `COMBOS` listesinden değiştirilir. Tarayıcıdaki encoder'ın fp16 olmasının sebebi şu: 8-bit quantize varyantlar PyTorch embedding'leriyle ortalama yalnızca 0.79 cosine benzerliği veriyor ve komşu sıralamasını bozuyor. fp16'da bu benzerlik ~1.000.

## Özellikler

- **Dataset'ler:** Flickr30k-1K (görüntü + 5 gerçek caption), Imagenette (10 sınıf, 8 prompt şablonu), CIFAR-10
- **Modeller:** CLIP ViT-B/32, SigLIP ViT-B/16-224
- **Projeksiyon:** PCA / UMAP / t-SNE / RTD-AE (deneysel). Seçenekler: L2 normalize; centering yok, global veya modalite başına
- **Metin sorgusu:** sorgu aynı text encoder'dan geçer ve 3B uzaya yerleştirilir (PCA/UMAP'te `transform`, t-SNE'de kNN interpolasyonu)
- **Seçim:** tıklanan noktanın yüksek boyutlu uzaydaki en yakın komşuları; Flickr'da gerçek eşlerin skoru ve sırası
- **Modality gap:** modalite merkezleri ve gap vektörü 3B'de çizilir; gap normu, eşleşen ve rastgele çift cosine'ları, modalite içi benzerlik ve retrieval top-1 metrikleri gösterilir
- **Görünüm:** nokta veya görüntü sprite'ları (thumbnail atlası), modalite veya sınıf renklendirmesi

| Etkileşim | Kısayol |
|---|---|
| Döndür / yakınlaştır / kaydır | sol sürükle / tekerlek / sağ sürükle |
| Seç | tıkla |
| Noktaya odaklan | çift tıkla |
| Seçimi temizle | `Esc` |
| Sorguya odaklan | `/` |

## RTD-AE (deneysel)

[Trofimov ve ark., *Learning Topology-Preserving Data Representations*, ICLR 2023](https://arxiv.org/abs/2302.00136). Bir autoencoder, yeniden oluşturma hatası ile Representation Topology Divergence'ın (H1 cross-barcode) toplamını en aza indirecek şekilde eğitilir. Uygulama [backend/rtd.py](backend/rtd.py) içinde ve makaledeki tanımdan bağımsız olarak yazıldı (resmi repo lisans içermiyor). Persistent homology için standart [giotto-ph](https://github.com/giotto-ai/giotto-ph) kullanılıyor.

Doğrulama: aynı girdilerde resmi koddan (giotto-ph uyarlamasıyla) birebir aynı loss ve gradyan değerleri elde edildi.

Flickr + CLIP (3000 nokta, 512 → 3) üzerindeki ölçümler:

| Yöntem | Trust | kNN@10 | Spearman ρ | Pearson | Triplet | Modalite silhouette (gerçek 0.39) |
|---|---|---|---|---|---|---|
| PCA | 0.805 | 0.049 | **0.792** | **0.857** | 0.735 | 0.65 |
| UMAP | 0.933 | 0.401 | 0.651 | 0.812 | 0.681 | 0.85 |
| t-SNE | **0.957** | **0.495** | 0.409 | 0.432 | 0.639 | 0.27 |
| AE (yalnızca MSE) | 0.951 | 0.322 | 0.558 | 0.553 | 0.672 | 0.38 |
| RTD-AE (sitedeki model) | 0.819 | 0.083 | 0.696 | 0.682 | **0.736** | 0.32 |

RTD terimi autoencoder'ın global metriklerini belirgin biçimde iyileştiriyor, ancak bu veride PCA'yı geçmiyor; yerel komşuluk ise zayıflıyor. Avantajı parametrik olması: metin sorguları eğitilmiş encoder'dan geçirilerek 3B'de kesin konumlarına yerleştiriliyor. Ayarlar makalenin 3B deneylerinden alındı: batch 80, 20 epoch ısınma ve ardından 60 epoch RTD, λ = 1.

## Yapı

```
backend/
  data.py     dataset yükleme (HF datasets), prompt şablonları
  models.py   CLIP/SigLIP encoder'ları (MODELS'e kayıt ekleyerek yeni model eklenir)
  space.py    embedding, ön işleme, projeksiyon, komşuluk, gap metrikleri
  rtd.py      türevlenebilir RTD ve RTD-AE eğitimi
  app.py      FastAPI endpoint'leri
frontend/     index.html + main.js (Three.js, build adımı yok) + backend.js (sunucu/statik veri katmanı)
scripts/      export_static.py → docs/ (GitHub Pages)
```

## Notlar

- Komşular ve metrikler modelin **gerçek** uzayında (cosine) hesaplanır. 3B görünüm yalnızca bir izdüşümdür; özellikle UMAP ve t-SNE'de global mesafeler yanıltıcı olabilir.
- UMAP'in ilk çağrısı Numba JIT derlemesi nedeniyle yavaştır (~15–30 sn). Sonraki çağrılar önbellekten gelir.
- SigLIP metinleri eğitimdeki gibi 64 token'a pad'lenir. Bu yapılmazsa embedding'ler bozulur.
