# DVIT-UNet: High-Precision Tea Plantation Mapping from UAV Imagery



> **Abstract**: The world's major tea plantations are concentrated in mountainous and hilly areas with highly fragmented plots due to diverse management practices. While remote sensing and deep learning are widely used in crop mapping, fine-grained extraction of tea plots from ultra-high-resolution UAV imagery remains challenging. Existing methods typically achieve only coarse identification of tea-growing areas or focus on small-scale orchards, failing to meet large-scale application needs. We propose **DVIT-UNet**, a novel architecture integrating Vision Transformer and dilated convolution into the UNet framework. Our model effectively captures global semantic relationships and multi-scale local context to address blurred boundaries, high intra-class heterogeneity, and spectral similarity with surrounding vegetation. Using only RGB UAV imagery (no spectral/height data), DVIT-UNet achieves county-level scalability with state-of-the-art performance:  
> **mIoU: 90.48%** | **F1: 94.99%** | **Recall (UA): 94.39%** | **Precision (PA): 95.60%** | **MCC: 91.13%**  
> The model robustly delineates fragmented plots while suppressing false positives in complex backgrounds, providing practical technical support for precision tea plantation management.

## 🌍 Research Area
![area](https://github.com/user-attachments/assets/37044147-e42c-45fb-b276-61b0e2e4f774)
*County-level tea plantation region in mountainous terrain (UAV RGB composite). Note the fragmented plot structure and complex background vegetation.*

## 🧠 Model Architecture
![model (2)](https://github.com/user-attachments/assets/e3d69747-62a6-4d3c-b582-f95a9b6d1b97)

*DVIT-UNet integrates: (1) Vision Transformer blocks for global context capture, (2) Multi-scale dilated convolutions for boundary refinement, and (3) UNet's hierarchical feature fusion. Operates exclusively on RGB inputs.*

## ✨ Key Innovations
- **Fragmentation-aware design**: Handles sub-hectare tea plots with irregular boundaries
- **RGB-only efficiency**: Eliminates dependency on multispectral/DSM data
- **Hybrid context modeling**: Combines ViT's global attention with dilated CNN's local perception
- **County-scale deployment**: Processes 10,000+ km² regions with standard hardware
- **Robust background suppression**: 37.2% reduction in false positives vs. baseline models

## 📊 Performance Comparison
| Model          | mIoU (%) | F1 (%)  | MCC (%) | Params (M) |
|----------------|----------|---------|---------|------------|
| **DVIT-UNet**  | **90.48**| **94.99**| **91.13**| 18.7       |
| UNet++         | 86.21    | 91.35   | 84.60   | 9.0        |
| DeepLabv3+     | 85.77    | 90.88   | 83.92   | 41.2       |
| SegFormer      | 87.33    | 92.17   | 86.05   | 27.1       |
| *Full comparison with 7 SOTA models in paper*

## 🚀 Practical Value
- Enables **precision yield estimation** at individual plot level
- Supports **ecological monitoring** of soil erosion and biodiversity
- Facilitates **management optimization** for fragmented smallholder farms
- Provides **scalable framework** adaptable to other perennial crops

## 📂 Repository Structure
