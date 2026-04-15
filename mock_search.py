"""
Mock Search Data for Testing Without API Calls
测试用的模拟数据（无需 API 调用）
"""

MOCK_PAPERS = [
    {
        "paperId": "arxiv_2001_00001",
        "title": "Deep Learning for Computer Vision: A Comprehensive Survey",
        "abstract": "This paper provides a comprehensive survey of deep learning techniques applied to computer vision tasks. We cover convolutional neural networks, transformer architectures, and their applications in image classification, object detection, and semantic segmentation.",
        "year": 2024,
        "authors": [{"name": "Zhang, Wei"}, {"name": "Li, Jun"}, {"name": "Wang, Yuting"}],
        "venue": "arXiv preprint arXiv:2001.00001",
        "citationCount": 156,
        "url": "https://arxiv.org/abs/2001.00001",
        "doi": None,
        "source": "arxiv"
    },
    {
        "paperId": "arxiv_2002_00002",
        "title": "Vision Transformers: A Survey",
        "abstract": "Vision Transformers (ViT) have emerged as a powerful alternative to convolutional neural networks for various computer vision tasks. This survey covers the architecture design, training strategies, and applications of ViT across different domains.",
        "year": 2023,
        "authors": [{"name": "Chen, Xi"}, {"name": "Liu, Yang"}, {"name": "Wu, Fei"}],
        "venue": "arXiv preprint arXiv:2002.00002",
        "citationCount": 203,
        "url": "https://arxiv.org/abs/2002.00002",
        "doi": None,
        "source": "arxiv"
    },
    {
        "paperId": "arxiv_2003_00003",
        "title": "Object Detection in Autonomous Driving: Methods and Challenges",
        "abstract": "This paper reviews recent advances in object detection for autonomous driving applications. We discuss YOLO, Faster R-CNN, and transformer-based detectors, focusing on real-time performance and accuracy trade-offs.",
        "year": 2024,
        "authors": [{"name": "Wang, Hong"}, {"name": "Zhao, Ming"}, {"name": "Liu, Kai"}],
        "venue": "arXiv preprint arXiv:2003.00003",
        "citationCount": 89,
        "url": "https://arxiv.org/abs/2003.00003",
        "doi": None,
        "source": "arxiv"
    },
    {
        "paperId": "arxiv_2004_00004",
        "title": "Self-Supervised Learning in Computer Vision: A Review",
        "abstract": "Self-supervised learning has revolutionized computer vision by enabling models to learn from unlabeled data. This paper reviews contrastive learning, masked image modeling, and their applications in pre-training vision transformers.",
        "year": 2023,
        "authors": [{"name": "Li, Yuhang"}, {"name": "Zhou, Kai"}, {"name": "Wu, Jiaxin"}],
        "venue": "arXiv preprint arXiv:2004.00004",
        "citationCount": 134,
        "url": "https://arxiv.org/abs/2004.00004",
        "doi": None,
        "source": "arxiv"
    },
    {
        "paperId": "arxiv_2005_00005",
        "title": "Medical Image Analysis Using Deep Learning: Recent Advances",
        "abstract": "Deep learning has achieved remarkable success in medical image analysis. This survey covers applications in disease diagnosis, organ segmentation, and image reconstruction, highlighting challenges specific to medical imaging.",
        "year": 2024,
        "authors": [{"name": "Yang, Xinhua"}, {"name": "Wang, Shuo"}, {"name": "Zhang, Li"}],
        "venue": "arXiv preprint arXiv:2005.00005",
        "citationCount": 67,
        "url": "https://arxiv.org/abs/2005.00005",
        "doi": None,
        "source": "arxiv"
    },
    {
        "paperId": "arxiv_2006_00006",
        "title": "Generative Adversarial Networks for Image Synthesis: A Survey",
        "abstract": "This paper provides a comprehensive review of GANs for image synthesis. We cover architectural innovations, training stability improvements, and applications in image generation, style transfer, and image-to-image translation.",
        "year": 2023,
        "authors": [{"name": "Liu, Ming-Yu"}, {"name": "Wang, Zeng}, {"name": "Chen, Ting"}],
        "venue": "arXiv preprint arXiv:2006.00006",
        "citationCount": 178,
        "url": "https://arxiv.org/abs/2006.00006",
        "doi": None,
        "source": "arxiv"
    },
    {
        "paperId": "arxiv_2007_00007",
        "title": "Attention Mechanisms in Computer Vision: A Comprehensive Review",
        "abstract": "Attention mechanisms have become integral to modern computer vision architectures. This survey covers self-attention, cross-attention, and their applications in vision transformers and beyond.",
        "year": 2024,
        "authors": [{"name": "Wu, Peng"}, {"name": "Zhang, Xiaolon"}, {"name": "Wang, Lei"}],
        "venue": "arXiv preprint arXiv:2007.00007",
        "citationCount": 145,
        "url": "https://arxiv.org/abs/2007.00007",
        "doi": None,
        "source": "arxiv"
    },
    {
        "paperId": "arxiv_2008_00008",
        "title": "Multi-Modal Learning for Visual Recognition",
        "abstract": "Multi-modal learning combines visual data with text, audio, or other modalities. This paper reviews recent approaches to vision-language models, contrastive learning, and their applications in visual recognition.",
        "year": 2023,
        "authors": [{"name": "Chen, Runing"}, {"name": "Wang, Xiaolong"}, {"name": "Li, Leo"}],
        "venue": "arXiv preprint arXiv:2008.00008",
        "citationCount": 192,
        "url": "https://arxiv.org/abs/2008.00008",
        "doi": None,
        "source": "arxiv"
    },
]

def get_mock_papers(count: int = None):
    """获取模拟论文数据"""
    if count:
        return MOCK_PAPERS[:count]
    return MOCK_PAPERS
