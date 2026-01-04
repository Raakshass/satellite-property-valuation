# Multimodal Architecture Diagram

```mermaid
graph TB
    subgraph Input["Input Data"]
        A[Satellite Images<br/>224x224 RGB]
        B[Tabular Features<br/>35 features]
    end
    
    subgraph ImageBranch["Image Branch"]
        C[ResNet18<br/>Pretrained on ImageNet]
        D[Feature Extractor<br/>512 features]
        E[Dense Layer<br/>512 → 256]
        F[Image Embedding<br/>256D]
    end
    
    subgraph TabularBranch["Tabular Branch"]
        G[Dense Layer<br/>35 → 128]
        H[Dense Layer<br/>128 → 128]
        I[Tabular Embedding<br/>128D]
    end
    
    subgraph Fusion["Late Fusion Layer"]
        J[Concatenate<br/>256D + 128D = 384D]
        K[Dense Layer<br/>384 → 256]
        L[Dense Layer<br/>256 → 128]
        M[Output Layer<br/>128 → 1]
    end
    
    N[Predicted Price]
    
    A --> C
    C --> D
    D --> E
    E --> F
    B --> G
    G --> H
    H --> I
    F --> J
    I --> J
    J --> K
    K --> L
    L --> M
    M --> N
    
    style A fill:#e1f5ff
    style B fill:#e1f5ff
    style F fill:#ffe1e1
    style I fill:#ffe1e1
    style N fill:#e1ffe1
