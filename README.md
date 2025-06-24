Comparative Analysis of RAG System Performance Across Open-Source Large Language Models: A Foundation for Adaptive Knowledge Integration
Comparative analysis of RAG performance on Open Source LLM
Abstract
This study presents a systematic evaluation of Retrieval-Augmented Generation (RAG) systems across multiple open-source large language models, establishing baseline performance metrics and identifying optimization opportunities. Using freely available models and datasets, we implement and evaluate RAG configurations to understand how different LLM architectures leverage external knowledge. This research provides foundational insights for developing adaptive RAG systems and serves as a precursor to comprehensive PhD dissertation work in information systems.
1. Introduction
1.1 Problem Statement
Current RAG implementations lack systematic evaluation across different LLM architectures, making it difficult to select optimal configurations for specific applications. Most studies focus on single models or proprietary systems, limiting reproducibility and practical application.
1.2 Research Questions
•	How do different open-source LLM architectures (LLaMA, Mistral, CodeLlama) perform with standard RAG configurations?
•	What retrieval strategies work best for different model sizes and capabilities?
•	How can we establish baseline metrics for future adaptive RAG research?
1.3 Scope and Limitations
This initial study focuses on 3-4 open-source models using publicly available datasets and standard evaluation metrics. The goal is to establish a foundation for larger-scale research during PhD studies.
2. Methodology (Feasible Approach)
2.1 Model Selection (Free/Open-Source)
Primary Models:
•	LLaMA 2 7B/13B (Meta) - General purpose, widely adopted
•	Mistral 7B - Efficient, high-performance
•	CodeLlama 7B - Code-specialized variant
•	Llama 3 8B - Latest iteration for comparison
Rationale: These models are freely available, can run on consumer hardware, and represent different architectural approaches.
2.2 Dataset Selection (Public Datasets)
Knowledge Bases:
•	MS MARCO - Web search dataset (11M passages)
•	Natural Questions - Wikipedia-based Q&A
•	SQuAD 2.0 - Reading comprehension
•	HotpotQA - Multi-hop reasoning
Benefits: Well-established benchmarks with existing baselines, allowing for meaningful comparisons.
2.3 Implementation Strategy
2.3.1 Minimal Viable Architecture
Documents → Chunking → Embedding (Sentence-BERT) → 
Vector Store (ChromaDB/FAISS) → Retrieval → LLM Generation
Technology Stack:
•	Embedding Model: all-MiniLM-L6-v2 (free, efficient)
•	Vector Store: ChromaDB or FAISS (local, no cost)
•	LLM Interface: Hugging Face Transformers
•	Evaluation: Custom Python scripts using ROUGE, BLEU, BERTScore
2.3.2 Experimental Design
Three RAG Configurations:
1.	Basic RAG: Top-3 retrieved documents, simple concatenation
2.	Enhanced RAG: Top-5 documents with relevance scoring
3.	Optimized RAG: Dynamic document selection based on query type
Evaluation Metrics:
•	Retrieval: Recall@K, MRR, NDCG
•	Generation: ROUGE-L, BLEU, BERTScore
•	Efficiency: Response time, token usage
2.4 Hardware Requirements (Budget-Friendly)
•	GPU: Single RTX 4090 or A100 (cloud rental: ~$1-2/hour)
•	RAM: 32GB minimum for 7B models
•	Storage: 1TB SSD for models and datasets
•	Estimated Cost: $500-1000 for compute over 2-3 months
3. Experimental Plan
3.1 Phase 1: Baseline Implementation (Month 1)
•	Set up development environment
•	Implement basic RAG pipeline
•	Test with single model (LLaMA 2 7B)
•	Validate evaluation metrics
3.2 Phase 2: Multi-Model Evaluation (Month 2)
•	Extend to all selected models
•	Run systematic experiments across datasets
•	Collect performance metrics
•	Identify patterns and anomalies
3.3 Phase 3: Analysis and Optimization (Month 3)
•	Statistical analysis of results
•	Identify best practices per model
•	Propose optimization strategies
•	Write paper and prepare submission
4. Expected Contributions
4.1 Academic Contributions
•	Systematic Comparison: First comprehensive evaluation of RAG across multiple open-source models
•	Baseline Establishment: Performance benchmarks for future research
•	Methodology Framework: Reproducible evaluation protocol
4.2 Practical Contributions
•	Model Selection Guide: Recommendations for different use cases
•	Implementation Best Practices: Optimal configurations for each model
•	Open-Source Toolkit: Code and data for community use
4.3 PhD Foundation
•	Research Direction: Identifies promising areas for dissertation work
•	Methodology Validation: Tests approaches for larger studies
•	Publication Record: Establishes research credibility
5. Realistic Timeline
Month 1: Setup and Baseline
•	Week 1-2: Literature review, environment setup
•	Week 3-4: Basic RAG implementation and testing
Month 2: Experimentation
•	Week 5-6: Multi-model implementation
•	Week 7-8: Systematic evaluation and data collection
Month 3: Analysis and Writing
•	Week 9-10: Results analysis and interpretation
•	Week 11-12: Paper writing and revision
6. Target Venues
6.1 Primary Targets (Conferences)
•	SIGIR (Information Retrieval) - July deadline
•	EMNLP (Natural Language Processing) - May deadline
•	CIKM (Information and Knowledge Management) - May deadline
6.2 Secondary Targets (Workshops/Journals)
•	ArXiv preprint - Immediate dissemination
•	ACL Workshop on Retrieval-Augmented Generation
•	Information Systems Journal - Relevant to PhD field
7. Resource Management
7.1 Time Investment
•	Daily: 4-6 hours (manageable with other commitments)
•	Weekly: 25-30 hours
•	Total: ~300 hours over 3 months
7.2 Cost Breakdown
•	Cloud Compute: $300-500
•	Software/Tools: $0 (open-source)
•	Conference Fees: $500-800 (if accepted)
•	Total: $800-1300
7.3 Risk Mitigation
•	Technical Issues: Start with simpler models, scale up gradually
•	Compute Constraints: Use Google Colab Pro or Kaggle for initial experiments
•	Timeline Delays: Focus on core experiments, defer optimizations if needed
8. Future Research Directions (PhD Dissertation)
8.1 Immediate Extensions
•	Adaptive RAG Systems: Dynamic strategy selection based on query analysis
•	Domain-Specific Optimization: Specialized RAG for different fields
•	Multi-Modal Integration: Combining text, image, and structured data
8.2 Long-Term Research Goals
•	Theoretical Framework: Mathematical models of RAG performance
•	Industrial Applications: Enterprise-scale RAG deployment
•	Ethical Considerations: Bias and fairness in knowledge retrieval
9. Success Metrics
9.1 Short-Term (Paper Acceptance)
•	Reproducible Results: Clear methodology and code availability
•	Novel Insights: New understanding of model-RAG interactions
•	Practical Value: Actionable recommendations for practitioners
9.2 Medium-Term (PhD Preparation)
•	Research Network: Connections with RAG/IR researchers
•	Technical Skills: Proficiency with LLMs and evaluation methods
•	Academic Credibility: Published work in relevant venue
9.3 Long-Term (Career Impact)
•	Dissertation Foundation: Clear research direction and methodology
•	Industry Relevance: Practical applications and collaborations
•	Academic Recognition: Established expertise in RAG systems
10. Conclusion
This focused research project provides a realistic pathway to contribute meaningful insights to the RAG literature while establishing a foundation for comprehensive PhD dissertation work. By constraining scope to open-source models and public datasets, the study remains feasible within budget and time constraints while still delivering valuable contributions to the field.
The systematic evaluation approach and reproducible methodology will benefit both academic and practitioner communities, while the identified research directions provide a clear roadmap for future doctoral work in information systems.
 
Appendices
Appendix A: Detailed Implementation Plan
•	Code structure and organization
•	Experiment tracking and data management
•	Evaluation script specifications
Appendix B: Literature Review Summary
•	Key papers in RAG evaluation
•	Gaps in current research
•	Methodological approaches
Appendix C: Preliminary Results Template
•	Data collection formats
•	Statistical analysis procedures
•	Visualization strategies
Estimated Paper Length: 8-12 pages (conference format)
Target Submission: 3-4 months from start
Expected Impact: Foundation for 3-4 year PhD research program

![image](https://github.com/user-attachments/assets/b2dd460c-d024-4ec0-97b2-54f7d0434b2a)
