# RFC: MrWorldwide – A Socratic AI Language Tutor via RAG and SLM Reasoning Distillation

## 1. Introduction
**Application Domain:** Natural Language Processing (NLP) in EdTech / AI-Assisted Language Learning.
**Motivation:** Most current AI language tutors act as simple dictionaries or answer keys, immediately correcting student mistakes rather than teaching them. This deprives students of the cognitive struggle necessary for true language acquisition. This project, titled **MrWorldwide**, aims to build an AI tutor that employs the **Socratic method**—guiding students to the correct answer through hints, targeted questions, and dynamic exercises. By separating factual curriculum (stored in a vector database) from pedagogical reasoning (handled by a distilled reasoning model), this system will provide high-quality, localized tutoring at a fraction of the computational cost of frontier models.

## 2. Problem Definition
**Problem:** To design a resource-efficient Small Language Model (SLM) capable of analyzing a student's linguistic error or curriculum stage, retrieving the relevant grammatical rule, and formulating a pedagogical hint or test rather than a direct translation.

* **Input:** A student's text input (potentially containing errors) and a retrieved context block containing the relevant grammar rule or lesson plan.
* **Output:** A two-part generation: an internal reasoning trace (hidden from the user) analyzing the linguistic context, followed by a user-facing hint, guiding question, or exercise.
* **Task Type:** Conditional Text Generation / Agentic RAG.
* **Constraints:** The model must be small enough to run on consumer hardware (≤ 8B parameters) while maintaining strict adherence to the pedagogical persona and preventing "catastrophic forgetting" of logical capabilities during fine-tuning.

## 3. State of the Art
Current approaches to AI tutoring generally rely on massive, closed-source LLMs or rigid, rule-based systems.

1.  **Reasoning Models:** The release of DeepSeek-R1 (DeepSeek-AI, 2025) demonstrated that Chain of Thought (CoT) reasoning can be effectively distilled from massive reinforcement learning models into smaller models using specific `<think>` tags.
2.  **Retrieval-Augmented Generation (RAG):** Lewis et al. (2020) established RAG as the standard for grounding LLM outputs. In MrWorldwide, RAG prevents the AI from hallucinating grammar rules and supplies the structured curriculum (lessons, tests).
3.  **Efficient Fine-Tuning:** QLoRA (Dettmers et al., 2023) allows for the parameter-efficient fine-tuning of large models on consumer GPUs by quantizing the base model and training a low-rank adapter.

## 4. Proposed Solution
We propose an agentic RAG pipeline driven by a fine-tuned reasoning SLM.

* **Model:** DeepSeek R1 Distilled 8B, fine-tuned using **QLoRA** via the **Unsloth** framework to heavily optimize VRAM usage and training speed.
* **Architecture & Workflow:** 1.  **Orchestration:** **LangGraph** will serve as the state machine managing the conversation flow and tracking the user's progress through the curriculum.
    2.  **Retrieval:** Depending on the LangGraph state, queries are routed to **ChromaDB** to fetch either a specific grammar rule (for error correction) or vocabulary lists (for generating tests).
    3.  **Generation:** The retrieved context and the student's input are passed to the 8B model. 
    4.  **Parsing:** The model generates a `<think>` block (where it formulates its teaching strategy) and a final response. LangGraph parses this output, logging the `<think>` block for evaluation and returning only the Socratic text/exercise to the student.

## 5. Dataset
Because standard language datasets focus on direct translation rather than teaching methodology, we will rely primarily on synthetic data distillation.

* **Dataset Name:** Custom Socratic Reasoning Corpus (Distilled) + SlimOrca-Dedup (Subset).
* **Source:** Synthetic data generated via API from a frontier model (e.g., GPT-4o), supplemented by logical reasoning subsets from Hugging Face (`Open-Orca/SlimOrca-Dedup`).
* **Description:** Approximately 1,500 to 2,000 highly curated JSONL (JSON Lines) examples to allow memory-efficient streaming during training. Each example will feature a simulated RAG context, a student query, and a target output that includes a detailed `<think>` reasoning trace followed by a pedagogical question.
* **Expected Preprocessing:** * Formatting all data into strict **ChatML** format (System, User, Assistant roles).
    * Filtering synthetic outputs to ensure the reasoning trace explicitly references the RAG context before formulating the final answer.
    * Shuffling in ~5% general reasoning tasks (from SlimOrca) to maintain the base model's structural logic.