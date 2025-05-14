# -*- coding: utf-8 -*-
'''
 Copyright (c) ByteDance Inc.
 Authors:
  - Tongping Liu (tongping.liu@bytedance.com)
'''

from transformers import pipeline, set_seed
from vllm import LLM, SamplingParams
import os
import time
os.environ['VLLM_ATTENTION_BACKEND'] = 'XFORMERS'
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com' 
set_seed(32)

# Common prefix.
prefix = (
    "In this document, we establish a foundational basis for processing "
    "structured and unstructured prompts related to storytelling, academic "
    "research, advertising, and travel recommendations. This framework allows "
    "for enhanced contextual understanding, ensuring high-quality responses "
    "that adhere to the intent and style specified in the prompt. "

    "### 1. Storytelling Framework "
    "Storytelling is an ancient art that conveys lessons, emotions, and "
    "experiences through structured narratives. Effective stories typically "
    "follow the classic three-act structure: "
    "- Act 1: Introduction – Introducing the setting, main characters, and initial conflict. "
    "- Act 2: Development – The protagonist faces increasing challenges that test their perseverance. "
    "- Act 3: Resolution – The protagonist overcomes obstacles, leading to a satisfying conclusion. "

    "When generating a story, it is important to consider: "
    "- Target Audience: Tailoring language and themes for children, young adults, or mature audiences. "
    "- Genre & Theme: Fantasy, science fiction, historical fiction, or educational storytelling. "
    "- Emotional Engagement: Developing relatable characters and conflicts. "
    "- Moral Lessons: Providing insight or wisdom. "

    "Example Story on Perseverance: "
    "In a distant land where the sun never set, a young apprentice named Alric "
    "sought to master the lost art of forging indestructible weapons. Mocked by "
    "his peers and burdened with endless failures, he ventured into the mountains "
    "in search of the fabled 'Fire of Eternity.' Through sheer determination and "
    "relentless practice, he overcame countless hardships, proving that perseverance "
    "transforms dreams into reality. "

    "### 2. Academic Paper Writing Framework "
    "Academic writing requires rigorous structure, logical flow, and strong supporting "
    "evidence. A well-structured research paper consists of: "
    "- Abstract: A concise summary highlighting key contributions. "
    "- Introduction: Providing background information and stating the research problem. "
    "- Literature Review: Discussing prior research and identifying gaps. "
    "- Methodology: Outlining the approach taken to address research questions. "
    "- Results & Discussion: Presenting findings and their implications. "
    "- Conclusion & Future Work: Summarizing insights and suggesting further research directions. "

    "Example Paper Abstract on Model Compression in Multimedia Processing: "
    "The efficient handling of diverse multimodal data streams—including images, "
    "video, and sensor data—is critical in multimedia processing. Model compression "
    "techniques, such as structured pruning and knowledge distillation, play a "
    "crucial role in reducing computational costs without compromising accuracy. "
    "However, balancing sparsity allocation and task-specific accuracy remains a "
    "challenge. This paper explores novel adaptive pruning strategies for multitask "
    "learning (MTL) models, ensuring optimal trade-offs between performance and efficiency. "

    "When writing a long-form research paper (4,000+ words), it is crucial to: "
    "- Use formal and precise language. "
    "- Cite relevant studies and provide strong empirical evidence. "
    "- Structure sections logically for coherence and readability. "

    "### 3. Advertising Campaign Development "
    "Advertising involves understanding the target market, crafting compelling messages, "
    "and strategically placing content across multiple media channels. "

    "Key Components of an Effective Advertising Campaign: "
    "- Target Audience: Understanding demographics, interests, and behaviors. "
    "- Core Messaging: Developing persuasive and memorable slogans. "
    "- Marketing Channels: Utilizing digital, print, and social media platforms. "
    "- Brand Differentiation: Highlighting unique selling points of the product. "

    "Example Campaign: "
    "Introducing X-Boost, the next-generation energy drink packed with organic caffeine "
    "and essential vitamins. Designed for young adults aged 18-30, X-Boost enhances focus, "
    "endurance, and mental clarity. With a bold, refreshing taste and zero artificial "
    "additives, X-Boost is the smart choice for go-getters. Tagline: 'Power Your Passion – Anytime, Anywhere!' "

    "### 4. Travel Guide and Recommendations "
    "As a travel guide, recommendations should be tailored based on: "
    "- Current Location: Providing attractions within a reasonable distance. "
    "- User Preferences: Museums, historical sites, outdoor activities, nightlife, etc. "
    "- Cultural & Seasonal Considerations: Suggesting experiences unique to the region. "

    "Example Travel Recommendation in Istanbul/Beyoğlu: "
    "For a cultural journey through Istanbul’s rich history, visit the Pera Museum, home to "
    "an extensive collection of Orientalist paintings and Anatolian artifacts. For contemporary "
    "art lovers, Istanbul Modern offers breathtaking exhibitions of Turkish and international artists. "
    "If you’re fascinated by history, explore the Galata Mevlevi Museum, where you can experience the "
    "mysticism of the Whirling Dervishes. Each of these museums provides a unique glimpse into Turkey’s "
    "vibrant artistic and historical landscape. "

    "### Conclusion "
    "By structuring responses effectively for storytelling, academic writing, advertising, and travel "
    "recommendations, we ensure comprehensive, engaging, and contextually relevant content that meets "
    "user expectations."
)


prompts = [
    " I want you to act as a storyteller. You will come up with entertaining stories that are engaging, imaginative and captivating for the audience. It can be fairy tales, educational stories or any other type of stories which has the potential to capture people’s attention and imagination. Depending on the target audience, you may choose specific themes or topics for your storytelling session e.g., if it’s children then you can talk about animals; If it’s adults then history-based tales might engage them better etc. My first request is “I need an interesting story on perseverance.",
]

prompts = [prefix + prompt for prompt in prompts]

warmup_params = SamplingParams(temperature=0, top_p=1, top_k=1, max_tokens=1)   # 只 Prefill
sampling_params = SamplingParams(temperature=0, top_p=1, top_k=1, max_tokens=16)


llm = LLM(model="facebook/opt-6.7b",
                        use_dattn=True,
                        enforce_eager=True,
                        preemption_mode="swap",         # [RECOMPUTE, SWAP]
                        enable_prefix_caching=False)

# ---------- 4. WARM-UP phase: prefill ----------
t0 = time.time()
llm.generate(prompts, warmup_params)
t_prefill = time.time() - t0
print(f"[warm-up] prefill+1tok latency: {t_prefill:.3f}s")

# ---------- 5. OFFLOAD KV  ----------
sched = llm.llm_engine.scheduler[0]    # single GPU worker
sched.offload_all_warmed()             # GPU → CPU (one memcpy)
print("[warm-up] offload done")

# 5.5 Add one empty engine step right after off‑loading:
llm.llm_engine.step()               # <-- let Worker execute it
print("[warm‑up] extra step done")


# ---------- 6. (OPTIONAL) inspect offload info ----------
rid_warmed = sched.warmed[0].request_id       # ??? should we put this in the swapped list?
print("offload map:", sched._offloaded_kv_info)
print("warmed rid:", rid_warmed)

# ---------- 7. LOAD KV before real inference ----------
# After offload
if sched._offloaded_kv_info:
    rid = next(iter(sched._offloaded_kv_info))
    if sched.has_cpu_cache(rid):
        sched.load_kv_cache(rid)
        print("[load] KV loaded back to GPU")
else:
    print("No CPU cache to load – offload may have failed.")

llm.llm_engine.step()               # <-- let Worker execute it
print("[load] extra step done")

# ---------- 8. REAL inference ----------
t1 = time.time()
outputs = llm.generate(prompts, sampling_params)
t_decode = time.time() - t1
print(f"[run] decode latency: {t_decode:.3f}s")


print("Results with `enable_prefix_caching`")
# Print the outputs.
total = 0
for index, output in enumerate(outputs):
    prompt = output.prompt
    generated_text = output.outputs[0].text
    total += len(generated_text)
    print(f"prompt-{index}, generated text:{len(generated_text)}")
    #print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
    #print(f"Prompt: {prompt!r}\n, Generated text: {generated_text!r}\n\n")

print(f"generated text with the total length-{total}")
