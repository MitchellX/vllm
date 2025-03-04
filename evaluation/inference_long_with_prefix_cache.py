# -*- coding: utf-8 -*-
'''
 Copyright (c) ByteDance Inc.
 Authors:
  - Tongping Liu (tongping.liu@bytedance.com)
'''

from transformers import pipeline, set_seed
from vllm import LLM, SamplingParams

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

# Sample prompts.
prompts = [
    " I want you to act as a storyteller. You will come up with entertaining stories that are engaging, imaginative and captivating for the audience. It can be fairy tales, educational stories or any other type of stories which has the potential to capture people’s attention and imagination. Depending on the target audience, you may choose specific themes or topics for your storytelling session e.g., if it’s children then you can talk about animals; If it’s adults then history-based tales might engage them better etc. My first request is “I need an interesting story on perseverance.",
    "Please write a paper of no less than 4,000 words based on the following abstract. You can firstly write the background and then do the literature review. In the domain of multimedia and multimodal processing, the efficient handling of diverse data streams-such as images, video, and sensor data-is paramount. Model compression and multitask learning (MTL) are crucial in this field, offering the potential to address the resource-intensive demands of processing and interpreting multiple forms of media simultaneously. However, effectively compressing a multitask model presents significant challenges due to the complexities of balancing sparsity allocation and accuracy performance across multiple tasks. ",
    " I want you to act as an advertiser. You will create a campaign to promote a product or service of your choice. You will choose a target audience, develop key messages and slogans, select the media channels for promotion, and decide on any additional activities needed to reach your goals. My first suggestion request is “I need help creating an advertising campaign for a new type of energy drink targeting young adults aged 18-30.”",
    "I want you to act as a travel guide. I will write you my location and you will suggest a place to visit near my location. In some cases, I will also give you the type of places I will visit. You will also suggest me places of similar type that are close to my first location. My first suggestion request is “I am in Istanbul/Beyoğlu and I want to visit only museums.”",
]


prompts = prompts * 4
# prompts = prompts * 8
# prompts = prompts * 10
# prompts = prompts * 15

generating_prompts = [prefix + prompt for prompt in prompts]


set_seed(32)

import os
os.environ['VLLM_ATTENTION_BACKEND'] = 'XFORMERS'
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com' 

# Create a sampling params object.
#sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=8192, ignore_eos=True)
#sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=8192, ignore_eos=True)
#sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=2048)
#sampling_params = SamplingParams(temperature=0, top_p=1, top_k=1,max_tokens=2048)
sampling_params = SamplingParams(temperature=0, top_p=1, top_k=1,max_tokens=512)
# sampling_params = SamplingParams(temperature=0, top_p=1, top_k=1, max_tokens=2048, )

# Create an LLM.
#llm = LLM(model="facebook/opt-6.7b")
#llm = LLM(model="facebook/opt-6.7b", use_dattn=True)
#llm = LLM(model="facebook/opt-6.7b", use_dattn=True, enforce_eager=True)
#llm = LLM(model="Qwen/Qwen-7B", use_dattn=True, trust_remote_code=True, enforce_eager=True, preemption_mode="swap")
# llm = LLM(model="facebook/opt-2.7B", use_dattn=True, enforce_eager=True, preemption_mode="swap", enable_prefix_caching=False)
prefix_cached_llm = LLM(model="facebook/opt-6.7b", use_dattn=True,  enforce_eager=True, preemption_mode="swap",enable_prefix_caching=True) # [RECOMPUTE, SWAP]
# llm = LLM(model="meta-llama/Llama-2-7b-chat-hf", use_dattn=True,  enforce_eager=True, preemption_mode="swap")
#llm = LLM(model="facebook/opt-6.7b", enforce_eager=True)
#llm = LLM(model="facebook/opt-6.7b", enforce_eager=True, preemption_mode="swap")
#llm = LLM(model="facebook/opt-2.7b", enforce_eager=True, preemption_mode="swap")
#llm = LLM(model="facebook/opt-6.7b", enforce_eager=True, preemption_mode="swap")
#llm = LLM(model="facebook/opt-6.7b", enforce_eager=True)
#llm = LLM(model="facebook/opt-1.3b", enforce_eager=True)
#llm = LLM(model="facebook/opt-125m", use_dattn=True, enforce_eager=True)
#llm = LLM(model="facebook/opt-125m", enforce_eager=True)
# Generate texts from the prompts. The output is a list of RequestOutput objects
# that contain the prompt, generated text, and other information.
# outputs = llm.generate(prompts, sampling_params)

# Warmup so that the shared prompt's KV cache is computed.
prefix_cached_llm.generate(generating_prompts[0], sampling_params)

# Generate with prefix caching.
outputs = prefix_cached_llm.generate(generating_prompts, sampling_params)

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
