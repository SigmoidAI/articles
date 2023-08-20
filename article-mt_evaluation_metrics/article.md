# Exploring Contemporary Machine Translation Evaluation Methods

### ***Author:*** Marius Sclearuc
 
__Abstract__:  The purpose of this article is to provide a comprehensive description of the underpinnings, effectiveness, and the pros and the cons of most frequently used Machine Translation evaluation methods.

***

Translation holds an essential role in fostering human connections across diverse languages and cultures. This, in turn, has propelled machine translation to the forefront of AI research, as it greatly facilitates meaningful communication accross our global, interconnected world.

Nevertheless, the intricancy of languages presents multiple translation challenges. Machine Translation (MT) models have to account for varying grammar, polysemous words, unique expressions, nuanced contexts, and the constant evolution of languages. And a quick peek at the pictures below shows us that even the most advanced MT models, like Google Translate & DeepL, sometimes fail to communicate the desired meaning. 

![First mistranslation example](/article-mt_evaluation_metrics/assets/abc.png)
*Google Translate. перстень is a (decorative) finger ring, while кольцо is a more general-purpose ring.*

![Second mistranslation example](/article-mt_evaluation_metrics/assets/def.png)
*DeepL. An appropriate translation woud be 'they had barely escaped'.*

To improve their efficiency, we have to first understand how do these models distinguish between a good & bad translation? To answer this question, I will explore various translation evaluation techniques currently available.

- - -

## __Human evaluation__
The first discussed metric typology is human evaluation. Just as the name suggests, it's done by human beings - usually by tens, maybe even hundreds or thousands of them. 

Its most valuable aspect is reliability,  as human evaluators can count on their intrinsic understanding of language nuances, cultural context, and fluency. This perspective contributes to achieving meaningful translations through evaluations while also offering a comprehensive assessment of the MT model's performance. 
On the downside, scaling it can prove quite time-consuming and costly. The more people on the job, the more it costs to maintain them - and the same can be said about time. Moreover, human judgments are subjective and sometimes even contradictory due to individual biases and interpretation differences, potentially leading to less reliable evaluations.

Without much further ado, let's discuss the 3 main human evaluation methods. To analyze them, I'll consider various translations of the above mistranslated sentence:

> "Scăpaseră ca prin urechile acului."

### **Rating**:
The most basic one of all the methods, it requires a person to evaluate a translation on a specific scale (usually 1 to 5) according to a specific metric/quality (like the adequacy or the fluency of the sentence). Better translation naturally get a higher score, while worse ones rank lower.

To exemplify, a sentence like:

> "They had barely escaped."

could be rated 5/5 on the fluency scale, while:

> "They had only narrowly managed to escape." 

would only get 3/5 due to its unnecessary verbosity.

A common pitfall of this approach is human subjectivity. One person's ratings can sometimes be inconsistent - even for them it can be difficult to assess the quality of the translation - or be influenced by other people's ratings. Moreover, different people have different approaches to scoring, as some might tend to the extremes, penalizing bad translations heavily, while others might be moderate in their scoring approach.

Research has shown that big pool of evaluators could help smoothen these issues[¹](#1-blatz-john--fitzgerald-erin--foster-george--g-simona--goutte-cyril--kulesza-alex--sanchis-alberto--ueng-nicola-2004-confidence-estimation-for-machine-translation-proceedings-of-coling-2004-10311512203551220401).

### **Relative ranking**:

In a nutshell, this method tries to atone for the shortcomings of the previous method by asking evaluators to choose the better trranslation from multiple pairs, instead of rating each of them. In this way, it can make precise distinctions between sentences, distinctions that would not be possible in the rating task. This becomes particularly crucial when evaluating numerous similar MT systems that generate nearly identical outputs for a wide range of source sentences.

To illustrate this concept, consider the following 2 phrases:

> "They had barely escaped." \
> "They eluded."

A MT model using ranking evaluation would suggest the first one first, as it describes the given situation more clearly.

The ranking task still suffers from a pitfalls, albeit of a different nature: incomparable sentences. Consider the following examples:

> "They narrowly dropped." \
> "They will escape."

One of them contains a mistranslation of the verb, while the other one gets the wrong verb tense. Neither of them is good enough, and yet one has to rank higher than the one, as ranking implies a choice needs to be made. And when translations get even larger, with different subsentences containing different types and numbers of errors, the task of the evaluators becomes ever harder.

### **Human-targeted Translation Error Rate (HTER)**:

Before diving into HTER, one should first understand the Translation Error Rate (TER) score. In simple terms[²](#2-matthew-snover-bonnie-dorr-rich-schwartz-linnea-micciulla-and-john-makhoul-2006-a-study-of-translation-edit-rate-with-targeted-human-annotation-in-proceedings-of-the-7th-conference-of-the-association-for-machine-translation-in-the-americas-technical-papers-pages-223–231-cambridge-massachusetts-usa-association-for-machine-translation-in-the-americas), the TER value of a given translation (also called hypothesis), computed against a set of reference sentences, is the minimum number of edits needed to change the hypothesis so that it exactly matches a reference, normalized by the average length of the references. Here, an edit represents the insertion, deletion, or substitution of a single word as well as shifts of word sequences.

For example, consider the following hypothesis:

> "Escaped they have."

against the same references:
> "They had barely evaded." \
> "They had only narrowly evaded."

In both cases, 'have' should be switched for 'had' (1 edit), 'escaped' should be deleted(1), and 'barely evaded'(2 edits)/'only narrowly evaded' (3 edits) inserted. In the first case, # of edits = 4, while in the second it's 5. Taking the smaller one, we get a TER score of 4/ 4.5 = 88.8%. 

This is quite an awful TER score, as obviously the lower it is, the better the translation. However, looking back, there's one obvious improvement we can make: continue using 'escaped' instead of 'evaded'. If we only shifted 'escaped' to the end of the sentence, we could've had a (slightly better) TER score of 66.6%. 

However, there's no "They had barely escaped." in the references. Here, HTER finally comes into play. 

Instead of 'fixing' the sentence agains given references, an evaluator can create their own target sentence (if it requires less editing). This method skillfully addresses the issues from the previous 2 tasks, eliminating the need for ratings assignment  and allowing post-editors to focus on ensuring semantic equivalence. This approach is especially beneficial for handling long sentences, as corrections can be made incrementally. Moreover, the post-editing process yields extra reference translations and edits highlighting areas of incorrect translation, which prove valuable for MT system development and error analysis[³](#3-michael-denkowski-and-alon-lavie-2010-choosing-the-right-evaluation-for-machine-translation-an-examination-of-annotator-and-automatic-metric-performance-on-human-judgment-tasks-in-proceedings-of-the-9th-conference-of-the-association-for-machine-translation-in-the-americas-research-papers-denver-colorado-usa-association-for-machine-translation-in-the-americas).

---

## __Computer evaluation__

Let's now turn our attention to the other metric typology - computer evaluation. While they do have a hard time breaking down translated text to assess aspects like fluency, accuracy, and meaning, the consistency and speed of computers does give them a slight advantage over human beings.

That's why computer evaluation plays a vital role nowadays. It offers a fast and efficient way to evaluate translations on a large scale, and it's a driving force behind AI and language tech progress. 

Without much further ado, here are a some of the most important MT computer evaluation metrics, encompassing traditional  ones as well as newer, promising approaches:

### **BiLingual Evaluation Understudy (BLEU)**:

BLEU, a semantic method developed by IBM in 2001, adheres to a straightforward principle: "The closer a machine translation is to a professional human translation, the better it is."[⁴](#4-kishore-papineni-salim-roukos-todd-ward-and-wei-jing-zhu-2002-bleu-a-method-for-automatic-evaluation-of-machine-translation-in-proceedings-of-the-40th-annual-meeting-of-the-association-for-computational-linguistics-pages-311–318-philadelphia-pennsylvania-usa-association-for-computational-linguistics) Its high corpus-level correlation with human judgment has established it as one of the most widely used MT metrics. Moreover, BLEU's applicability often extends to other tasks, like text generation, paraphrase creation, and text summarization.

In simple terms, BLEU first calculates a precision score based on the ratio of matching n-grams (contiguous word sequences of length n) between the translation and given reference sentences, rewarding translations that align well with the reference text. The scores are then combined using a modified geometric mean and finally multiplied by a brevity penalty to account for translation length. The end result represents a value between 0 and 1. The higher the score, the closer the sentence to the reference text, meaning the better the translation. 

While the authors of the paper recommended using weights of 0.25 for the 1-, 2-, 3-, and 4-grams, and 0 for the rest of them, the algorithm can be used with any given set of weights (summing up to 1, naturally) in order to maximize/minimize the influence the longer matching sub-sentences. 

Despite its wide usage, the BLEU score has a wide range of shortcomings. For example, it's mainly a corpus-based metric - using it on individual sentences leads to bad performance. It doesn't discern between function words (such as 'the') and content words ('cat'), meaning the absence of either gets penalized in the same way. And it doesn't take synonyms into account.

### **Metric for Evaluation of Translation with Explicit ORdering (METEOR)**[⁵](#5-satanjeev-banerjee-and-alon-lavie-2005-meteor-an-automatic-metric-for-mt-evaluation-with-improved-correlation-with-human-judgments-in-proceedings-of-the-acl-workshop-on-intrinsic-and-extrinsic-evaluation-measures-for-machine-translation-andor-summarization-pages-65–72-ann-arbor-michigan-association-for-computational-linguistics):

Here's where METEOR comes into play. Designed to atone for the imperfections of BLEU, it aims to achieve high correlation with human judgement both on sentence and corpus level. The original paper presented a 0.964 correlation level with human judgement at the corpus level, and a 0.403 at sentence level.

METEOR does so by first aligning the candidate and a reference translation. It links individual words between the sentences, making sure that candidate words are mapped to their reference sentence equivalent. If there's no possible equivalent, the word doesn't get mapped. The goal is to create an alignment with the least amoung of intersections (crosses) between mappings when there are multiple alignments with the same mappings count. An example is shown below.

Alingment 1             |  Alignment 2
:-------------------------:|:-------------------------:
![](/article-mt_evaluation_metrics/assets/alignment1.png)  |  ![](/article-mt_evaluation_metrics/assets/alignment2.png)

[Source](https://en.wikipedia.org/wiki/METEOR). Alignment 2 is better-suited, as there are fewer crosses between word mappings.

After the alignment is computed, the precision score P (ratio of mappings to total word count in the candidate sentence) and the recall score R (ratio of mappings to total word count in the reference translation) are calculated. An Fmean score is finally calculated as a weighted harmonic ratio between P and R:

![Fmean](/article-mt_evaluation_metrics/assets/fmean.png)

However, this score only accounts for single-word sentence similarity - and so METEOR computes a penalty for the given alignment. To do this, words are organized into the smallest number of chunks, with each chunk representing a set of adjacent unigrams in both the hypothesis and the reference sentence. A penalty p is computed as:

![penalty](/article-mt_evaluation_metrics/assets/penalty.png)

and the final METEOR score is computed as:

![meteor](/article-mt_evaluation_metrics/assets/meteor.png)

This method of evaluation a translation has several advantages over the previous one. For example, creating the initial alignment, synonymity is also taken into account, allowing for more accurate evaluation. The use of mapping and chunks over fixed-size n-grams accounts also helps to account for better grammaticality and fluency.

## METEOR-NEXT[⁶](#6-michael-denkowski-and-alon-lavie-2010-meteor-next-and-the-meteor-paraphrase-tables-improved-evaluation-support-for-five-target-languages-in-proceedings-of-the-joint-fifth-workshop-on-statistical-machine-translation-and-metricsmatr-pages-339–342-uppsala-sweden-association-for-computational-linguistics):

As the name suggests, this metric is an improvement attempt over METEOR. It still mainly relies on the same algorithm as METEOR, albeit with some important tweaks. 

The first one is the use of paraphrasing. Instead of matching words only based on form and synonymity, paraphrasing allows to watch whole subsentences, if they convey the same meaning. This is done by creating a corpus-specific paraphrase table using the so-called *pivot phrase* method[⁷](#7-colin-bannard-and-chris-callison-burch-2005-paraphrasing-with-bilingual-parallel-corpora-in-proceedings-of-the-43rd-annual-meeting-of-the-association-for-computational-linguistics-acl05-pages-597–604-ann-arbor-michigan-association-for-computational-linguistics).

Second, instead of using fixed parameters, the METEOR-NEXT introduces a bunch of new variables. Precision and recall are calculated as a weighted mean (with weights acting as one of the input variables). Fmean and the penalty similarly get a new look:

![meteornext](/article-mt_evaluation_metrics/assets/meteornext.png)

By fine-tuning these parameters, it has been shown[⁸](#8-michael-denkowski-and-alon-lavie-2010-extending-the-meteor-machine-translation-evaluation-metric-to-the-phrase-level-in-human-language-technologies-the-2010-annual-conference-of-the-north-american-chapter-of-the-association-for-computational-linguistics-pages-250–253-los-angeles-california-association-for-computational-linguistics) that it's possible to achieve higher correlation to post-editing human judgement (such as HTER).

## Crosslingual Optimized Metric for Evaluation of Translation (COMET)[⁹](#9-ricardo-rei-craig-stewart-ana-c-farinha-and-alon-lavie-2020-comet-a-neural-framework-for-mt-evaluation-in-proceedings-of-the-2020-conference-on-empirical-methods-in-natural-language-processing-emnlp-pages-2685–2702-online-association-for-computational-linguistics):

One of the most advanced tools available, COMET represents a highly adaptable PyTorch-based framework for training MT evaluation models. It has been shown to outperform traditional MT evaluation metrics on a variety of tasks. In the WMT 2022 Metrics Shared Task, it achieved the highest correlation with human judgments[¹⁰](#10-freitag-markus--rei-ricardo--mathur-nitika--lo-chi-kiu--stewart-craig--avramidis-eleftherios--kocmi-tom--foster-george--lavie-alon--martins-andré-2022-results-of-wmt22-metrics-shared-task-stop-using-bleu---neural-metrics-are-better-and-more-robust), making it one of the most promising models of the future.

While there are different scoring variations of the framework (such as ranking, HTER, and MQM), we will focus on the most basic one of them: rating. 

At its core, it utilizes a pre-trained, cross-lingual model (such as BERT or XML-RoBERTa). It uses the encoder to tokenize the candidate translation, the original source, but also a reference sentence. After passing the result through a pooling layer, the method combines (by concatenation) features like element-wise products and differences in order to create a single vector input for a feed-forward regressor. A mean squared error is finally computed, representing the 'distance' between the predicted score and quality assessment.

The model is trained on datasets containing human ratings of the sentences to minimize the error. A visual representation is shown below:

![architecture](/article-mt_evaluation_metrics/assets/architecture.png)

The COMET model architecture offers several advantages compared to traditional MT evaluation metrics. Firstly, it considers both the source sentence and the reference translation, hence inferring valuable contextual information. Secondly, COMET (being a neural model) can grasp complex data patterns and enhance prediction accuracy beyond traditional metrics. Thirdly, its multilingual nature enables the evaluation of MT systems across various languages.

---
## Conclusion:

All in all, this exploration of current machine learning evaluation metrics highlights the progress made in assessing the efficacy of models. While the metrics discussed above unveil substantial advancements in measuring translation quality, it is essential to acknowledge that this is not an exhaustive list. There are many other traditional models - such as NIST, ROUGE, and TER - as well as promising contenders - like BERTScore and YiSi. 

The evolution of evaluation methodologies continues to move the field of Machine Translation forward, enriching our understanding of how languages work and refining the quality assessment of machine learning models.

## References:
##### 1. Blatz, John & Fitzgerald, Erin & Foster, George & G, Simona & Goutte, Cyril & Kulesza, Alex & Sanchis, Alberto & Ueng, Nicola. (2004). Confidence Estimation for Machine Translation. Proceedings of COLING 2004. 10.3115/1220355.1220401.

##### 2. Matthew Snover, Bonnie Dorr, Rich Schwartz, Linnea Micciulla, and John Makhoul. 2006. [A Study of Translation Edit Rate with Targeted Human Annotation](https://aclanthology.org/2006.amta-papers.25). In *Proceedings of the 7th Conference of the Association for Machine Translation in the Americas: Technical Papers*, pages 223–231, Cambridge, Massachusetts, USA. Association for Machine Translation in the Americas.

##### 3. Michael Denkowski and Alon Lavie. 2010. [Choosing the Right Evaluation for Machine Translation: an Examination of Annotator and Automatic Metric Performance on Human Judgment Tasks](https://aclanthology.org/2010.amta-papers.20). In *Proceedings of the 9th Conference of the Association for Machine Translation in the Americas: Research Papers*, Denver, Colorado, USA. Association for Machine Translation in the Americas.

##### 4. Kishore Papineni, Salim Roukos, Todd Ward, and Wei-Jing Zhu. 2002. [Bleu: a Method for Automatic Evaluation of Machine Translation](https://aclanthology.org/P02-1040). In *Proceedings of the 40th Annual Meeting of the Association for Computational Linguistics*, pages 311–318, Philadelphia, Pennsylvania, USA. Association for Computational Linguistics.

##### 5. Satanjeev Banerjee and Alon Lavie. 2005. [METEOR: An Automatic Metric for MT Evaluation with Improved Correlation with Human Judgments](https://aclanthology.org/W05-0909). In *Proceedings of the ACL Workshop on Intrinsic and Extrinsic Evaluation Measures for Machine Translation and/or Summarization*, pages 65–72, Ann Arbor, Michigan. Association for Computational Linguistics.

##### 6. Michael Denkowski and Alon Lavie. 2010. [METEOR-NEXT and the METEOR Paraphrase Tables: Improved Evaluation Support for Five Target Languages](https://aclanthology.org/W10-1751). In *Proceedings of the Joint Fifth Workshop on Statistical Machine Translation and MetricsMATR*, pages 339–342, Uppsala, Sweden. Association for Computational Linguistics.

##### 7. Colin Bannard and Chris Callison-Burch. 2005. [Paraphrasing with Bilingual Parallel Corpora](https://aclanthology.org/P05-1074). In *Proceedings of the 43rd Annual Meeting of the Association for Computational Linguistics (ACL'05)*, pages 597–604, Ann Arbor, Michigan. Association for Computational Linguistics.

##### 8. Michael Denkowski and Alon Lavie. 2010. [Extending the METEOR Machine Translation Evaluation Metric to the Phrase Level](https://aclanthology.org/N10-1031). In *Human Language Technologies: The 2010 Annual Conference of the North American Chapter of the Association for Computational Linguistics*, pages 250–253, Los Angeles, California. Association for Computational Linguistics.

##### 9. Ricardo Rei, Craig Stewart, Ana C Farinha, and Alon Lavie. 2020. [COMET: A Neural Framework for MT Evaluation](https://aclanthology.org/2020.emnlp-main.213). In *Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP)*, pages 2685–2702, Online. Association for Computational Linguistics.

##### 10. Freitag, Markus & Rei, Ricardo & Mathur, Nitika & Lo, Chi-Kiu & Stewart, Craig & Avramidis, Eleftherios & Kocmi, Tom & Foster, George & Lavie, Alon & Martins, André. (2022). Results of WMT22 Metrics Shared Task: Stop Using BLEU - Neural Metrics Are Better and More Robust.