import json
import os
import re
from collections import Counter
import math
from Levenshtein import ratio
import statistics
from collections import defaultdict
from nltk.corpus import indian
from nltk.tag import tnt

# TRAINING THE HINDI DATA
train_data = indian.tagged_sents('hindi.pos')
tnt_pos_tagger = tnt.TnT()
tnt_pos_tagger.train(train_data)


num_of_ngrams = int(
    input("POKE=> Enter the number of words in a keyphrase (n-gram) : "))
# pre-processing the text.


def pre_process(text, language):

    def split_hindi_sentences(text):
        # Defining a regex pattern for hindi sentence delimiters
        # The pattern includes common delimiters like '|', '!', '?', and newline ('\n')
        sentence_delimiters = re.compile(r'[\।\.!?\n\·]')

        # Split the text into sentences based on the defined delimiters
        sentences = re.split(sentence_delimiters, text)

        # Remove empty sentences and strip whitespace from the remaining sentences
        sentences = [sentence.strip()
                     for sentence in sentences if sentence.strip()]

        return sentences

    def is_number(s):
        try:
            float(s.replace(',', ''))
            return True
        except ValueError:
            return False

    def is_unparsable(s):
        if not is_number(s):
            special_character_list = ['!', '"', '#', '$', '%', '&', "'",
                                      '(', ')', '*', '+', ',', '-', '.', '/', ':', ';', '<', '=', '>', '?', '@', '[', '\\', ']', '^', '_', '`', '{', '|', '}', '~']
            special_character_count = 0

            for char in s:
                if char in special_character_list:
                    special_character_count += 1

            for char in s:
                if char.isdigit():
                    return True

            if special_character_count >= 1:
                return True

        return False

    def hindi_token_tagger(token, language):
        if is_number(token):
            return ('d', False)  # (tag = 'd', is_stopword = False)
        elif token.isupper():
            return ('a', False)
        elif is_unparsable(token):
            return ('u', False)
        else:
            if token in language:
                return ('p', True)
            return ('p', False)

    sentences = split_hindi_sentences(text)
    all_chunks = list()
    annotated_sentences = list()

    for sentence in sentences:
        append_to_annotated_sentences = (sentence, list())
        chunks = re.split(r'[\.,\(\)!?]', sentence)
        all_chunks.extend(chunks)
        for chunk in chunks:
            # doc = nlp(chunk)
            # chunk = " ".join([token.lemma_ for token in doc])
            tokens = chunk.split()
            for token in tokens:
                tag, is_stopword = hindi_token_tagger(token, language)
                append_to_annotated_sentences[1].append(
                    (token, tag, is_stopword))
        annotated_sentences.append(append_to_annotated_sentences)

    # returns tuple( [ tuple(sentence , (token,tag,is_stopword)) ] , [ all_chunks ] )
    return (annotated_sentences, all_chunks)


class Term:
    def __init__(self):
        self.TF = 0
        self.offsets_sentences = []
        self.TF_a = 0
        self.TF_U = 0
        self.stopword = False
        self.TCase = 0
        self.TPos = 0
        self.TFNorm = 0
        self.TRel = 0
        self.TSent = 0
        self.score = 0


class TermProcessor:
    def __init__(self):
        self.terms = None
        self.cooccur = None

    def compute_term_statistics(self, sentences, chunks, w):
        self.terms = defaultdict(Term)
        self.cooccur = defaultdict(lambda: defaultdict(int))
        # Loop through each sentence
        for sentence_index in range(len(sentences)):
            # sentence = sentences[sentence_index][0]
            for chunk in chunks:

                # Split the chunk into tokens and normalize to lowercase
                tokens = chunk.split()
                # print(tokens)
                # print(sentence)
                # Loop through tokens
                for i in range(len(tokens)):

                    term = tokens[i]

                    self.terms[term].TF += 1
                    self.terms[term].offsets_sentences.append(sentence_index)
                    # checking for acronym
                    if tokens[i].isupper():
                        self.terms[term].TF_a += 1
                    # checking if the first letter is uppercase
                    if tokens[i][0].isupper() and i > 0:
                        self.terms[term].TF_U += 1
                    # fill the cooccur matrix
                    for j in range(1, w + 1):
                        if i - j >= 0:
                            prev_term = tokens[i - j]

                            self.cooccur[term][prev_term] += 1
                            self.cooccur[prev_term][term] += 1

    def get_T_rel(self, annotated_sentences, w, term, termm):

        left_terms = list()
        right_terms = list()
        for sentence in annotated_sentences:
            if term in sentence[0]:
                split_sentence = sentence[0].split()
                for i in range(len(split_sentence)):
                    if split_sentence[i] == term:
                        for j in range(w):
                            try:
                                if i-j-1 >= 0:

                                    for k in range(len(sentence[1])):
                                        if sentence[1][k][0] == term and (sentence[1][k][1] == 'p' or sentence[1][k][1] == 'a'):
                                            left_terms.append(
                                                split_sentence[i-(j+1)])

                                if i+j+1 < len(split_sentence):
                                    for k in range(len(sentence[1])):
                                        if sentence[1][k][0] == term and (sentence[1][k][1] == 'p' or sentence[1][k][1] == 'a'):
                                            right_terms.append(
                                                split_sentence[i+(j+1)])
                            except:
                                print("trelerror")

        dl = 0
        dr = 0
        if len(left_terms):
            dl = len(set(left_terms)) / len(left_terms)
        if len(right_terms):
            dr = len(set(right_terms)) / len(right_terms)

        max_frequency = 0
        for termmm in self.terms.values():
            if termmm.TF > max_frequency:
                max_frequency = termmm.TF

        return 1 + (dl + dr)*(termm.TF / max_frequency)

    def compute_features(self, sentences, w):
        validTFs = [term.TF for term in self.terms.values()
                    if not term.stopword]
        # print(validTFs)
        # Calculate average and standard deviation of non-stopword TF values

        avgTF = statistics.mean(validTFs)
        stdTF = statistics.stdev(validTFs)

        for termm, term in self.terms.items():
            # Calculate features for each term
            # TCase refers to the value related to word which have first letter as capital
            term.TCase = max(term.TF_a, term.TF_U) / (1 + math.log(term.TF))
            # TPos refers to the position of the word w.r.t sentence
            term.TPos = math.log(
                math.log(3 + statistics.median(term.offsets_sentences)))
            # TFNORM is the normalization of term frequency
            term.TFNorm = term.TF / (avgTF + stdTF)

            maxTF = max([term.TF for term in self.terms.values()])
            # print(termm);

            term.TRel = self.get_T_rel(sentences, w, termm, term)

            term.TSent = len(term.offsets_sentences) / len(sentences)

    def calculate_term_score(self):

        for term in self.terms.values():
            term.score = (term.TPos * term.TRel) / (term.TCase +
                                                    ((term.TFNorm + term.TSent) / term.TRel))

    def process(self, sentences, chunks, w):
        self.compute_term_statistics(sentences, chunks, w)
        # for term, stats in self.terms.items():
        #     print(f"Term: {term}")
        #     print(f"TF: {stats.TF}")
        #     print(f"Offsets in Sentences: {stats.offsets_sentences}")
        #     print(f"TF_a: {stats.TF_a}")
        #     print(f"TF_U: {stats.TF_U}")
        #
        # for term1 in self.cooccur:
        #     for term2, count in self.cooccur[term1].items():
        #         print(f"Co-occurrence of '{term1}' and '{term2}': {count}")

        self.compute_features(sentences, w)
        self.calculate_term_score()
        return self.terms


# Example usage:
# sentences = [" Mahatma Gandhi was a great philosopher", "He went to South Africa for education"]
# chunks = 3
# w = 1


def generate_ngrams(sentences, chunks, n, stop_words):
    ngrams = []
    for sentance in sentences:
        for chunk in chunks:
            # tokenzinging the sentances using space as a delimiter
            # tokens = sentance[0].split()
            tokens = chunk.split()
            for x in range(len(tokens)-n+1):
                if ((tokens[x] not in stop_words) & (tokens[x+n-1] not in stop_words)):
                    tagged_words = (tnt_pos_tagger.tag(tokens[x:x+n]))
                    flag = 0
                    for tword in tagged_words:
                        if ((tword[1] == 'NN' or tword[1] == 'NNP' or tword[1] == 'NNC') & (flag == 0)) & (tagged_words[-1][1] == 'NN' or tagged_words[0][1] == 'NNP'):
                            ngrams.append(" ".join(tokens[x:x+n]))
                            flag = 1

    return ngrams


def generate_candidatekeywords(ngrams):
    # generating candidate keyword with attributes as key frequency and score
    tokens = ngrams
    counts = Counter(ngrams)
    tokens = list(set(tokens))
    candidate_keywords = [
        {"key": token, "KF": counts[token], 'Skr': 0}for token in tokens]
    return candidate_keywords


def scoring(candidate_keywords, stop_words, terms):
    # scoring
    for candidate in candidate_keywords:
        tokens = candidate['key'].split()
        prod_s = 1
        sum_s = 1
        for i in range(len(tokens)):
            if (tokens[i] not in stop_words):
                prod_s *= terms[tokens[i]].score
                sum_s += terms[tokens[i]].score
            else:
                probbefore = candidate['KF']/terms[tokens[i-1]].TF
                probafter = candidate['KF']/terms[tokens[i]].TF
                Bigram_probability = probbefore * probafter
                prod_s *= 1+(1-Bigram_probability)
                sum_s += (1-Bigram_probability)
        candidate['Skr'] = prod_s/candidate['KF']*(sum_s+1)


def distance_similarity(term1, term2):
    # Use Levenshtein similarity (ratio) for distance similarity
    return ratio(term1, term2)


def data_deduplication(candidate_keywords, threshold):
    if candidate_keywords:
        keywords = [candidate_keywords[0]]
    else:
        keywords = []
    # print(keywords)
    for candidate in candidate_keywords[1:]:
        skip = False

        for key in keywords:
            # print(distance_similarity(candidate['key'], key['key']))
            if distance_similarity(candidate['key'], key['key']) > threshold:
                skip = True
                break

        if not skip:
            keywords.append(candidate)

    return keywords


# these are hindi stopwords.
language = list(set([
    "पर", "इन", "वह", "यिह", "वुह", "जिन्हें", "जिन्हों", "तिन्हें", "तिन्हों", "किन्हों",
    "किन्हें", "इत्यादि", "द्वारा", "इन्हें", "इन्हों", "उन्हों", "बिलकुल", "निहायत",
    "ऱ्वासा", "इन्हीं", "उन्हीं", "उन्हें", "इसमें", "जितना", "दुसरा", "कितना",
    "दबारा", "साबुत", "वग़ैरह", "दूसरे", "कौनसा", "लेकिन", "होता", "करने", "किया",
    "लिये", "अपने", "नहीं", "दिया", "इसका", "करना", "वाले", "सकते", "इसके", "सबसे",
    "होने", "करते", "बहुत", "वर्ग", "करें", "होती", "अपनी", "उनके", "कहते", "होते",
    "करता", "उनकी", "इसकी", "सकता", "रखें", "अपना", "उसके", "जिसे", "तिसे", "किसे",
    "किसी", "काफ़ी", "पहले", "नीचे", "बाला", "यहाँ", "जैसा", "जैसे", "मानो", "अंदर",
    "भीतर", "पूरा", "सारा", "होना", "उनको", "वहाँ", "वहीं", "जहाँ", "जीधर", "उनका",
    "इनका", "के", "हैं", "गया", "बनी", "एवं", "हुआ", "साथ", "बाद", "लिए", "कुछ",
    "कहा", "यदि", "हुई", "इसे", "हुए", "अभी", "सभी", "कुल", "रहा", "रहे", "इसी",
    "उसे", "जिस", "जिन", "तिस", "तिन", "कौन", "किस", "कोई", "ऐसे", "तरह", "किर",
    "साभ", "संग", "यही", "बही", "उसी", "फिर", "मगर", "का", "एक", "यह", "से", "को",
    "इस", "कि", "जो", "कर", "मे", "ने", "तो", "ही", "या", "हो", "था", "तक", "आप",
    "ये", "थे", "दो", "वे", "थी", "जा", "ना", "उस", "एस", "पे", "उन", "सो", "भी",
    "और", "घर", "तब", "जब", "अत", "व", "न", "है", "की", "में", "अंदर", "अत", "अदि",
    "अप", "अपना", "अपनि", "अपनी", "अपने", "अभि", "अभी", "आदि", "आप", "इंहिं", "इंहें",
    "इंहों", "इतयादि", "इत्यादि", "इन", "इनका", "इन्हीं", "इन्हें", "इन्हों", "इस", "इसका", "इसकि",
    "इसकी", "इसके", "इसमें", "इसि", "इसी", "इसे", "उंहिं", "उंहें", "उंहों", "उन", "उनका",
    "उनकि", "उनकी", "उनके", "उनको", "उन्हीं", "उन्हें", "उन्हों", "उस", "उसके", "उसि",
    "उसी", "उसे", "एक", "एवं", "एस", "एसे", "ऐसे", "ओर", "और", "कइ", "कई", "कर",
    "करता", "करते", "करना", "करने", "करें", "कहते", "कहा", "का", "काफि", "काफ़ी", "कि",
    "किंहें", "किंहों", "कितना", "किन्हें", "किन्हों", "किया", "किर", "किस", "किसि", "किसी", "किसे",
    "की", "कुछ", "कुल", "के", "को", "कोइ", "कोई", "कोन", "कोनसा", "कौन", "कौनसा", "गया",
    "घर", "जब", "जहाँ", "जहां", "जा", "जिंहें", "जिंहों", "जितना", "जिधर", "जिन", "जिन्हें", "जिन्हों",
    "जिस", "जिसे", "जीधर", "जेसा", "जेसे", "जैसा", "जैसे", "जो", "तक", "तब", "तरह", "तिंहें",
    "तिंहों", "तिन", "तिन्हें", "तिन्हों", "तिस", "तिसे", "तो", "था", "थि", "थी", "थे", "दबारा", "दवारा",
    "दिया", "दुसरा", "दुसरे", "दूसरे", "दो", "द्वारा", "न", "नहिं", "नहीं", "ना", "निचे", "निहायत", "नीचे",
    "ने", "पर", "पहले", "पुरा", "पूरा", "पे", "फिर", "बनि", "बनी", "बहि", "बही", "बहुत", "बाद",
    "बाला", "बिलकुल", "भि", "भितर", "भी", "भीतर", "मगर", "मानो", "मे", "में", "यदि", "यह", "वाला",
    "यहाँ", "यहां", "यहि", "यही", "या", "यिह", "ये", "रखें", "रवासा", "रहा", "रहे", "ऱ्वासा", "लिए", "लिये",
    "लेकिन", "व", "वगेरह", "वरग", "वर्ग", "वह", "वहाँ", "वहां", "वहिं", "वहीं", "वाले", "वुह", "वे", "वग़ैरह",
    "संग", "सकता", "सकते", "सबसे", "सभि", "सभी", "साथ", "साबुत", "साभ", "सारा", "से", "सो", "हि", "बारे",
    "ही", "हुअ", "हुआ", "हुइ", "हुई", "हुए", "हे", "हें", "है", "हैं", "हो", "होता", "होति", "होती", "होते", "होना", "होने"]))


"""
# Uncomment for testing

# text = "एकलव्य फाउण्डेशन भारत के मध्य प्रदेश राज्य में कार्यरत एक अशासकीय संस्था  है। यह बच्चों की शिक्षा के क्षेत्र में आधारभूत कार्य कर रही है। यह सन् १९८२ में एक अखिल भारतीय संस्था के रूप में पंजीकृत हुई थी। प्राथमिक शिक्षा के क्षेत्र में वैज्ञानिक पद्धति एंव बालशिक्षा में तकनीकी विकास पर एकलव्य फाउण्डेशन द्वारा क्रियान्वन कराया जा रहा है। भोपाल स्थित संस्था के कार्यालय द्वारा विभिन्न शैक्षणिक कार्यक्रम चलाये जा रहे हैं। महाभारत के पात्र एकलव्य के जीवन चरित्र से प्रभावित इस संस्था का दर्शन शिक्षा के उन्न्यन में समाज में महत्वपूर्ण भूमिका रखता है।"

text = input('POKE=> Enter Hindi Text to Extract Keywords: ')

sentences, chunks = pre_process(text, language)


# print(sentences)
term_processor = TermProcessor()
result_terms = term_processor.process(sentences, chunks, w=2)

ngrams = generate_ngrams(sentences, chunks, n=3, stop_words=language)
candidate_keywords = generate_candidatekeywords(ngrams)
scoring(candidate_keywords, language, result_terms)


# Print terms with their features
def print_final():
    for term, features in result_terms.items():
        print(f"Term: {term}")
        print(f"TF: {features.TF}")
        print(f"TCase: {features.TCase}")
        print(f"TPos: {features.TPos}")
        print(f"TFNorm: {features.TFNorm}")
        print(f"TRel: {features.TRel}")
        print(f"TSent: {features.TSent}")
        print(f"Score:{features.score}")


# print(ngrams)
candidate_keywords = sorted(
    candidate_keywords, key=lambda x: x['Skr'], reverse=False)
# print(candidate_keywords)

candidate_keywords_length = len(candidate_keywords)
for i in range(20):
    if i < candidate_keywords_length:
        print(
            f"{i+1} -> Keyword: {candidate_keywords[i]['key']} ; Score: {candidate_keywords[i]['Skr']}")
"""


def write_to_file(content, folder_path, file_name):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    file_path = os.path.join(folder_path, file_name)

    with open(file_path, 'w', encoding='utf-8') as file:
        file.write(content)


input_folder_path = "./processed_data_noisy/"
output_folder_path = "./keywords_data_noisy/"


def convert_keyphrases_to_json(list_of_dicts, folder, filename):
    if not os.path.exists(folder):
        os.makedirs(folder)

    file_path = os.path.join(folder, filename)

    for i, item in enumerate(list_of_dicts):
        item["index"] = i

    reduced_list_of_dicts = [
        {"index": item["index"], "keyphrase": item["key"], "score": item["Skr"]} for item in list_of_dicts]

    json_data = json.dumps(reduced_list_of_dicts, indent=2, ensure_ascii=False)
    with open(file_path, 'w', encoding='utf-8') as json_file:
        json_file.write(json_data)


def iterate_through_txt_files(folder_path):
    txt_files = [f for f in os.listdir(folder_path) if f.endswith('.txt')]
    i = 0
    for txt_file in txt_files:
        file_path = os.path.join(folder_path, txt_file)
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
            sentences, chunks = pre_process(content, language)
            # print(sentences)
            term_processor = TermProcessor()
            result_terms = term_processor.process(sentences, chunks, w=2)

            ngrams = generate_ngrams(
                sentences, chunks, num_of_ngrams, stop_words=language)
            candidate_keywords = generate_candidatekeywords(ngrams)
            candidate_keywords = sorted(
                candidate_keywords, key=lambda x: x['Skr'], reverse=False)
            scoring(candidate_keywords, language, result_terms)

            threshold = 0.7
            Final_keywords = data_deduplication(candidate_keywords, threshold)
            Final_keywords = sorted(
                Final_keywords, key=lambda x: x['Skr'], reverse=False)
            file_name = f"keywords_{os.path.splitext(txt_file)[0]}.json"
            convert_keyphrases_to_json(
                Final_keywords, output_folder_path, file_name)
            i += 1


iterate_through_txt_files(input_folder_path)
