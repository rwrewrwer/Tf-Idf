from ckip_transformers.nlp import CkipWordSegmenter, CkipPosTagger, CkipNerChunker #anaconda控制台輸入pip install ckip_transformer與pip install sklearn
from sklearn.feature_extraction.text import TfidfVectorizer#vscode安裝pip manager後搜尋ckip_transformer與sklearn-plus
from sklearn.metrics.pairwise import cosine_similarity
import time
start = time.time()
# 初始化驅動程序
ws_driver = CkipWordSegmenter(model="albert-base", device=-1)#若是NVIDIA顯卡請將-1改為0，速度將顯著提升
pos_driver = CkipPosTagger(model="albert-base", device=-1)#AMD顯卡需要做相關設定才能使用CUDA，若沒有設定請維持-1
ner_driver = CkipNerChunker(model="albert-base", device=-1)
#過濾涵式
def clean(sentence_ws, sentence_pos): 
    short_sentence = []
    stop_pos = set(['Nep', 'Nh', 'Nb'])  # 這 3 種詞性不保留
    for word_ws, word_pos in zip(sentence_ws, sentence_pos):
        if word_pos.startswith("V") or word_pos.startswith("N"):
            if word_pos not in stop_pos and len(word_ws) > 1:
                short_sentence.append(word_ws)
    return " ".join(short_sentence)

questions = []
answers = []

with open("在此放入檔案位置", "r+", encoding="utf8") as f: #放入欲使用之QA集
    lines = f.readlines()
    for line in lines:
        text = line.replace("\t", " ").replace("\n", "")
        texe = text.split(" ", 1)
        question = texe[0]  # 問題
        answer = texe[1]  # 答案
        questions.append(question)
        answers.append(answer)
#斷詞並且呼叫過濾涵式
ws = ws_driver(questions)
pos = pos_driver(ws)

corpus = []
for sentence_ws, sentence_pos in zip(ws, pos):
    short = clean(sentence_ws, sentence_pos)
    corpus.append(short)
#計算TF-IDF與各問題之向量
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(corpus)
end = time.time()
print("載入時間:"+str(end-start))
while True:
#使用者輸入句子，輸入q中斷
    user_input = input("請輸入您的句子（輸入q退出）：")
    start_q = time.time()
    if user_input == "q":
        break
#將輸入句子斷詞並過濾且計算TF-IDF與對比向量餘弦相似度
    user_input_ws = ws_driver([user_input])[0]
    user_input_pos = pos_driver([user_input_ws])[0]
    user_input_cleaned = clean(user_input_ws, user_input_pos)
    user_input_vector = vectorizer.transform([user_input_cleaned])

    similarities = cosine_similarity(user_input_vector, tfidf_matrix).flatten()
    most_similar_index = similarities.argsort()[-1:][::-1]  # 取相似度最高的句子的索引

    print("最接近的問題及答案：")
    for idx in most_similar_index:
        question = questions[idx]
        answer = answers[idx]
        similarity = similarities[idx]
        print("問題：", question)
        print("答案：", answer)
        print("相似度：", similarity)
        print("---")
        end_q = time.time()
        print("回答時間:"+str(end_q-start_q))