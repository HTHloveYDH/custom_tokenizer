import re
from collections import Counter

class BytePairEncode:
    def __init__(self):
        self.ws_token = '_'
        self.unk_token = '<UNK>'

        self.corpus = {}
        self.word_count = {}
        self.vocab = Counter()

        self.id_tokens = {}
        self.token_ids = {}
        self.unk_id = 0  # 用于处理未识别的token

    def preprocess(self, line):
        return re.sub('\s+', ' ', line)

    def process_sentence(self, sentence):
        words = sentence.split()
        for word in words:
            word = self.ws_token + word
            if word not in self.corpus:
                self.corpus[word] = [ch for ch in word]
                self.word_count[word] = 1
            else:
                self.word_count[word] += 1

    def init_state(self, content):
        # 初始化 corpus 和 word_count
        for line in content:
            sentence = self.preprocess(line.strip())
            self.process_sentence(sentence)

        alphabet = {}
        for word, chrs in self.corpus.items():
            for ch in chrs:
                alphabet[ch] = alphabet.get(ch, 0) + self.word_count[word]
        self.vocab.update(alphabet)


    def _dump_init(self):
        print('-'*12 + " dump initial state " + "-"*12)
        print()
        print("--> dump corpus <--")
        for word, text in self.corpus.items():
            print(f"{word} => {text}")
        print()
        print("--> dump words=>counts <--")
        for word, count in self.word_count.items():
            print(f"{word} => {count}")
        print()
        print("--> dump vocab <--")
        for token,count in self.vocab.items():
            print(f"{token} => {count}")
        print("-" * 40)

    def gen_bigrams(self):
        bigram_counter = Counter()
        for word, text in self.corpus.items():
            for i in range(len(text) - 1):
                bigram = text[i] + text[i+1]
                bigram_counter[bigram] += self.word_count[word]
        return bigram_counter

    def update_vocab(self, symbol, count):
        if symbol in self.vocab:
            self.vocab[symbol] += count
        else:
            self.vocab[symbol] = count

    def merge_pair(self):
        """合并频率最高的字符对"""
        bigrams = self.gen_bigrams()
        if not bigrams:
            return
        top_bigram, top_count = bigrams.most_common(1)[0]
        print(f"=> top_bigram:{top_bigram}, top_count:{top_count}")
        if top_count == 1:
            return top_bigram, top_count    # 如果没有频率大于1的字符对，停止合并

        # 遍历每个词，合并最频繁的bigram
        for word, text in self.corpus.items():
            merged = False
            for i in range(len(text) - 1):
                if text[i] + text[i+1] == top_bigram:
                    self.update_vocab(text[i], -self.word_count[word])
                    self.update_vocab(text[i+1], -self.word_count[word])
                    text[i] = top_bigram  # 合并bigram
                    text[i+1] = ''  # 清除后一个字符
                    merged = True
            if merged:
                # 更新词库，去除空字符
                self.corpus[word] = [token for token in text if token]
        self.update_vocab(top_bigram, top_count)
        return top_bigram, top_count


    def assign_ids(self):

        # 为每个 token 分配唯一 ID
        idx = 32  # 从 32 开始，ID 0 作为 <UNK>
        self.token_ids[self.unk_token] = self.unk_id
        self.id_tokens[self.unk_id] = self.unk_token

        for token in sorted(self.vocab):
            self.token_ids[token] = idx
            self.id_tokens[idx] = token
            idx += 1


    def train(self, text, steps=None):
        self.init_state(text)  # 初始化属性
        self._dump_init()
        if steps is not None:
            for step in range(steps):
                print("=" * 12 + f" step:{step} " + "=" * 12)
                self.merge_pair()  # 合并最频繁的bigram
        else:
            step = 0
            print("=" * 12 + f" step:{step} " + "=" * 12)
            top_bigram, top_count = self.merge_pair()
            while top_count != 1:
                step += 1
                print("=" * 12 + f" step:{step} " + "=" * 12)
                top_bigram, top_count = self.merge_pair()


        # 训练完成后为词汇表分配 ID
        self.assign_ids()

    def _segment(self, text):
        tokens = list(text)
        while True:
            bigrams = [(tokens[i], tokens[i + 1]) for i in range(len(tokens) - 1)]
            pairs = [''.join(bigram) for bigram in bigrams]
            mergeable = [pair for pair in pairs if pair in self.vocab]
            if not mergeable:
                break
            # 找到频率最高的 bigram
            best_pair = max(mergeable, key=lambda pair: self.vocab[pair])
            # 合并 best_pair
            new_tokens = []
            i = 0
            while i < len(tokens):
                if i < len(tokens) - 1 and tokens[i] + tokens[i + 1] == best_pair:
                    new_tokens.append(best_pair)
                    i += 2  # 跳过合并的字符
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            tokens = new_tokens
        return tokens

    def encode(self, text):
        if not text:
            return
        text = self.preprocess(text)
        text = self.ws_token + re.sub(' ', self.ws_token, text.strip())
        seg_txt = self._segment(text)

        # 使用 token_ids 进行编码，如果不在词表中，则使用 <UNK> 的 ID
        seg_ids = [self.token_ids.get(token, self.unk_id) for token in seg_txt]
        return seg_txt, seg_ids

    def decode(self, ids):
        # 使用 id_tokens 进行解码
        text = "".join([self.id_tokens.get(idx, self.unk_token) for idx in ids]).replace(self.ws_token, ' ').strip()
        return text


class RainbowPrinter:
    def __init__(self):
        self.idx = 0
        # self.format_str = '\x1b[1;%s;40m%s\x1b[0m'  # 使用背景色 40-47 或前景色 30-37
        self.format_str = '\x1b[1;%sm%s\x1b[0m'

    def print_word(self, word):
        self.idx += 1
        if self.idx == 7:
            self.idx = 1
        print(self.format_str % (30 + self.idx, word), end=' ')

    def print_words(self, words):
        if isinstance(words, (list, tuple)):  # 简化 isinstance 的条件
            for token_word in words:
                self.print_word(token_word)
            print()  # 打印换行
        else:
            raise TypeError(f"Expected list or tuple, got {type(words).__name__}")


# content = ["hug "*10 + "pug "*5 + "pun "*12 + "bun "*4 + "hugs "*5]

content = [
    "这是OpenAI",
    "这是OpenAI 团队前一段时间放出来的预印版论文。 他们的目标是学习一个通用的表示，能够在大量任务上进行应用。",
    "这篇论文的亮点主要在于， 他们利用了Transformer网络代替了LSTM作为语言模型来更好地捕获长距离语言结构",
    "然后在进行具体任务有监督微调时，使用了模型作为附属任务训练目标。",
    "论文，亮点",

]

bpe = BytePairEncode()
bpe.train(content)

print(bpe.token_ids)

printer = RainbowPrinter()
# seg_txt, seg_ids = bpe.encode("")
# seg_txt, seg_ids = bpe.encode("这是OpenAI")
seg_txt, seg_ids = bpe.encode("论文的亮点是用语言模型完成对应的目标任务")
printer.print_words(seg_txt)
print(seg_ids)
# print(bpe.decode([48, 49, 56, 33, 46, 70, 64, 34, 176, 156, 109, 141, 154, 178, 149, 0, 0, 0, 135, 156, 159, 114]))