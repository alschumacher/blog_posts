from algorthmic_roman_num import generate_roman_numerals

roman_numerals = generate_roman_numerals(range(1, 4000000))
numeral_pairs = list(zip([str(i) for i in range(1, 4000000)], roman_numerals))

class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS"}
        self.n_words = 2

    def addSentence(self, sentence):
        for word in sentence:
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

arabic_numerals_lang = Lang('arabic_numerals')
roman_numerals_lang = Lang('roman_numerals')

for numeral in numeral_pairs:
    arabic, roman = numeral
    arabic_numerals_lang.addSentence(arabic)
    roman_numerals_lang.addSentence(roman)
