# 输入
# 输入只有一行，包含一个字符串S(长度不会超过100)，代表整个句子，句子中只包含大小写的英文字母，每个单词之间有一个空格。
# 输出
# 输出句子S的平均重量V(四舍五入保留两位小数)。
def func(sentence):
    words = sentence.split()
    sum = 0
    for word in words:
        sum += len(word)
    return sum / len(words)







if __name__ == '__main__':
    sen = input()
    print('%.2f' % func(sen))