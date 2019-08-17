from xpinyin import Pinyin
translator=Pinyin()

def Chinese2Pinyin(chinese):
    chinese = chinese.replace(" ","")
    print(chinese)
    one_kw_pinyin=translator.get_pinyin(chinese, ' ',tone_marks='numbers').strip()
    return one_kw_pinyin
