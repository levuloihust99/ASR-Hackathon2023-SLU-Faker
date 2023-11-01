from vietnam_number import w2n,w2n_single
import re

def myFunc(e):
    return e[0]

spoken_num = ['không','một','hai','ba','bốn','năm','sáu','bẩy','tám','chín','mười','bảy','tư']
add_num = ['mươi','tư','mốt','lăm','trăm','nghìn','triệu']

def find_all_text_number(text):
    single_words = text.split(" ")
    all_number = []
    i = 0
    while i < len(single_words):
        if single_words[i] in spoken_num:
            if single_words[i] == "năm":
                if (i > 0 and (single_words[i-1] == "tháng")) or (i > 1 and single_words[i-2] == "tháng"):
                    i = i + 1
                else:
                    start = j = i
                    number = ""
                    while j < len(single_words) and (single_words[j] in spoken_num or single_words[j] in add_num):
                        number += single_words[j]
                        number += " "
                        j = j + 1
                    end = j - 1
                    number = number.strip()
                    i = j
                    if number:
                        all_number.append([start,end,number])
            elif single_words[i] == "không":
                if (i > 0 and (single_words[i-1] in spoken_num)) or (i+1 < len(single_words) and single_words[i+1] in ["giờ","chiếc","cái"]) or (i+1 < len(single_words) and single_words[i+1] in add_num):
                    start = j = i
                    number = ""
                    while j < len(single_words) and (single_words[j] in spoken_num or single_words[j] in add_num):
                        number += single_words[j]
                        number += " "
                        j = j + 1
                    end = j - 1
                    number = number.strip()
                    i = j
                    if number:
                        all_number.append([start,end,number])
                else:
                    i = i + 1
            
            elif single_words[i] == "tư":
                if (i > 0 and (single_words[i-1] == 'tháng')) or (i+1 < len(single_words) and single_words[i+1] in spoken_num) or (i+1 < len(single_words) and single_words[i+1] in spoken_num):
                    start = j = i
                    number = ""
                    while j < len(single_words) and (single_words[j] in spoken_num or single_words[j] in add_num):
                        number += single_words[j]
                        number += " "
                        j = j + 1
                    end = j - 1
                    number = number.strip()
                    i = j
                    if number:
                        all_number.append([start,end,number])
                else:
                    i = i + 1
            else:
                start = j = i
                number = ""
                while j < len(single_words) and (single_words[j] in spoken_num or single_words[j] in add_num):
                    number += single_words[j]
                    number += " "
                    if single_words[j] == "tư":
                        j = j + 1
                        break
                    j = j + 1
                    
                end = j - 1
                number = number.strip()
                i = j
                if number:
                    all_number.append([start,end,number])
        else:
            i += 1
    return all_number

def word_2_number(all_numbers,text):
    if len(all_numbers):
        all_numbers.sort(key=myFunc)
        for item_ in all_numbers:
            number = item_[2]
            number = number.strip()
            try:
                norm_number = str(w2n_single(str(number)))
            except:
                norm_number = str(w2n(str(number)))
            text = re.sub(number, norm_number, text, 1)
    text = re.sub(" phần trăm","%",text)
    return text




