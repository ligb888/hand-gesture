def confirm_gesture(history):
    count_dict = {}
    last = history[-15:]
    for i, item in enumerate(last):
        if item[1] in count_dict.keys():
            count_dict[item[1]] = count_dict[item[1]] + 1
        else:
            count_dict[item[1]] = 1
    for key, item in enumerate(count_dict):
        if count_dict[item] >= 12:
            return item
    return ""


mode = confirm_gesture([[0, "kk"], [0, "kk"], [0, "kk"], [0, "kk"], [0, "kk"], [0, "kk"], [0, "kk"], [0, "kk"], [0, "kk"],
                        [0, "kk"], [0, "kk"], [0, "kk"], [0, "kk"], [0, "kk"], [0, "kk"], [0, "kk"], [0, "kk"], [0, "kk"],
                        [0, "kk"], [0, "kk"], [0, "kk"], [0, "kk"], [0, "kk"], [0, "kk"], [0, "kk"], [0, "kk"], [0, "kk"],
                        [0, "kk"], [0, "11"], [0, "11"], [0, "11"], [0, "11"]])
print(mode)
