rank_to_device_mapping = {0: 7, 1: 0, 2: 64, 3: 65, 4: 66, 5: 67, 6: 68, 7: 69, 8: 5, 9: 6, 10: 70, 11: 71, 12: 72, 13: 73, 14: 74, 15: 75, 16: 1, 17: 2, 18: 3, 19: 4, 20: 76, 21: 77, 22: 78, 23: 79, 24: 40, 25: 41, 26: 42, 27: 43, 28: 44, 29: 45, 30: 46, 31: 47, 32: 24, 33: 25, 34: 26, 35: 27, 36: 28, 37: 29, 38: 30, 39: 31, 40: 32, 41: 33, 42: 34, 43: 35, 44: 36, 45: 37, 46: 38, 47: 39, 48: 8, 49: 9, 50: 10, 51: 11, 52: 12, 53: 13, 54: 14, 55: 15, 56: 16, 57: 17, 58: 18, 59: 19, 60: 20, 61: 21, 62: 22, 63: 23, 64: 48, 65: 49, 66: 50, 67: 51, 68: 52, 69: 53, 70: 54, 71: 55, 72: 56, 73: 57, 74: 58, 75: 59, 76: 60, 77: 61, 78: 62, 79: 63}
l_values = [[3, 7, 14, 28, 28], [16, 16, 16, 16, 16]]

ans = ""
for k, v in rank_to_device_mapping.items():
    ans += str(k) + ":" + str(v) + ","
    
print("{" + ans[:-1] + "}")

ans = ""
for y in l_values:
    for x in y:
        ans += str(x) + ","
    
print(ans[:-1])

'''
ans = ""
for k in range(64):
    ans += str(k) + ":" + str(k) + ","
    
print("{" + ans[:-1] + "}")
'''